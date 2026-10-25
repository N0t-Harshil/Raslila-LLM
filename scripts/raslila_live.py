"""Raslila live: an always-on session you can talk to while it is thinking.

Usage:
    python scripts/raslila_live.py --checkpoint checkpoints/raslila_adaptive/raslila_step19999.pt
    python scripts/raslila_live.py --checkpoint ... --inbox data/live_inbox --offload

How it behaves, and why it is built this way:

- It does not exit when an answer is finished. It keeps the model loaded and
  the whole conversation in memory, and waits.
- While idle it uses no compute. The worker blocks on an input queue; after
  --idle-sleep seconds it also "rests": frees cached GPU memory and, with
  --offload, moves the weights to CPU so the GPU holds almost nothing. The next
  input wakes it up again, with the conversation still there.
- You can type at any time, including while it is generating. New input is
  not queued behind the current answer and does not abort the session: it is
  added to the context and generation carries on from there, so the model
  picks up what you added mid-thought.
- With --inbox DIR, any .txt file dropped into DIR (by you, another terminal or
  another program) is read in exactly like typed input, then renamed *.done.
- With an adaptive-depth checkpoint (max_think_cycles > 0) every token reports
  how many thinking cycles it used: easy tokens exit after 1, hard ones loop.

Commands (type them on their own line):
    /stop         stop the current answer (context is kept)
    /more [N]     keep generating N more tokens (default --tokens)
    /think N      cap thinking cycles at N (adaptive checkpoints only)
    /temp T       sampling temperature
    /reset        forget the conversation
    /status       context length, device, memory, thinking stats
    /save PATH    write the conversation to a text file
    /quit         end the session
Ctrl+C stops the current answer; it does not end the session.
"""
import argparse
import gc
import glob
import os
import queue
import sys
import threading
import time

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import torch
import yaml

from raslila.raslila import Raslila, RaslilaConfig

EOS = '<|endoftext|>'


def load_model(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    b3 = dict(raw['raslila'])
    train_cfg = raw.get('training') or {}
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'config' in ckpt:
        b3 = ckpt['config']
    else:
        b3['vocab_size'] = ckpt['model']['token_embedding.weight'].shape[0]
    model = Raslila(RaslilaConfig(**b3))
    model.load_state_dict(ckpt['model'])
    model.eval()
    window = int(args.window or train_cfg.get('seq_len') or model.config.max_seq_len)
    return model, window


class Live:
    def __init__(self, model, tok, window, args):
        self.model = model
        self.tok = tok
        self.window = window
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.inbox = queue.Queue()
        self.context = []                 # the whole conversation, as token ids
        self.budget = 0                   # tokens left to generate in this answer
        self.temperature = args.temperature
        self.think = None                 # None = model's own max_think_cycles
        self.resting = False
        self.last_active = time.time()
        self.alive = True
        self.eos_id = tok.token_to_id(EOS)
        self.answer_tokens = 0
        self.answer_cycles = 0
        self.answer_t0 = 0.0
        self.total_cycles = []

    # ---- output --------------------------------------------------------
    def say(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()

    # ---- resting / waking ---------------------------------------------
    def rest(self):
        if self.resting:
            return
        if self.args.offload and self.device.type == 'cuda':
            self.model.to('cpu')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.resting = True
        self.say('\n[resting -- {} tokens of conversation kept; type to wake]\n'.format(
            len(self.context)))

    def wake(self):
        if not self.resting:
            return
        if self.args.offload and self.device.type == 'cuda':
            self.model.to(self.device)
        self.resting = False

    # ---- input handling -----------------------------------------------
    def take(self, item):
        """Absorb one human input: a command or text to add to the context."""
        text = item.rstrip('\n')
        if not text.strip():
            return
        if text.startswith('/'):
            self.command(text)
            return
        generating = self.budget > 0
        ids = self.tok.encode(('\n' if self.context else '') + text + '\n').ids
        self.context.extend(ids)
        self.budget = self.args.tokens
        self.last_active = time.time()
        if generating:
            # Mid-answer input is folded into the running thought, not queued.
            self.say('\n[+ added your input, continuing]\n')
        else:
            self.answer_tokens = 0
            self.answer_cycles = 0
            self.answer_t0 = time.time()

    def command(self, text):
        parts = text.split()
        cmd, rest = parts[0].lower(), parts[1:]
        if cmd == '/quit':
            self.alive = False
        elif cmd == '/stop':
            if self.budget:
                self.finish_answer()
            self.budget = 0
        elif cmd == '/more':
            self.budget = int(rest[0]) if rest else self.args.tokens
            self.answer_tokens = 0
            self.answer_cycles = 0
            self.answer_t0 = time.time()
        elif cmd == '/think' and rest:
            self.think = max(1, int(rest[0]))
            self.say('[thinking cycles capped at {}]\n'.format(self.think))
        elif cmd == '/temp' and rest:
            self.temperature = max(1e-3, float(rest[0]))
            self.say('[temperature {}]\n'.format(self.temperature))
        elif cmd == '/reset':
            self.context = []
            self.budget = 0
            self.say('[conversation cleared]\n')
        elif cmd == '/status':
            self.status()
        elif cmd == '/save' and rest:
            with open(rest[0], 'w', encoding='utf-8') as f:
                f.write(self.tok.decode(self.context))
            self.say('[saved {} tokens to {}]\n'.format(len(self.context), rest[0]))
        else:
            self.say(__doc__.split('Commands')[1])

    def status(self):
        mem = ''
        if torch.cuda.is_available():
            mem = ', GPU memory {:.0f} MB allocated'.format(torch.cuda.memory_allocated() / 1e6)
        avg = (sum(self.total_cycles) / len(self.total_cycles)) if self.total_cycles else 0
        cap = self.think or self.model.config.max_think_cycles
        self.say('[context {} tokens (window {}), {}{}{}; thinking: avg {:.2f} cycles/token, cap {}]\n'.format(
            len(self.context), self.window,
            'resting' if self.resting else 'awake on ' + str(next(self.model.parameters()).device),
            mem, ', generating' if self.budget else '', avg, cap or 'fixed depth'))

    def poll_inbox_dir(self):
        d = self.args.inbox
        if not d:
            return
        for path in sorted(glob.glob(os.path.join(d, '*.txt'))):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                os.replace(path, path + '.done')
            except OSError:
                continue          # still being written; pick it up next poll
            self.say('\n[inbox: {}]\n'.format(os.path.basename(path)))
            self.inbox.put(text)

    # ---- generation ----------------------------------------------------
    def step(self):
        if not self.context:
            self.budget = 0
            return
        ids = torch.tensor([self.context[-self.window:]], dtype=torch.long, device=self.device)
        use_bf16 = self.device.type == 'cuda'
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            out = self.model(ids, think_cycles=self.think)
        last = out['logits'][0, -1].float()
        last[0] = float('-inf')                       # never sample <|pad|>
        if self.args.top_k > 0:
            kth = torch.topk(last, min(self.args.top_k, last.numel())).values[-1]
            last[last < kth] = float('-inf')
        nxt = int(torch.multinomial(torch.softmax(last / self.temperature, dim=-1), 1))
        cycles = out.get('cycles_used') or 0
        self.answer_cycles += cycles
        self.total_cycles.append(cycles)
        self.total_cycles = self.total_cycles[-10000:]
        if nxt == self.eos_id:
            self.finish_answer()
            self.budget = 0
            return
        self.context.append(nxt)
        self.answer_tokens += 1
        self.budget -= 1
        self.say(self.tok.decode([nxt]))
        if self.budget == 0:
            self.finish_answer()

    def finish_answer(self):
        dt = max(1e-6, time.time() - self.answer_t0)
        think = ''
        if self.answer_tokens and self.answer_cycles:
            think = ', {:.2f} thinking cycles/token'.format(self.answer_cycles / self.answer_tokens)
        self.say('\n[{} tokens in {:.1f}s{}]\n> '.format(self.answer_tokens, dt, think))
        self.last_active = time.time()

    # ---- the always-on loop -------------------------------------------
    def run(self):
        next_inbox_poll = 0.0
        while self.alive:
            now = time.time()
            if now >= next_inbox_poll:
                self.poll_inbox_dir()
                next_inbox_poll = now + 1.0
            # Drain everything the human has added so far. When there is work,
            # do not wait; when idle, block (zero compute) until input arrives.
            try:
                timeout = 0 if self.budget > 0 else 0.5
                item = self.inbox.get(timeout=timeout) if timeout else self.inbox.get_nowait()
                self.take(item)
                self.inbox.task_done()
                while True:
                    self.take(self.inbox.get_nowait())
                    self.inbox.task_done()
            except queue.Empty:
                pass
            if not self.alive:
                break
            if self.budget > 0:
                self.wake()
                self.step()
                self.last_active = time.time()
            elif not self.resting and time.time() - self.last_active > self.args.idle_sleep:
                self.rest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', default='config.yaml',
                    help='used only for training.seq_len and old checkpoints')
    ap.add_argument('--tokenizer', default='data/tokenizer/raslila_bpe.json')
    ap.add_argument('--tokens', type=int, default=200, help='max tokens per answer')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--window', type=int, default=None,
                    help='context window (default: training.seq_len)')
    ap.add_argument('--idle-sleep', type=float, default=60.0,
                    help='seconds idle before resting (frees GPU cache)')
    ap.add_argument('--offload', action='store_true',
                    help='while resting, move the weights to CPU as well')
    ap.add_argument('--inbox', default=None,
                    help='directory watched for .txt files to add as input')
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.inbox:
        os.makedirs(args.inbox, exist_ok=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    model, window = load_model(args)
    live = Live(model, tok, window, args)
    n = sum(p.numel() for p in model.parameters())
    mode = ('adaptive depth, up to {} thinking cycles'.format(model.config.max_think_cycles)
            if model.config.max_think_cycles > 0
            else 'fixed depth ({} sweeps)'.format(model.config.depth_sweeps))
    live.say('Raslila live: {:.1f}M params on {}, {}. Type anything; /help for commands.\n> '.format(
        n / 1e6, live.device, mode))

    worker = threading.Thread(target=live.run, daemon=True)
    worker.start()
    stdin_open = True
    while live.alive:
        if not stdin_open:
            # stdin closed (e.g. piped input ran out). With an inbox the session
            # stays up and keeps serving it; without one, finish the current
            # answer and then end.
            if not args.inbox and live.budget == 0 and live.inbox.unfinished_tasks == 0:
                live.alive = False
                break
            time.sleep(0.5)
            continue
        try:
            line = sys.stdin.readline()
            if line == '':
                stdin_open = False
                continue
            live.inbox.put(line)
            if line.strip().lower() == '/quit':
                break
        except KeyboardInterrupt:
            live.inbox.put('/stop')
    worker.join(timeout=5)
    live.say('\n[session ended]\n')


if __name__ == '__main__':
    main()
