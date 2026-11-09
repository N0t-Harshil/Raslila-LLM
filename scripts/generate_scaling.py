"""Sample text from a scaling-study run (Raslila v2 or baseline).

    python scripts/generate_scaling.py --run raslila_125m --prompt "def fibonacci(n):"
    python scripts/generate_scaling.py --run raslila_125m --think 4   # allow 4 thinking cycles

For Raslila with thinking loops it also reports how many cycles each token used
(it stops looping as soon as its prediction has settled).
"""
import argparse
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
os.chdir(_ROOT)

import torch

from train_scaling import build_model
from raslila.data_mix import load_data_config


def load_run(run, device):
    ck = torch.load(os.path.join('runs', run, 'ckpt.pt'), map_location='cpu', weights_only=False)
    model = build_model(ck['model_config'])
    model.load_state_dict(ck['model'])
    return model.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--prompt', default='The most important idea in science is')
    ap.add_argument('--tokens', type=int, default=200)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--think', type=int, default=None)
    ap.add_argument('--window', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(load_data_config('configs/data.yaml')['tokenizer'])
    eos = tok.token_to_id('<|endoftext|>')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck = load_run(args.run, device)
    adaptive = getattr(model.config, 'max_think_cycles', 0) > 0
    looped = bool(getattr(model, 'looped', False))     # v4: thinks until its answer settles
    cap = (args.think or max(model.config.eval_loops)) if looped else None
    print('[{} | step {} | {:.2f}B tokens trained]'.format(args.run, ck['step'], ck['tokens_seen'] / 1e9))
    ids = tok.encode(args.prompt).ids
    sys.stdout.write(args.prompt)
    cycles = []
    for _ in range(args.tokens):
        x = torch.tensor([ids[-args.window:]], device=device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            if looped:
                out = model(x, loops=cap, halt=True)
            else:
                out = model(x, think_cycles=args.think) if adaptive else model(x)
        logits = out['logits'][0, -1].float()
        logits[0] = float('-inf')
        if args.top_k:
            kth = torch.topk(logits, args.top_k).values[-1]
            logits[logits < kth] = float('-inf')
        nxt = int(torch.multinomial(torch.softmax(logits / args.temperature, -1), 1))
        if out.get('cycles_used'):
            cycles.append(out['cycles_used'])
        if nxt == eos:
            break
        ids.append(nxt)
        sys.stdout.write(tok.decode([nxt]))
        sys.stdout.flush()
    print()
    if cycles:
        hist = {c: cycles.count(c) for c in sorted(set(cycles))}
        print('[thinking cycles per token: avg {:.2f}; {}]'.format(
            sum(cycles) / len(cycles), ', '.join('{}: {}'.format(k, v) for k, v in hist.items())))


if __name__ == '__main__':
    main()
