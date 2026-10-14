"""Train Raslila on Colab TPU (single core) via torch_xla.

Same config.yaml, corpus, adaptive LR, logging and checkpoint format as
train_raslila.py. Checkpoints are saved with CPU tensors so they load back on
TPU, GPU, or CPU (generate_raslila.py included).

Colab setup:
    import torch; print(torch.__version__)   # e.g. 2.8.x
    !pip install -q "torch_xla[tpu]"         # must match torch major.minor;
                                             # if it errors, pin torch first:
                                             # !pip install -q torch==X.Y.* torch_xla==X.Y.*

Usage:
    python scripts/train_raslila_tpu.py --steps 3000         # from config.yaml
    python scripts/train_raslila_tpu.py --steps 8000 --resume ckpt/raslila_step2999.pt
"""
import sys
import os
import time
import argparse

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import torch
import torch.nn.functional as F
import numpy as np

# This script is TPU-only: select the PJRT TPU client before torch_xla loads.
os.environ.setdefault('PJRT_DEVICE', 'TPU')

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    torch_xla = None
    xm = None

from raslila.raslila import Raslila, RaslilaConfig
from train_raslila import (load_config, load_corpus, make_batch,
                           load_bpe, tokenize_corpus_bpe, legacy_cache_path,
                           TOKENIZER_FILE)


def save_checkpoint_cpu(model, step, out_dir, best_loss, plateau_streak, lr,
                        config_dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'raslila_step{}.pt'.format(step))
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({
        'model': model_state,
        'config': config_dict,
        'step': step,
        'best_loss': best_loss,
        'plateau_streak': plateau_streak,
        'lr': lr,
    }, path)
    print('Saved: {}'.format(path))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--lr-patience', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint .pt (model weights + step; optimizer restarts fresh)')
    parser.add_argument('--tok-mode', type=str, default=None,
                        choices=['bytes', 'bpe'],
                        help='tokenization (default: training.tok_mode)')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='BPE tokenizer json (default: training.tokenizer)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if xm is None:
        raise SystemExit('torch_xla is not installed. On Colab TPU run:\n'
                         '  !pip install -q "torch_xla[tpu]"  (must match torch major.minor)')

    raw, b3, tr = load_config(args.config)
    try:
        device = torch_xla.device()
    except Exception:
        device = xm.xla_device()
    print('Device: {}'.format(device))
    sys.stdout.flush()

    steps = args.steps if args.steps is not None else int(tr.get('max_steps', 1000))
    batch_size = args.batch_size if args.batch_size is not None else int(tr.get('batch_size', 8))
    seq_len = args.seq_len if args.seq_len is not None else int(tr.get('seq_len', 32))
    lr = args.lr if args.lr is not None else float(tr.get('learning_rate', 3e-4))
    grad_clip = float(tr.get('grad_clip', 1.0))
    out_dir = args.out_dir if args.out_dir is not None else tr.get('out_dir', 'checkpoints/raslila')
    save_every = int(tr.get('save_every', 1000))
    data_dir = args.data_dir if args.data_dir is not None else tr.get('data_dir')
    tok_mode = args.tok_mode if args.tok_mode is not None else tr.get('tok_mode', 'bytes')
    tokenizer = args.tokenizer if args.tokenizer is not None else tr.get(
        'tokenizer', TOKENIZER_FILE)

    # HuggingFace corpus (the default), same path as train_raslila.py. The TPU
    # trainer is single-process, so there is no rank-0/barrier dance: build the
    # cache here, then load it.
    ds_resolved = None
    ds_cfg = raw.get('dataset')
    if ds_cfg:
        from raslila.hf_dataset import resolve as resolve_dataset, ensure_dataset
        ds_resolved = resolve_dataset(ds_cfg)
        tokenizer = ds_resolved['tokenizer']
        tok_mode = 'bpe'
        data_dir = None
        ensure_dataset(ds_resolved)

    model_cfg = RaslilaConfig(**b3)
    model = Raslila(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print('Raslila nodes: {}'.format(model.n_nodes))
    print('Raslila params: {:,} ({:.1f}M)'.format(n_params, n_params / 1e6))
    print('Steps: {}, batch: {}, seq_len: {}, lr: {}'.format(steps, batch_size, seq_len, lr))
    sys.stdout.flush()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=float(tr.get('weight_decay', 0.01)))
    warmup = max(1, int(tr.get('warmup_steps', 200)))
    if args.warmup is not None:
        warmup = args.warmup
    adaptive_lr = bool(tr.get('adaptive_lr', True))
    lr_factor = float(tr.get('lr_factor', 0.5))
    lr_patience = int(tr.get('lr_patience', 200))
    if args.lr_patience is not None:
        lr_patience = args.lr_patience
    lr_threshold = float(tr.get('lr_threshold', 1e-3))
    # Hard floor for the plateau decay, same fix as train_raslila.py. Without
    # it the decay is geometric and unbounded: a long run walks the LR to zero
    # and spends its last quarter frozen. The 15M GPU run hit 4.7e-6 this way.
    min_lr = float(tr.get('min_lr', 0.0) or 0.0)
    best_loss = float('inf')
    plateau_streak = 0

    def cur_lr():
        return optimizer.param_groups[0]['lr']

    def set_lr(new_lr):
        for g in optimizer.param_groups:
            g['lr'] = float(new_lr)

    # Resume: model weights + schedule state (fresh optimizer on XLA).
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        model.to(device)
        if ckpt.get('lr') is not None:
            set_lr(ckpt['lr'])
        best_loss = float(ckpt.get('best_loss', best_loss))
        plateau_streak = int(ckpt.get('plateau_streak', plateau_streak))
        start_step = int(ckpt.get('step', 0)) + 1
        print('Resumed from {} at step {} (lr {:.2e})'.format(
            args.resume, start_step - 1, cur_lr()))
        sys.stdout.flush()

    corpus = None
    if ds_resolved is not None:
        corpus = np.load(ds_resolved['cache'])
        print('Corpus (BPE): {:,} tokens from {} [{}]'.format(
            len(corpus), ds_resolved['name'], ds_resolved['cache']))
        if len(corpus) and int(corpus.max()) >= model_cfg.vocab_size:
            raise SystemExit(
                'Token id {} exceeds raslila.vocab_size={}: the corpus was '
                'tokenized with a larger vocabulary than the model has '
                'embeddings for.'.format(int(corpus.max()), model_cfg.vocab_size))
        sys.stdout.flush()
    elif data_dir:
        if tok_mode == 'bpe':
            bpe = load_bpe(tokenizer)
            corpus = tokenize_corpus_bpe(
                bpe, data_dir, cache_path=legacy_cache_path(data_dir))
            print('Corpus (BPE): {:,} tokens from {}'.format(len(corpus), data_dir))
            sys.stdout.flush()
        else:
            corpus = load_corpus(data_dir)
            print('Corpus (bytes): {} tokens from {}'.format(len(corpus), data_dir))
            sys.stdout.flush()

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'train_log.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8') as cf:
            cf.write('step,lr,loss,alpha,beta,grad_norm,time_s\n')

    mref = model
    mtp_w = float(getattr(model_cfg, 'mtp_weight', 0.0) or 0.0)
    t0 = time.perf_counter()
    window_loss = 0.0
    window_size = 0
    grad_norm = 0.0
    vocab = model_cfg.vocab_size

    print('Compiling XLA graph (first step takes minutes)...')
    sys.stdout.flush()
    for step in range(start_step, start_step + steps):
        gstep = step + 1
        if corpus is not None:
            input_ids, target_ids = make_batch(corpus, batch_size, seq_len, step, steps)
        else:
            input_ids = torch.randint(1, min(vocab, 4096), (batch_size, seq_len))
            target_ids = torch.randint(1, min(vocab, 4096), (batch_size, seq_len))
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids)
        logits = out['logits']
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               target_ids.view(-1), ignore_index=0)
        if mtp_w > 0 and len(mref.mtp_heads) > 0:
            final_repr = out['final_repr']
            aux = []
            for k, head in enumerate(mref.mtp_heads, start=2):
                if final_repr.size(1) <= k:
                    break
                aux.append(F.cross_entropy(
                    head(final_repr[:, :-k]).reshape(-1, logits.size(-1)),
                    target_ids[:, k:].reshape(-1), ignore_index=0))
            if aux:
                loss = loss + mtp_w * torch.stack(aux).mean()
        loss.backward()
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        xm.optimizer_step(optimizer)
        xm.mark_step()

        window_loss += float(loss.detach().to('cpu'))
        window_size += 1
        last_step = (gstep == start_step + steps)

        if step % 50 == 0 or last_step:
            win_avg = window_loss / max(1, window_size)
            # Warmup ramp, then plateau-aware decay (same policy as GPU trainer).
            if step < warmup:
                set_lr(lr * min(1.0, gstep / warmup))
            elif adaptive_lr:
                if win_avg < best_loss - lr_threshold * best_loss:
                    best_loss = win_avg
                    plateau_streak = 0
                else:
                    plateau_streak += 50
                    if plateau_streak >= lr_patience:
                        set_lr(max(cur_lr() * lr_factor, min_lr))
                        plateau_streak = 0
            dt = time.perf_counter() - t0
            gn = float(grad_norm)
            print('Step {:5d} | lr {:.2e} | loss {:.4f} | alpha {:+.3f} | beta {:+.3f} | grad_norm {:.4f} | {:.1f}s'.format(
                gstep - 1, cur_lr(), win_avg,
                float(model.alpha.data), float(model.beta.data), gn, dt))
            sys.stdout.flush()
            with open(csv_path, 'a', encoding='utf-8') as cf:
                cf.write('{},{:.6e},{:.6f},{:.6f},{:.6f},{:.6f},{:.2f}\n'.format(
                    gstep - 1, cur_lr(), win_avg,
                    float(model.alpha.data), float(model.beta.data), gn, dt))
            window_loss = 0.0
            window_size = 0

        if gstep % save_every == 0 or last_step:
            save_checkpoint_cpu(model, gstep - 1, out_dir,
                                best_loss, plateau_streak, cur_lr(),
                                __import__('dataclasses').asdict(model.config))

    print('\nDone in {:.1f}s'.format(time.perf_counter() - t0))


if __name__ == '__main__':
    main()
