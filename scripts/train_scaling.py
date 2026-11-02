"""One trainer for the scaling study: Raslila v2 (grid.py) and the baseline.

    # one GPU
    python scripts/train_scaling.py --config configs/scaling/raslila_125m.yaml
    # all 4 GPUs on one run (DDP)
    torchrun --standalone --nproc_per_node=4 scripts/train_scaling.py --config configs/scaling/raslila_1b.yaml
    # CPU smoke test, random tokens, no data needed
    python scripts/train_scaling.py --config configs/scaling/smoke_raslila.yaml --synthetic

Everything that matters for a fair comparison is identical across families:
data (same token cache, same sampling), tokenizer, optimiser, schedule, batch
size in tokens, token budget per size tier, and evaluation windows.

Outputs in out_dir:
    ckpt.pt        latest checkpoint (auto-resumed; written atomically)
    log.csv        one row per log interval
    metrics.json   params, tokens, FLOPs, throughput, val-loss history, final
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
os.chdir(_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from raslila.grid import RaslilaGrid, GridConfig
from raslila.baseline import Baseline, BaselineConfig

PEAK = {'A100': 312e12, 'H100': 989e12, 'H200': 989e12, 'B200': 2250e12, 'L40': 181e12,
        'A10': 125e12, '4090': 165e12, '3090': 71e12}


def peak_flops(device):
    if device.type != 'cuda':
        return None
    name = torch.cuda.get_device_name(device)
    for k, v in PEAK.items():
        if k in name:
            return v
    return 312e12


def build_model(mcfg):
    mcfg = dict(mcfg)
    family = mcfg.pop('family')
    if family == 'raslila':
        return RaslilaGrid(GridConfig(**mcfg))
    if family == 'baseline':
        return Baseline(BaselineConfig(**mcfg))
    raise SystemExit('model.family must be raslila or baseline, got {!r}'.format(family))


def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    return total, total - model.tok_emb.weight.numel()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    for key in ('model', 'train'):
        if key not in cfg:
            raise SystemExit('{} has no `{}:` section'.format(path, key))
    return cfg


def main():
    ap = argparse.ArgumentParser(description='Raslila scaling-study trainer')
    ap.add_argument('--config', required=True)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--synthetic', action='store_true', help='random tokens (smoke test)')
    ap.add_argument('--max-steps', type=int, default=None, help='stop early (smoke tests)')
    ap.add_argument('--micro-batch', type=int, default=None, help='override train.micro_batch')
    ap.add_argument('--compile', action='store_true', help='torch.compile the model')
    ap.add_argument('--fresh', action='store_true', help='ignore an existing checkpoint')
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr = cfg['train']
    out_dir = args.out_dir or cfg.get('out_dir') or os.path.join(
        'runs', os.path.splitext(os.path.basename(args.config))[0])

    # ---- distributed ------------------------------------------------------
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    ddp = world > 1
    if ddp:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(local)
    device = torch.device('cuda', local) if torch.cuda.is_available() else torch.device('cpu')
    main_proc = rank == 0
    log = print if main_proc else (lambda *a, **k: None)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed = int(tr.get('seed', 1234))
    torch.manual_seed(seed + rank)

    # ---- data ---------------------------------------------------------------
    seq_len = int(tr['seq_len'])
    if args.synthetic:
        vocab = int(cfg['model']['vocab_size'])
        rng0 = np.random.default_rng(0)
        tokens = rng0.integers(2, vocab, size=4_000_000, dtype=np.int64).astype(np.uint16)
        val_tokens = 200_000
        data_info = {'synthetic': True}
    else:
        from raslila.data_mix import load_data_config, open_tokens
        ds = load_data_config(cfg.get('data', 'configs/data.yaml'))
        tokens, data_info = open_tokens(ds)
        val_tokens = ds['val_tokens']
        if data_info['vocab_size'] > int(cfg['model']['vocab_size']):
            raise SystemExit('tokenizer vocab {} > model vocab_size {}'.format(
                data_info['vocab_size'], cfg['model']['vocab_size']))
    n_tok = len(tokens)
    if n_tok < val_tokens + 10 * seq_len:
        raise SystemExit('token cache too small: {:,}'.format(n_tok))

    def batch_at(rng, b):
        offs = rng.integers(val_tokens, n_tok - seq_len - 1, size=b)
        arr = np.stack([np.asarray(tokens[o:o + seq_len + 1], dtype=np.int64) for o in offs])
        t = torch.from_numpy(arr)
        if device.type == 'cuda':
            t = t.pin_memory().to(device, non_blocking=True)
        return t[:, :-1], t[:, 1:]

    # ---- model ----------------------------------------------------------------
    model = build_model(cfg['model']).to(device)
    family = model.family
    total_params, nonembed = param_counts(model)
    adaptive = family == 'raslila' and model.config.max_think_cycles > 0
    max_cycles = model.config.max_think_cycles if adaptive else 0

    micro = int(args.micro_batch or tr['micro_batch'])
    batch_tokens = int(float(tr['batch_tokens']))
    accum = max(1, round(batch_tokens / (micro * seq_len * world)))
    eff_batch_tokens = accum * micro * seq_len * world
    budget = int(float(tr['tokens'])) if 'tokens' in tr else int(float(tr.get('tokens_per_param', 20)) * total_params)
    steps = max(1, budget // eff_batch_tokens)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    warmup = int(tr.get('warmup_steps') or max(50, int(0.02 * steps)))
    warmup = min(warmup, max(1, steps // 5))
    lr_max = float(tr['lr'])
    lr_min = lr_max * float(tr.get('min_lr_frac', 0.1))

    def lr_at(step):
        if step < warmup:
            return lr_max * (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, steps - warmup))
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))

    # Weight decay on weight matrices only. Norm gains, positional roles and
    # biases are excluded BY NAME: Raslila stores per-block norm gains as
    # [N, 1, D] tensors, so a `dim >= 2` rule alone would wrongly decay them
    # towards zero (shrinking every block's input).
    def no_wd(name, p):
        return p.dim() < 2 or 'norm' in name.split('.')[-1] or name.endswith('role')
    decay = [p for n, p in model.named_parameters() if not no_wd(n, p)]
    no_decay = [p for n, p in model.named_parameters() if no_wd(n, p)]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': float(tr.get('weight_decay', 0.1))},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=lr_max, betas=(0.9, 0.95), eps=1e-8,
                            fused=device.type == 'cuda')

    # ---- resume -------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    start_step, tokens_seen, flops_seen, train_seconds = 0, 0, 0.0, 0.0
    history = []
    if os.path.exists(ckpt_path) and not args.fresh:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_step = ck['step']
        tokens_seen, flops_seen = ck['tokens_seen'], ck['flops_seen']
        train_seconds = ck.get('train_seconds', 0.0)
        history = ck.get('history', [])
        log('Resumed {} at step {:,} ({:,} tokens)'.format(ckpt_path, start_step, tokens_seen))
    if start_step >= steps:
        log('Already finished ({} >= {} steps).'.format(start_step, steps))

    raw = model
    if args.compile:
        model = torch.compile(model)
    if ddp:
        find_unused = family == 'raslila' and (adaptive or model_depth_sweeps(raw) == 1)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=find_unused)

    use_amp = device.type == 'cuda'
    peak = peak_flops(device)
    grad_clip = float(tr.get('grad_clip', 1.0))
    # z-loss (PaLM): a tiny penalty on log Z keeps the output logits from
    # drifting large, a known cause of late-run loss spikes. Same for both
    # families; validation loss is plain cross-entropy.
    z_loss = float(tr.get('z_loss', 1e-4))
    log_every = int(tr.get('log_every', 10))
    eval_every = int(tr.get('eval_every', 500))
    eval_batches = int(tr.get('eval_batches', 40))
    save_every_s = 60 * float(tr.get('save_every_minutes', 30))

    log('=' * 78)
    log('{} | {}'.format(family.upper(), args.config))
    log('params {:,} total, {:,} non-embedding'.format(total_params, nonembed))
    if family == 'raslila':
        c = raw.config
        log('grid {} x {} x {} = {} blocks, d_node {}, node_depth {}, {}'.format(
            *c.grid_shape, raw.L * raw.N, c.d_node, c.node_depth,
            'adaptive 1..{} cycles'.format(c.max_think_cycles) if adaptive
            else '{} sweeps'.format(c.depth_sweeps)))
    log('tokens {:,} = {:,} steps x {:,} tokens/step (micro {} x seq {} x accum {} x gpus {})'.format(
        steps * eff_batch_tokens, steps, eff_batch_tokens, micro, seq_len, accum, world))
    log('lr {} -> {} cosine, warmup {} | corpus {:,} tokens'.format(lr_max, lr_min, warmup, n_tok))
    log('=' * 78)

    # ---- eval ---------------------------------------------------------------------
    def evaluate():
        raw.eval()
        rng = np.random.default_rng(999)                # same windows for every model
        n_ex = max_cycles if adaptive else 1
        sums = [0.0] * n_ex
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            for _ in range(eval_batches):
                offs = rng.integers(0, val_tokens - seq_len - 1, size=micro)
                arr = np.stack([np.asarray(tokens[o:o + seq_len + 1], dtype=np.int64) for o in offs])
                t = torch.from_numpy(arr).to(device)
                out = raw(t[:, :-1], think_cycles=max_cycles if adaptive else None, halt=False) \
                    if adaptive else raw(t[:, :-1])
                for k, ex in enumerate(out['exit_logits']):
                    sums[k] += F.cross_entropy(ex.float().reshape(-1, ex.size(-1)),
                                               t[:, 1:].reshape(-1), ignore_index=0).item()
        raw.train()
        return [s / eval_batches for s in sums]

    def save():
        if not main_proc:
            return
        tmp = ckpt_path + '.tmp'
        torch.save({'model': raw.state_dict(), 'opt': opt.state_dict(), 'step': step,
                    'tokens_seen': tokens_seen, 'flops_seen': flops_seen,
                    'train_seconds': train_seconds, 'history': history,
                    'model_config': cfg['model'], 'train_config': tr}, tmp)
        os.replace(tmp, ckpt_path)

    def write_metrics(final=False):
        if not main_proc:
            return
        last = history[-1] if history else None
        m = {'family': family, 'config': args.config, 'model': cfg['model'],
             'params_total': total_params, 'params_nonembed': nonembed,
             'tokens_seen': tokens_seen, 'flops_seen': flops_seen,
             'train_seconds': train_seconds, 'gpus': world,
             'tokens_per_second': tokens_seen / max(1e-9, train_seconds),
             'mfu': (flops_seen / max(1e-9, train_seconds)) / (peak * world) if peak else None,
             'steps_done': step, 'steps_total': steps, 'finished': final,
             'val_loss': last['val_loss'] if last else None,
             'val_loss_per_cycle': last['val_per_cycle'] if last else None,
             'history': history, 'data': data_info if not args.synthetic else 'synthetic'}
        with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(m, f, indent=2)

    csv_path = os.path.join(out_dir, 'log.csv')
    if main_proc and not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'tokens', 'flops', 'loss', 'lr', 'grad_norm',
                                    'tok_per_s', 'mfu', 'val_loss'])

    # ---- train --------------------------------------------------------------------
    model.train()
    step = start_step
    skipped = 0
    win_loss, win_n, win_tok, win_t0 = 0.0, 0, 0, time.time()
    last_save = time.time()
    while step < steps:
        t_step = time.time()
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        cycles = random.Random(seed * 7919 + step).randint(1, max_cycles) if adaptive else None
        step_loss = 0.0
        step_flops = 0.0
        for micro_i in range(accum):
            rng = np.random.default_rng([seed, step, rank, micro_i])
            x, y = batch_at(rng, micro)
            sync = (micro_i == accum - 1) or not ddp
            ctx = model.no_sync() if (ddp and not sync) else _null()
            with ctx:
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    out = model(x, think_cycles=cycles) if adaptive else model(x)
                    exits = out['exit_logits']
                    ce, zl = [], []
                    for e in exits:
                        ef = e.float().reshape(-1, e.size(-1))
                        ce.append(F.cross_entropy(ef, y.reshape(-1), ignore_index=0))
                        if z_loss > 0:
                            zl.append(torch.logsumexp(ef, dim=-1).pow(2).mean())
                    task_loss = torch.stack(ce).mean()
                    loss = task_loss + (z_loss * torch.stack(zl).mean() if zl else 0.0)
                (loss / accum).backward()
            step_loss += task_loss.item() / accum          # logged loss = cross-entropy only
            if family == 'raslila':
                step_flops += raw.flops_per_token(seq_len, sweeps=out['sweeps'],
                                                  exits=len(exits)) * x.numel()
            else:
                step_flops += raw.flops_per_token(seq_len) * x.numel()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(gn):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            log('  step {}: non-finite grad norm, update skipped ({} so far)'.format(step, skipped))
            if skipped > 50:
                raise SystemExit('Diverged: >50 non-finite steps. Lower train.lr and resume.')
        else:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t_step
        step += 1
        train_seconds += dt
        tokens_seen += eff_batch_tokens
        flops_seen += step_flops * world
        win_loss += step_loss
        win_n += 1
        win_tok += eff_batch_tokens

        val = None
        if step % eval_every == 0 or step == steps:
            per = evaluate() if main_proc else None
            if main_proc:
                val = per[-1]
                history.append({'step': step, 'tokens': tokens_seen, 'flops': flops_seen,
                                'val_loss': val, 'val_per_cycle': per,
                                'train_loss': win_loss / max(1, win_n)})
                write_metrics(final=(step == steps))
            if ddp:
                torch.distributed.barrier()

        if step % log_every == 0 or val is not None or step == steps:
            el = time.time() - win_t0
            tps = win_tok / max(1e-9, el)
            mfu = (flops_seen / max(1e-9, train_seconds)) / (peak * world) if peak else None
            eta = (steps - step) * (train_seconds / max(1, step - start_step)) / 3600 \
                if step > start_step else 0
            msg = 'step {:>6}/{} | loss {:.4f} | lr {:.2e} | gn {:.2f} | {:,.0f} tok/s{} | {:.2f}B tok | ETA {:.1f}h'.format(
                step, steps, win_loss / max(1, win_n), lr_at(step - 1), float(gn), tps,
                ' | MFU {:.1%}'.format(mfu) if mfu else '', tokens_seen / 1e9, eta)
            if val is not None:
                msg += ' | VAL {:.4f}'.format(val)
                if adaptive:
                    msg += ' (per cycle: ' + ' '.join('{:.3f}'.format(v) for v in per) + ')'
            log(msg)
            if main_proc:
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([step, tokens_seen, flops_seen, win_loss / max(1, win_n),
                                            lr_at(step - 1), float(gn), tps, mfu, val])
            win_loss, win_n, win_tok, win_t0 = 0.0, 0, 0, time.time()

        if time.time() - last_save > save_every_s or step == steps:
            save()
            last_save = time.time()

    write_metrics(final=step >= steps)
    if main_proc and history:
        log('DONE. final val loss {:.4f} | {:,} tokens | {:.3e} FLOPs | {:.1f} GPU-hours'.format(
            history[-1]['val_loss'], tokens_seen, flops_seen, train_seconds * world / 3600))
    if ddp:
        torch.distributed.destroy_process_group()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def model_depth_sweeps(m):
    return int(getattr(m.config, 'depth_sweeps', 2))


if __name__ == '__main__':
    main()
