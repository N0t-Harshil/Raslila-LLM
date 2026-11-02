"""Speed test: training throughput of Raslila v2 vs the baseline, per GPU.

    python scripts/speed_test.py                      # 125m pair
    python scripts/speed_test.py --tier 350m --compile
    python scripts/speed_test.py --configs configs/scaling/raslila_1b.yaml

For each config it finds the largest micro-batch that fits (starting from the
config's value, doubling until out-of-memory), then times real training steps
(forward + backward + AdamW, bf16) on random tokens. Reports tokens/s, MFU
(useful FLOPs / GPU peak) and peak memory, and writes runs/speed_test.json.

This is the first go/no-go number of the study: if Raslila cannot get within a
few times the baseline's tokens/s at equal parameters, it cannot win on cost.
"""
import argparse
import gc
import json
import os
import sys
import time

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
os.chdir(_ROOT)

import torch
import torch.nn.functional as F
import yaml

from train_scaling import build_model, param_counts, peak_flops


def bench(mcfg, micro, seq, steps, cycles, compile_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(mcfg).to(device)
    raw = model
    if compile_:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == 'cuda')
    vocab = mcfg['vocab_size']
    amp = device.type == 'cuda'
    if amp:
        torch.cuda.reset_peak_memory_stats()
    times, flops = [], 0.0
    for i in range(steps + 3):
        x = torch.randint(2, vocab, (micro, seq + 1), device=device)
        if amp:
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            out = model(x[:, :-1], think_cycles=cycles) if cycles else model(x[:, :-1])
            loss = torch.stack([F.cross_entropy(e.float().reshape(-1, vocab), x[:, 1:].reshape(-1))
                                for e in out['exit_logits']]).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if amp:
            torch.cuda.synchronize()
        if i >= 3:                                    # skip warm-up / compile steps
            times.append(time.time() - t0)
            if raw.family == 'raslila':
                flops += raw.flops_per_token(seq, sweeps=out['sweeps'], exits=len(out['exit_logits'])) * micro * seq
            else:
                flops += raw.flops_per_token(seq) * micro * seq
    sec = sum(times)
    total, nonembed = param_counts(raw)
    peak = peak_flops(device)
    res = {'micro': micro, 'tok_s': micro * seq * len(times) / sec,
           'mfu': (flops / sec) / peak if peak else None,
           'mem_gb': torch.cuda.max_memory_allocated() / 1e9 if amp else None,
           'params': total, 'flops_per_token': flops / (micro * seq * len(times))}
    del model, raw, opt, out, loss
    gc.collect()
    if amp:
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default='125m')
    ap.add_argument('--configs', nargs='*', default=None)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--seq', type=int, default=None)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--no-search', action='store_true', help='use the config micro_batch only')
    args = ap.parse_args()
    configs = args.configs or ['configs/scaling/baseline_{}.yaml'.format(args.tier),
                               'configs/scaling/raslila_{}.yaml'.format(args.tier)]
    if not torch.cuda.is_available():
        print('WARNING: no GPU -- CPU numbers say nothing about A100 speed.')
    rows = []
    for path in configs:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        m, tr = cfg['model'], cfg['train']
        seq = args.seq or int(tr['seq_len'])
        cyc_list = [None]
        if m['family'] == 'raslila' and int(m.get('max_think_cycles', 0)) > 0:
            cyc_list = list(range(1, int(m['max_think_cycles']) + 1))
        for cycles in cyc_list:
            micro, best = int(tr['micro_batch']), None
            while True:
                try:
                    r = bench(m, micro, seq, args.steps, cycles, args.compile)
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if best is None and micro > 1:
                        micro //= 2
                        continue
                    break
                best = r if (best is None or r['tok_s'] > best['tok_s']) else best
                if args.no_search or not torch.cuda.is_available() or micro >= 256:
                    break
                micro *= 2
            if best is None:
                print('{}: does not fit even at micro_batch 1'.format(path))
                continue
            best.update(config=path, family=m['family'], cycles=cycles, seq=seq)
            rows.append(best)
            print('{:<40} cycles {:<4} micro {:>3} | {:>9,.0f} tok/s | MFU {} | mem {} | {:.1f}M params'.format(
                path, str(cycles or '-'), best['micro'], best['tok_s'],
                '{:.1%}'.format(best['mfu']) if best['mfu'] else 'n/a',
                '{:.1f} GB'.format(best['mem_gb']) if best['mem_gb'] else 'n/a', best['params'] / 1e6))
    base = [r for r in rows if r['family'] == 'baseline']
    if base:
        b = base[0]['tok_s']
        print('\nRelative to the baseline (tokens/s, same tier):')
        for r in rows:
            print('  {:<40} cycles {:<4} {:>6.2f}x slower'.format(
                r['config'], str(r['cycles'] or '-'), b / r['tok_s']))
    os.makedirs('runs', exist_ok=True)
    with open('runs/speed_test.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print('\nsaved runs/speed_test.json')


if __name__ == '__main__':
    main()
