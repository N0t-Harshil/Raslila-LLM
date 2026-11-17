"""Adaptive thinking: does each token need all its loops? MEASURED on a trained run.

    python scripts/adaptive_eval.py --run raslila_150m --run raslila_150m_lean
    python run.py adaptive --run raslila_150m

Fixed loops: every token runs the same number of loops (what the report shows).
Adaptive: every token stops on its own once its state stops changing
(RaslilaCortex.adaptive_forward). For each threshold: validation loss, average
loops actually used per token, how many tokens stopped after 2, 3, 4, ... loops,
and inference FLOPs per token at that average. No training needed.

Writes runs/<run>/adaptive.json.
"""
import argparse
import json
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
os.chdir(_ROOT)

import numpy as np
import torch
import torch.nn.functional as F


def val_windows(n, seq, synthetic, vocab, seed=0):
    rng = np.random.default_rng(seed)
    if synthetic:
        return rng.integers(2, vocab, size=(n, seq + 1))
    from raslila.data_mix import load_data_config, open_tokens
    ds = load_data_config('configs/data.yaml')
    toks, _ = open_tokens(ds)
    offs = rng.integers(0, ds['val_tokens'] - seq - 2, size=n)
    return np.stack([np.asarray(toks[o:o + seq + 1], dtype=np.int64) for o in offs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True)
    ap.add_argument('--windows', type=int, default=64, help='held-out windows of --seq tokens')
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--cap', type=int, default=16, help='max loops for any token')
    ap.add_argument('--thresholds', default='0.5,0.3,0.2,0.1,0.05,0.02,0.01,0.005')
    ap.add_argument('--fixed', default='1,2,3,4,8')
    ap.add_argument('--synthetic', action='store_true', help='random tokens (testing only)')
    args = ap.parse_args()
    from benchmark import load_run
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda')
    ths = [float(x) for x in args.thresholds.split(',')]
    fixed = [int(x) for x in args.fixed.split(',')]

    for run in args.run:
        model, ck = load_run(run, device)
        if not getattr(model, 'looped', False):
            print('{}: not a looped (v4) model, skipped'.format(run))
            continue
        arr = torch.from_numpy(val_windows(args.windows, args.seq, args.synthetic, model.config.vocab_size))
        batches = [arr[i:i + args.batch].to(device) for i in range(0, len(arr), args.batch)]

        def nll(logits, y):
            return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum').item()

        n_tok = arr.size(0) * args.seq
        fixed_res = {}
        with torch.no_grad(), torch.autocast(**amp):
            tot = {L: 0.0 for L in fixed}
            for b in batches:
                out = model(b[:, :-1], exit_at=fixed)
                for L, lg in zip(out['exit_loops'], out['exit_logits']):
                    tot[L] += nll(lg, b[:, 1:])
        for L in fixed:
            fixed_res[L] = {'val_loss': tot[L] / n_tok, 'mean_loops': float(L),
                            'inference_flops': model.inference_flops_per_token(args.seq, L)}

        adapt = []
        for th in ths:
            loss, hist = 0.0, np.zeros(args.cap + 1, dtype=np.int64)
            with torch.autocast(**amp):
                for b in batches:
                    out = model.adaptive_forward(b[:, :-1], threshold=th, cap=args.cap)
                    loss += nll(out['logits'], b[:, 1:])
                    hist += np.bincount(out['loops_used'].reshape(-1).cpu().numpy(), minlength=args.cap + 1)[:args.cap + 1]
            mean_l = float((hist * np.arange(args.cap + 1)).sum() / hist.sum())
            adapt.append({'threshold': th, 'val_loss': loss / n_tok, 'mean_loops': mean_l,
                          'max_loops': int(np.nonzero(hist)[0].max()),
                          'inference_flops': model.inference_flops_per_token(args.seq, mean_l),
                          'tokens_by_loops': {str(i): int(c) for i, c in enumerate(hist) if c}})

        print('=' * 78)
        print('ADAPTIVE THINKING {} (step {}), {} held-out tokens, cap {}'.format(run, ck['step'], n_tok, args.cap))
        print('{:<22} {:>9} {:>11} {:>13}   {}'.format('mode', 'val loss', 'mean loops', 'GFLOPs/token', 'tokens stopping after N loops'))
        for L in fixed:
            r = fixed_res[L]
            print('{:<22} {:>9.4f} {:>11.2f} {:>13.3f}'.format('fixed {} loops'.format(L), r['val_loss'], L, r['inference_flops'] / 1e9))
        for r in adapt:
            tot_t = sum(r['tokens_by_loops'].values())
            dist = '  '.join('{}:{:.0%}'.format(k, v / tot_t) for k, v in r['tokens_by_loops'].items() if v / tot_t >= 0.005)
            print('{:<22} {:>9.4f} {:>11.2f} {:>13.3f}   {}'.format('adaptive th={:g}'.format(r['threshold']), r['val_loss'],
                                                               r['mean_loops'], r['inference_flops'] / 1e9, dist))
        ref = fixed_res[max(fixed)]['val_loss']
        ok = [r for r in adapt if r['val_loss'] <= ref + 0.002]
        if ok:
            best = min(ok, key=lambda r: r['mean_loops'])
            print('-> same quality as {} fixed loops (within 0.002) with {:.2f} loops per token on average (th={:g})'.format(
                max(fixed), best['mean_loops'], best['threshold']))
        else:
            print('-> no threshold kept the quality of {} fixed loops within 0.002'.format(max(fixed)))
        res = {'run': run, 'step': ck['step'], 'tokens': n_tok, 'cap': args.cap,
               'fixed': {str(k): v for k, v in fixed_res.items()}, 'adaptive': adapt}
        with open(os.path.join('runs', run, 'adaptive.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)
        print('  -> runs/{}/adaptive.json'.format(run))
        del model
    return 0


if __name__ == '__main__':
    sys.exit(main())
