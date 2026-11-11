"""Brain-likeness of a trained model -- MEASURED, not claimed.

    python scripts/brain_metrics.py --run raslila_150m --run baseline_150m
    python run.py brain --run raslila_150m --run baseline_150m

Two measurements from the papers in docs/10:

1. Dimensionality hierarchy (HRM, Wang et al. 2025, sec. 4). In the cortex,
   higher areas use higher-dimensional activity; the participation ratio
   PR = (sum eig)^2 / sum eig^2 of the activity covariance rises along the
   hierarchy (mouse cortex: top/bottom ~2.25x; HRM's two modules: ~3x). Here:
   PR of token representations after every stage -- Raslila v4: prelude, each
   thinking loop, coda; baseline: each layer.

2. Spatial clustering in the 3D cube (TopoLM, Rathi et al. 2025). Do
   neighbouring neurons in the cube respond alike? spatial r = Pearson r
   between the activity correlation of each unit pair and 1/(distance+1).
   Controls: the same units randomly re-placed in the cube (~0), and the
   baseline laid out in the same cube shape.

Writes runs/<run>/brain.json. Uses held-out tokens from the token cache
(--synthetic: random tokens, for testing only).
"""
import argparse
import json
import math
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
os.chdir(_ROOT)

import numpy as np
import torch

from raslila.nn_common import rope_tables


def participation_ratio(x):
    x = x.double()
    x = x - x.mean(0)
    ev = torch.linalg.eigvalsh((x.t() @ x) / max(1, x.size(0) - 1)).clamp_min(0)
    return float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12))


def cube_coords(shape):
    g = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), -1)
    return g.reshape(-1, 3)


def spatial_r(x, shape, rng):
    """TopoLM's smoothness statistic: Pearson r between the activity correlation
    of every unit pair and their closeness 1/(cube distance + 1). 0 = no spatial
    organisation, > 0 = nearby neurons respond alike. Returns (model, shuffled control)."""
    x = x.double()
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    c = ((x.t() @ x) / x.size(0)).cpu().numpy()
    coords = cube_coords(shape)
    iu = np.triu_indices(len(coords), 1)
    close = 1.0 / (np.abs(coords[:, None, :] - coords[None, :, :]).max(-1) + 1.0)

    def r(cm):
        a, b = cm[iu], close[iu]
        return float(np.corrcoef(a, b)[0, 1])
    perm = rng.permutation(len(coords))
    return r(c), r(c[np.ix_(perm, perm)])


def default_shape(d):
    for a in range(int(round(d ** (1 / 3))), 0, -1):
        if d % a == 0:
            rest = d // a
            for b in range(int(math.sqrt(rest)), 0, -1):
                if rest % b == 0:
                    return [a, b, rest // b]
    return [1, 1, d]


@torch.no_grad()
def stages(model, idx):
    """Token representations after every stage, as (name, [N, D]) pairs."""
    out = []
    if getattr(model, 'looped', False):
        cfg = model.config
        cos, sin = rope_tables(idx.size(1), cfg.d_model // cfg.n_heads, cfg.rope_theta, idx.device)
        e = model._run(model.prelude, model.tok_emb(idx), cos, sin)
        out.append(('prelude', e))
        g = torch.Generator(device='cpu').manual_seed(1234)
        s = torch.randn(e.shape, generator=g).to(e.device, e.dtype) * cfg.init_state_std
        for r in range(1, max(cfg.eval_loops + [cfg.test_loops]) + 1):
            s = model._loop(s, e, cos, sin)
            out.append(('loop {}'.format(r), s))
        out.append(('coda', model._run(model.coda, s, cos, sin)))
    else:
        cfg = model.config
        cos, sin = rope_tables(idx.size(1), cfg.d_model // cfg.n_heads, cfg.rope_theta, idx.device)
        x = model.tok_emb(idx)
        for i, blk in enumerate(model.blocks):
            x = blk(x, cos, sin)
            out.append(('layer {}'.format(i + 1), x))
    return [(n, t.reshape(-1, t.size(-1)).float()) for n, t in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True)
    ap.add_argument('--windows', type=int, default=8, help='held-out windows of --seq tokens')
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--tokens', type=int, default=4096, help='tokens sampled for the statistics')
    ap.add_argument('--synthetic', action='store_true', help='random tokens (testing only)')
    args = ap.parse_args()
    from benchmark import load_run
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for run in args.run:
        model, ck = load_run(run, device)
        vocab = model.config.vocab_size
        rng = np.random.default_rng(0)
        if args.synthetic:
            arr = rng.integers(2, vocab, size=(args.windows, args.seq))
        else:
            from raslila.data_mix import load_data_config, open_tokens
            ds = load_data_config('configs/data.yaml')
            toks, _ = open_tokens(ds)
            offs = rng.integers(0, ds['val_tokens'] - args.seq - 1, size=args.windows)
            arr = np.stack([np.asarray(toks[o:o + args.seq], dtype=np.int64) for o in offs])
        idx = torch.from_numpy(np.asarray(arr, dtype=np.int64)).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            st = stages(model, idx)
        pick = torch.from_numpy(rng.permutation(st[0][1].size(0))[:args.tokens]).to(device)
        shape = (getattr(model.config, 'cube_shape', None) or default_shape(st[0][1].size(1)))
        res = {'run': run, 'step': ck['step'], 'cube_shape': shape, 'stages': []}
        print('=' * 78)
        print('BRAIN METRICS {} (step {}), cube {}'.format(run, ck['step'], 'x'.join(map(str, shape))))
        print('{:<10} {:>8} {:>16} {:>18}'.format('stage', 'PR', 'spatial r', 'shuffled control'))
        for name, x in st:
            xs = x[pick]
            pr = participation_ratio(xs)
            sr, ctrl = spatial_r(xs, shape, rng)
            res['stages'].append({'stage': name, 'participation_ratio': pr,
                                  'spatial_r': sr, 'spatial_r_shuffled': ctrl})
            print('{:<10} {:>8.1f} {:>16.3f} {:>18.3f}'.format(name, pr, sr, ctrl))
        prs = [s['participation_ratio'] for s in res['stages']]
        res['pr_last_over_first'] = prs[-1] / max(1e-9, prs[0])
        ratio = res['pr_last_over_first']
        best_sr = max(s['spatial_r'] for s in res['stages'])
        noise = max(abs(s['spatial_r_shuffled']) for s in res['stages'])
        res['hierarchy_brain_like'] = ratio >= 1.5
        res['spatial_brain_like'] = best_sr > max(0.02, 3 * noise)
        print('PR last/first stage: {:.2f}x -> {} (brain-like = rises with depth; mouse cortex ~2.25x, HRM ~3x)'.format(
            ratio, 'brain-like' if res['hierarchy_brain_like'] else 'NOT brain-like'))
        print('best spatial r {:.3f} vs shuffled noise {:.3f} -> {} 3D clustering'.format(
            best_sr, noise, 'brain-like' if res['spatial_brain_like'] else 'NO'))
        with open(os.path.join('runs', run, 'brain.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)
        print('  -> runs/{}/brain.json'.format(run))
        del model
    return 0


if __name__ == '__main__':
    sys.exit(main())
