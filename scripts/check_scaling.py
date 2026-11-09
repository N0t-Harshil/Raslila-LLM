"""Validate the scaling-study configs before spending GPU time.

    python scripts/check_scaling.py

For every tier: both configs build (on the meta device, no memory used),
parameter counts are within 25% of each other, and everything that must be
identical for a fair comparison IS identical (data, tokens, seq_len,
batch_tokens, lr, schedule, seed). Prints params, FLOPs/token and the
estimated GPU-hours at 40% MFU on an A100.
"""
import glob
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
os.chdir(_ROOT)

import torch
import yaml

from train_scaling import build_model, param_counts

MUST_MATCH = ('tokens', 'seq_len', 'batch_tokens', 'lr', 'min_lr_frac', 'weight_decay',
              'grad_clip', 'seed', 'eval_batches')
A100_EFFECTIVE = 312e12 * 0.40


def main():
    errors = []
    tiers = sorted({os.path.basename(p).split('_', 1)[1][:-5]
                    for p in glob.glob('configs/scaling/*_*.yaml') if 'smoke' not in p},
                   key=lambda t: float(t[:-1]) * (1e3 if t.endswith('b') else 1))
    print('{:<6} {:<9} {:>9} {:>11} {:>12} {:>9} {:>12}'.format(
        'tier', 'family', 'params', 'non-embed', 'GFLOP/token', 'tokens', 'A100-hours*'))
    for tier in tiers:
        cfgs = {}
        for fam in ('baseline', 'raslila'):
            path = 'configs/scaling/{}_{}.yaml'.format(fam, tier)
            if not os.path.exists(path):
                errors.append('missing ' + path)
                continue
            with open(path, 'r', encoding='utf-8') as f:
                cfgs[fam] = yaml.safe_load(f)
        if len(cfgs) < 2:
            continue
        counts = {}
        for fam, cfg in cfgs.items():
            if cfg['model'].get('family') != fam:
                errors.append('{} {}: model.family is {!r}'.format(tier, fam, cfg['model'].get('family')))
            with torch.device('meta'):
                m = build_model(cfg['model'])
            total, ne = param_counts(m)
            counts[fam] = total
            seq = int(cfg['train']['seq_len'])
            if getattr(m, 'looped', False):
                fpt = m.expected_flops_per_token(seq)
            elif fam == 'raslila' and m.config.max_think_cycles > 0:
                # training samples 1..max cycles uniformly: average sweeps and exits
                k = m.config.max_think_cycles
                fpt = sum(m.flops_per_token(seq, sweeps=m.sweeps_for_cycles(c), exits=c)
                          for c in range(1, k + 1)) / k
            else:
                fpt = m.flops_per_token(seq)
            tok = float(cfg['train']['tokens'])
            print('{:<6} {:<9} {:>8.1f}M {:>10.1f}M {:>12.2f} {:>8.1f}B {:>12,.0f}'.format(
                tier, fam, total / 1e6, ne / 1e6, fpt / 1e9, tok / 1e9, fpt * tok / A100_EFFECTIVE / 3600))
            if cfg.get('data') != cfgs['baseline'].get('data'):
                errors.append('{}: data config differs between families'.format(tier))
        for key in MUST_MATCH:
            a = cfgs['baseline']['train'].get(key)
            b = cfgs['raslila']['train'].get(key)
            if a != b:
                errors.append('{}: train.{} differs (baseline {} vs raslila {})'.format(tier, key, a, b))
        ratio = counts['raslila'] / counts['baseline']
        if not 0.75 <= ratio <= 1.25:
            errors.append('{}: parameter counts too far apart ({:.2f}x)'.format(tier, ratio))
    print('* estimated at 40% MFU on one A100 (real MFU is measured by the speed test);\n'
          '  divide by the number of GPUs for wall-clock time.')
    if not os.path.exists('configs/data.yaml'):
        errors.append('configs/data.yaml missing')
    else:
        from raslila.data_mix import load_data_config
        ds = load_data_config('configs/data.yaml')
        print('\ndata: {:.1f}B tokens, vocab {}, sources: {}'.format(
            ds['max_tokens'] / 1e9, ds['vocab_size'],
            ', '.join('{} {:.0%}'.format(s['name'], s['weight']) for s in ds['sources'])))
        for tier in tiers:
            path = 'configs/scaling/baseline_{}.yaml'.format(tier)
            with open(path, 'r', encoding='utf-8') as f:
                tok = float(yaml.safe_load(f)['train']['tokens'])
            if tok > ds['max_tokens'] - ds['val_tokens']:
                print('  note: tier {} trains on {:.1f}B tokens > corpus {:.1f}B -> repeats data'.format(
                    tier, tok / 1e9, ds['max_tokens'] / 1e9))
    if errors:
        print('\nFAILED:')
        for e in errors:
            print('  - ' + e)
        return 1
    print('\nAll scaling configs OK.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
