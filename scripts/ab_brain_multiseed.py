"""Run the brain-mechanism A/B across several seeds to establish the noise floor.

A single-seed delta of ±0.01 nats may just be seed variance. This runs each
variant on multiple seeds and reports mean ± std, so we only believe a
difference that survives the spread.
"""
import os
import sys
import importlib.util

import numpy as np

spec = importlib.util.spec_from_file_location(
    'ab', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'ab_brain_mechanisms.py'))
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

SEEDS = [11, 22, 33, 44, 55, 66]
# Focused set: the first 3-seed pass showed inhibition and the combined set as
# the only significant effects, so spend the extra seeds confirming those rather
# than re-measuring variants that were flat.
VARIANTS = [
    ('baseline (feedforward, additive)', {}),
    ('+ inhibition (signed edges)', {'inhibit': True}),
    ('+ all three', {'depth_sweeps': 2, 'inhibit': True, 'state_dim': 16}),
]


def main():
    ab.STEPS = int(os.environ.get('AB_STEPS', '200'))
    data = ab.corpus()
    print('{} steps x {} seeds ({}), BPE vocab {}'.format(
        ab.STEPS, len(SEEDS), ab.VOCAB, ab.VOCAB))
    print()
    print('{:<34s} {:>18s} {:>12s}'.format('variant', 'loss mean±std', 'vs base'))
    print('-' * 68)

    results = {}
    for name, kw in VARIANTS:
        vals = []
        for s in SEEDS:
            loss, _, _, _ = ab.run(kw, data, seed=s)
            vals.append(loss)
        results[name] = vals
        print('{:<34s} {:.4f} ± {:.4f}'.format(name, np.mean(vals), np.std(vals)))

    base = np.array(results[VARIANTS[0][0]])
    print()
    print('paired vs baseline (same seed each time), mean ± std of the difference:')
    for name, _ in VARIANTS[1:]:
        d = np.array(results[name]) - base
        # std of the paired difference is the right error bar: seeds share data order
        print('{:<34s} {:+.4f} ± {:.4f}   {}'.format(
            name, d.mean(), d.std(),
            'BETTER' if d.mean() < -2 * d.std() / np.sqrt(len(d))
            else 'worse' if d.mean() > 2 * d.std() / np.sqrt(len(d))
            else 'no significant change'))


if __name__ == '__main__':
    main()