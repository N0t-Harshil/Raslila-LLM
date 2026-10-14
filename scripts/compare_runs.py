"""Compare two training runs and apply the decision rule.

Usage:
    python scripts/compare_runs.py ck_bidir ck_feed
    python scripts/compare_runs.py ck_bidir ck_feed --min-steps 20000

Prints a comparison of the tail loss and a verdict. The verdict is mechanical
on purpose: the decision rule is fixed BEFORE anyone looks at the numbers, so
the result cannot be rationalised after the fact.
"""
import argparse
import csv
import glob
import os
import sys


def read_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, '*.csv')))
    if not files:
        return None
    rows = list(csv.DictReader(open(files[0])))
    if not rows:
        return None
    losses = [float(r['loss']) for r in rows]
    return {
        'csv': files[0],
        'steps': int(float(rows[-1]['step'])),
        'n_rows': len(rows),
        'last5': losses[-5:],
        'tail': sum(losses[-5:]) / len(losses[-5:]),
        'best': min(losses),
        'curve': losses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_a', help='candidate run dir (e.g. ck_bidir)')
    ap.add_argument('run_b', help='baseline run dir (e.g. ck_feed)')
    ap.add_argument('--tail', type=int, default=5,
                    help='average the last N logged losses (default 5)')
    ap.add_argument('--min-steps', type=int, default=0,
                    help='fail if either run is shorter than this')
    args = ap.parse_args()

    a = read_run(args.run_a)
    b = read_run(args.run_b)

    missing = [r for r, v in ((args.run_a, a), (args.run_b, b)) if v is None]
    if missing:
        print('No CSV found in: {}'.format(', '.join(missing)))
        print('Did the run finish? Logs are written as <out-dir>/train_log.csv')
        return 1

    for name, r in ((args.run_a, a), (args.run_b, b)):
        t = r['last5'][-args.tail:]
        print('{:<16s} steps={:<7d} tail({})={:.4f}  best={:.4f}'.format(
            name, r['steps'], len(t), sum(t) / len(t), r['best']))
    print()

    if args.min_steps and (a['steps'] < args.min_steps or b['steps'] < args.min_steps):
        print('INCOMPLETE: both runs must reach {} steps before comparing.'.format(
            args.min_steps))
        return 1

    if a['steps'] != b['steps']:
        print('WARNING: unequal length ({} vs {}). Compare only at equal steps.'
              .format(a['steps'], b['steps']))

    ta = sum(a['last5'][-args.tail:]) / len(a['last5'][-args.tail:])
    tb = sum(b['last5'][-args.tail:]) / len(b['last5'][-args.tail:])
    delta = ta - tb

    print('candidate tail : {:.4f}'.format(ta))
    print('baseline tail  : {:.4f}'.format(tb))
    print('difference     : {:+.4f} nats (negative = candidate better)'.format(delta))
    print()

    # The rule, fixed in advance: bidirectional flow costs 2x compute, so it is
    # only justified by a clear win. A tie or a loss does not justify it.
    if delta < -0.01:
        print('VERDICT: KEEP depth_sweeps=2 -- candidate is better.')
    elif delta > 0.01:
        print('VERDICT: REVERT to depth_sweeps=1 -- baseline is better.')
        print('The 2x compute cost is not justified. Record this as a negative result.')
    else:
        print('VERDICT: TIE (within +/-0.01 nats).')
        print('2x compute bought no measurable gain -> REVERT to depth_sweeps=1,')
        print('or extend both runs before deciding. Do not keep the 2x cost on a tie.')
    return 0


if __name__ == '__main__':
    sys.exit(main())