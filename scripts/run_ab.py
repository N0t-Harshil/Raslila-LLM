"""Run the bidirectional A/B: config.yaml (arm A) then the feedforward control.

The whole point of this pair is that it isolates ONE variable -- `raslila.depth_sweeps`
(2 = down-then-up, 1 = feedforward only). Everything else must match, or the
comparison is invalid. This script refuses to launch if the two configs differ by
anything else, because a drifted A/B that looks like a clean experiment is worse
than no experiment: it produced a confident 3-seed "win" that a 6-seed rerun
erased.

Arms run SEQUENTIALLY by default, each owning all 4 GPUs. `--parallel` runs them
concurrently on 2 GPUs each instead -- faster wall-clock, but the two arms then
compete for memory and the step times are no longer comparable.

    python scripts/run_ab.py --steps 20000
    python scripts/run_ab.py --steps 20000 --nproc 4
    python scripts/run_ab.py --steps 20000 --parallel      # 2 GPUs per arm
    python scripts/run_ab.py --compare-only                # just print the table
"""
import argparse
import csv
import os
import subprocess
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

ARM_A = 'config.yaml'
ARM_B = 'config_baseline_feedforward.yaml'


def _load(path):
    import yaml
    with open(os.path.join(_ROOT, path), encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_pair():
    """Abort unless the two arms differ by exactly `raslila.depth_sweeps`."""
    a, b = _load(ARM_A), _load(ARM_B)
    ra, rb = dict(a['raslila']), dict(b['raslila'])

    diffs = {k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k)}
    top = {k for k in set(a) | set(b) if k != 'raslila' and a.get(k) != b.get(k)}

    if top:
        raise SystemExit(
            'ABORT: the A/B pair differs outside the `raslila:` section: {}\n'
            'That is no longer a controlled comparison of depth_sweeps.'
            .format(sorted(top)))
    if diffs != {'depth_sweeps'}:
        raise SystemExit(
            'ABORT: the A/B pair differs by {}, not just depth_sweeps.\n'
            'Regenerate {} from {} so the two stay matched.'.format(
                sorted(diffs), ARM_B, ARM_A))

    va, vb = ra['depth_sweeps'], rb['depth_sweeps']
    if (va, vb) != (2, 1):
        raise SystemExit(
            'ABORT: expected depth_sweeps 2 vs 1, got {} vs {}.'.format(va, vb))

    pa, pb = ra.get('vocab_size'), rb.get('vocab_size')
    print('A/B pair verified:')
    print('  {}  depth_sweeps={}  vocab={}'.format(ARM_A, va, pa))
    print('  {}  depth_sweeps={}  vocab={}'.format(ARM_B, vb, pb))
    print('  differing keys: exactly one (raslila.depth_sweeps)\n')
    return a, b


def _build_cmd(cfg, out_dir, steps, nproc, fsdp):
    script = os.path.join(_ROOT, 'scripts', 'train_raslila.py')
    tail = ['--config', cfg, '--out-dir', out_dir]
    if steps:
        tail += ['--steps', str(steps)]
    if fsdp:
        tail.append('--fsdp')
    if nproc > 1:
        return (['torchrun', '--standalone',
                 '--nproc_per_node={}'.format(nproc)] + [script] + tail)
    return [sys.executable, script] + tail


def run_arm(cfg, out_dir, steps, nproc, fsdp):
    cmd = _build_cmd(cfg, out_dir, steps, nproc, fsdp)
    print('>>> ' + ' '.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=_ROOT)


def _final_loss(out_dir):
    path = os.path.join(_ROOT, out_dir, 'train_log.csv')
    if not os.path.exists(path):
        return None, 0
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('loss'):
                rows.append(r)
    if not rows:
        return None, 0
    return float(rows[-1]['loss']), len(rows)


def main():
    ap = argparse.ArgumentParser(description='Run the depth_sweeps A/B.')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--nproc', type=int, default=4)
    ap.add_argument('--parallel', action='store_true',
                    help='run both arms at once on nproc/2 GPUs each')
    ap.add_argument('--fsdp', action='store_true',
                    help='pass --fsdp through (needed for config_20b.yaml)')
    ap.add_argument('--compare-only', action='store_true',
                    help='do not train; just print the final-loss comparison')
    args = ap.parse_args()

    a, b = check_pair()

    out_a = 'checkpoints/raslila_ab_a'
    out_b = 'checkpoints/raslila_ab_b'

    if args.compare_only:
        pass
    elif args.parallel:
        half = max(1, args.nproc // 2)
        print('running BOTH arms concurrently on {} GPUs each\n'.format(half))
        pa = subprocess.Popen(
            _build_cmd(ARM_A, out_a, args.steps, half, args.fsdp), cwd=_ROOT)
        pb = subprocess.Popen(
            _build_cmd(ARM_B, out_b, args.steps, half, args.fsdp), cwd=_ROOT)
        ra, rb = pa.wait(), pb.wait()
        if ra or rb:
            raise SystemExit('arm failed: A={} B={}'.format(ra, rb))
    else:
        print('running arm A ({}) on {} GPUs\n'.format(ARM_A, args.nproc))
        rc = run_arm(ARM_A, out_a, args.steps, args.nproc, args.fsdp)
        if rc:
            raise SystemExit('arm A failed with exit code {}'.format(rc))
        print('\nrunning arm B ({}) on {} GPUs\n'.format(ARM_B, args.nproc))
        rc = run_arm(ARM_B, out_b, args.steps, args.nproc, args.fsdp)
        if rc:
            raise SystemExit('arm B failed with exit code {}'.format(rc))

    la, na = _final_loss(out_a)
    lb, nb = _final_loss(out_b)
    print('\n' + '=' * 62)
    print('A/B result  (lower loss wins; arm A costs 2x compute)')
    print('=' * 62)
    print('{:<34} {:>10} {:>8}'.format('arm', 'steps', 'loss'))
    print('{:<34} {:>10} {:>8}'.format(
        'A  depth_sweeps=2  (bidir)', na, 'n/a' if la is None else '{:.4f}'.format(la)))
    print('{:<34} {:>10} {:>8}'.format(
        'B  depth_sweeps=1  (fwd)', nb, 'n/a' if lb is None else '{:.4f}'.format(lb)))
    if la is not None and lb is not None:
        d = lb - la
        verdict = ('A wins by {:.4f} nats'.format(d) if d > 0
                   else 'B wins by {:.4f} nats'.format(-d) if d < 0 else 'tie')
        print('\nverdict: {}'.format(verdict))
        print('decision rule (fixed in advance): keep depth_sweeps=2 ONLY if A')
        print('reaches a LOWER loss at equal steps. It costs 2x compute.')
    else:
        print('\nno train_log.csv rows found -- check the out_dirs above.')


if __name__ == '__main__':
    main()
