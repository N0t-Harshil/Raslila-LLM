"""Scaling report: Raslila v2 vs the baseline, from every finished run.

    python scripts/scaling_report.py            # reads runs/*/metrics.json

Prints a table, fits a power law  loss = a * C^-b  per family (C = training
FLOPs actually spent), and says, at the compute of the largest run, how much
compute the other family would need for the same loss (the "compute
multiplier"). Writes runs/scaling_report.md and runs/scaling.png.

HOW TO READ IT. One architecture beats another at scale only if its curve is
below the other's at equal compute AND stays below as compute grows (equal or
steeper slope). A single point means nothing; three or more sizes per family
are needed before the fit is worth showing anyone. The extrapolation to 10B+
is a projection, labelled as such -- it is how labs decide whether a large run
is worth paying for, not a measured result.
"""
import glob
import json
import math
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(_ROOT)


def load_runs(pattern='runs/*/metrics.json'):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        if m.get('data') == 'synthetic' or not m.get('val_loss'):
            continue
        m['run'] = os.path.basename(os.path.dirname(path))
        runs.append(m)
    return runs


def fit(points):
    """Least squares on log L = log a - b log C. Returns (a, b) or None."""
    if len(points) < 2:
        return None
    xs = [math.log(c) for c, _ in points]
    ys = [math.log(l) for _, l in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return math.exp(my - slope * mx), -slope


def main():
    runs = load_runs()
    if not runs:
        print('No finished runs with real data under runs/. Train something first:\n'
              '  python run.py scale --tier 50m')
        return 1
    lines = ['# Raslila v2 vs baseline -- scaling report', '',
             '| run | family | params | tokens | train FLOPs | val loss | loss per cycle | tok/s/GPU | MFU | GPU-h | done |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in sorted(runs, key=lambda r: (r['family'], r['params_total'])):
        per = m.get('val_loss_per_cycle') or []
        lines.append('| {} | {} | {:.0f}M | {:.2f}B | {:.2e} | **{:.4f}** | {} | {:,.0f} | {} | {:.1f} | {} |'.format(
            m['run'], m['family'], m['params_total'] / 1e6, m['tokens_seen'] / 1e9, m['flops_seen'],
            m['val_loss'], ' / '.join('{:.3f}'.format(v) for v in per) if len(per) > 1 else '-',
            m['tokens_per_second'] / max(1, m['gpus']),
            '{:.1%}'.format(m['mfu']) if m.get('mfu') else 'n/a',
            m['train_seconds'] * m['gpus'] / 3600, 'yes' if m.get('finished') else 'partial'))

    fits = {}
    for fam in ('baseline', 'raslila'):
        pts = [(m['flops_seen'], m['val_loss']) for m in runs if m['family'] == fam and m.get('finished')]
        fits[fam] = (fit(pts), len(pts))
    lines += ['', '## Fit: loss = a * C^-b (finished runs only)', '']
    for fam, (f, n) in fits.items():
        if f:
            lines.append('- **{}**: a = {:.3f}, b = {:.4f} ({} runs){}'.format(
                fam, f[0], f[1], n, '  <- fewer than 3 points, not reliable' if n < 3 else ''))
        else:
            lines.append('- **{}**: needs at least 2 finished runs ({} so far)'.format(fam, n))

    fb, fr = fits['baseline'][0], fits['raslila'][0]
    if fb and fr:
        cmax = max(m['flops_seen'] for m in runs)
        lb, lr = fb[0] * cmax ** -fb[1], fr[0] * cmax ** -fr[1]
        # compute the baseline would need to match Raslila's loss at cmax
        need_b = (lr / fb[0]) ** (-1 / fb[1])
        mult = need_b / cmax
        lines += ['', '## Head to head at C = {:.2e} FLOPs'.format(cmax), '',
                  '- baseline predicted loss {:.4f}, Raslila {:.4f}'.format(lb, lr),
                  '- compute multiplier: Raslila matches a baseline that spends **{:.2f}x** its compute'.format(mult),
                  '  (>1 = Raslila is more compute-efficient; <1 = it is worse)',
                  '- slopes: baseline b = {:.4f}, Raslila b = {:.4f} -> {}'.format(
                      fb[1], fr[1], 'Raslila improves FASTER with scale' if fr[1] > fb[1]
                      else 'Raslila improves SLOWER with scale (gap widens)')]
        for c_proj, label in ((6 * 10e9 * 200e9, '10B params, 200B tokens (Chinchilla)'),):
            lines.append('- PROJECTION (extrapolated, not measured) at {} = {:.1e} FLOPs: baseline {:.3f}, Raslila {:.3f}'.format(
                label, c_proj, fb[0] * c_proj ** -fb[1], fr[0] * c_proj ** -fr[1]))
    text = '\n'.join(lines) + '\n'
    print(text)
    os.makedirs('runs', exist_ok=True)
    with open('runs/scaling_report.md', 'w', encoding='utf-8') as f:
        f.write(text)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('(matplotlib not installed: no plot)')
        return 0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {'baseline': '#4c72b0', 'raslila': '#dd8452'}
    for fam in ('baseline', 'raslila'):
        rs = [m for m in runs if m['family'] == fam]
        if not rs:
            continue
        ax.scatter([m['flops_seen'] for m in rs], [m['val_loss'] for m in rs],
                   color=colors[fam], label=fam, zorder=3)
        for m in rs:
            ax.annotate('{:.0f}M'.format(m['params_total'] / 1e6), (m['flops_seen'], m['val_loss']),
                        textcoords='offset points', xytext=(4, 4), fontsize=8)
        f = fits[fam][0]
        if f:
            cs = sorted(m['flops_seen'] for m in rs)
            xs = [cs[0] * (cs[-1] * 10 / cs[0]) ** (i / 50) for i in range(51)]
            ax.plot(xs, [f[0] * x ** -f[1] for x in xs], color=colors[fam], ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('training FLOPs')
    ax.set_ylabel('validation loss (nats/token)')
    ax.set_title('Raslila v2 vs standard transformer (same data, tokens, recipe)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig('runs/scaling.png', dpi=150)
    print('saved runs/scaling_report.md and runs/scaling.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
