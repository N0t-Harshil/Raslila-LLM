"""Scaling report: Raslila v2 vs the baseline, from every finished run.

    python scripts/scaling_report.py            # reads runs/*/metrics.json

Prints a table, fits a power law  loss = a * C^-b  per family (C = training
FLOPs actually spent), and says, at the compute of the largest run, how much
compute the other family would need for the same loss (the "compute
multiplier"). It also compares every model side by side: the training table,
benchmarks (scripts/benchmark.py), the effect of each thinking cycle, and a
head-to-head at each size.

Writes (all under runs/):
    scaling_report.md   everything below, as one readable page
    compare.csv         one row per run, every number (Excel / pandas)
    scaling.png         loss vs training FLOPs + fitted curves
    curves.png          validation loss during training, every run overlaid
    benchmarks.png      benchmark scores, every run side by side

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
import time

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

# task -> (primary metric, higher is better, chance); mirrors scripts/benchmark.py
# (copied, not imported, so the report runs without torch installed)
TASKS = {
    'hellaswag': ('acc_norm', True, 0.25), 'arc_easy': ('acc', True, 0.25),
    'arc_challenge': ('acc_norm', True, 0.25), 'sciq': ('acc', True, 0.25),
    'openbookqa': ('acc_norm', True, 0.25), 'winogrande': ('acc', True, 0.50),
    'lambada': ('acc', True, 0.0), 'wikitext_bpb': ('bpb', False, None),
    'gsm8k_bpb': ('bpb', False, None), 'humaneval_bpb': ('bpb', False, None),
    'gsm8k': ('exact_match', True, 0.0), 'humaneval': ('pass@1', True, 0.0),
}
TIER_ORDER = {'50m': 0, '125m': 1, '350m': 2, '1b': 3}


def _order(runs):
    return sorted(runs, key=lambda r: (TIER_ORDER.get(r['tier'], 9), r['params_total'], r['family']))


def _tasks(runs):
    return [t for t in TASKS if any(t in r['bench'] for r in runs)]


def _fmt(metric, v):
    return '{:.4f}'.format(v) if metric == 'bpb' else '{:.1%}'.format(v)


def comparison_sections(runs):
    out = []
    tasks = _tasks(runs)
    rs = _order(runs)
    if tasks:
        out += ['', '## Benchmarks (zero-shot; gsm8k 3-shot)', '',
                'Higher is better, except *_bpb (bits per byte: lower is better). '
                'Bold = the better of the two models at that size. Raslila uses all its thinking cycles.', '',
                '| task | metric | chance | ' + ' | '.join(r['run'] for r in rs) + ' |',
                '|---|---|---|' + '---|' * len(rs)]
        for t in tasks:
            metric, hib, chance = TASKS[t]
            row = []
            for r in rs:
                v = r['bench'].get(t, {}).get('value')
                if v is None:
                    row.append('-')
                    continue
                mate = [o for o in rs if o['tier'] == r['tier'] and o is not r
                        and o['bench'].get(t, {}).get('value') is not None]
                better = bool(mate) and ((v > mate[0]['bench'][t]['value']) if hib
                                         else (v < mate[0]['bench'][t]['value']))
                row.append(('**{}**' if better else '{}').format(_fmt(metric, v)))
            out.append('| {} | {} | {} | {} |'.format(
                t, metric, '-' if chance is None else '{:.0%}'.format(chance), ' | '.join(row)))

    h2h = []
    for tier in sorted({r['tier'] for r in runs}, key=lambda t: TIER_ORDER.get(t, 9)):
        b = [r for r in runs if r['tier'] == tier and r['family'] == 'baseline']
        z = [r for r in runs if r['tier'] == tier and r['family'] == 'raslila']
        if not (b and z):
            continue
        b, z = b[0], z[0]
        wins = losses = 0
        for t in tasks:
            vb, vz = b['bench'].get(t, {}).get('value'), z['bench'].get(t, {}).get('value')
            if vb is None or vz is None or abs(vz - vb) < 1e-9:
                continue
            if (vz > vb) == TASKS[t][1]:
                wins += 1
            else:
                losses += 1
        h2h.append('| {} | {:.4f} | {:.4f} | {:+.4f} | {:.2f}x | {} / {} | {}{} |'.format(
            tier, b['val_loss'], z['val_loss'], z['val_loss'] - b['val_loss'],
            z['tokens_per_second'] / max(1e-9, b['tokens_per_second']), wins, losses,
            'Raslila' if z['val_loss'] < b['val_loss'] else 'baseline',
            '' if (b.get('finished') and z.get('finished')) else ' (partial runs)'))
    if h2h:
        out += ['', '## Head to head at the same size', '',
                'diff = Raslila minus baseline (negative = Raslila better). Under ~0.02 is noise at one size.', '',
                '| tier | baseline val loss | Raslila val loss | diff | Raslila speed vs baseline '
                '| benchmarks Raslila won / lost | lower val loss |',
                '|---|---|---|---|---|---|---|'] + h2h

    loop = []
    for r in rs:
        if r['family'] != 'raslila':
            continue
        per = r.get('val_loss_per_cycle') or []
        if len(per) > 1:
            loop.append('| {} | validation loss | {} | {:+.4f} |'.format(
                r['run'], ' -> '.join('{:.4f}'.format(v) for v in per), per[-1] - per[0]))
        for t in tasks:
            pc = r['bench'].get(t, {}).get('per_cycle')
            if pc and len(pc) > 1:
                loop.append('| {} | {} | {} | {:+.4f} |'.format(
                    r['run'], t, ' -> '.join(_fmt(TASKS[t][0], v) for v in pc), pc[-1] - pc[0]))
    if loop:
        out += ['', '## Thinking loops: does another cycle help?', '',
                'Same model, same forward pass, read out after cycle 1, 2, ... '
                'For loss / bpb a negative change is better; for accuracy a positive one.', '',
                '| run | measure | cycle 1 -> last | change |', '|---|---|---|---|'] + loop
    return out


def write_csv(runs):
    import csv
    tasks = _tasks(runs)
    cols = ['run', 'family', 'tier', 'finished', 'params_total', 'params_nonembed', 'tokens_seen',
            'flops_seen', 'val_loss', 'val_loss_per_cycle', 'tokens_per_second', 'mfu', 'gpus',
            'gpu_hours', 'steps_done', 'steps_total'] + tasks
    with open('runs/compare.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in _order(runs):
            row = dict(r, gpu_hours=r['train_seconds'] * r['gpus'] / 3600,
                       val_loss_per_cycle=' '.join('{:.4f}'.format(v) for v in (r.get('val_loss_per_cycle') or [])))
            for t in tasks:
                row[t] = r['bench'].get(t, {}).get('value')
            w.writerow([row.get(c) for c in cols])


COLORS = {'baseline': '#4c72b0', 'raslila': '#dd8452'}


def plot_curves(plt, runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    styles = ['-', '--', '-.', ':']
    for r in _order(runs):
        h = r.get('history') or []
        if not h:
            continue
        for ax, key in zip(axes, ('tokens', 'flops')):
            ax.plot([p[key] for p in h], [p['val_loss'] for p in h], ls=styles[TIER_ORDER.get(r['tier'], 0) % 4],
                    color=COLORS.get(r['family'], 'gray'), label=r['run'])
    for ax, xl in zip(axes, ('tokens seen', 'training FLOPs')):
        ax.set_xscale('log')
        ax.set_xlabel(xl)
        ax.set_ylabel('validation loss (nats/token)')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('Validation loss during training (orange = Raslila, blue = baseline)')
    fig.tight_layout()
    fig.savefig('runs/curves.png', dpi=150)
    plt.close(fig)


def plot_bench(plt, runs):
    tasks = _tasks(runs)
    rs = [r for r in _order(runs) if r['bench']]
    if not rs or not tasks:
        return
    groups = [([t for t in tasks if TASKS[t][0] != 'bpb'], 'accuracy (higher is better)'),
              ([t for t in tasks if TASKS[t][0] == 'bpb'], 'bits per byte (lower is better)')]
    groups = [g for g in groups if g[0]]
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 4.8), squeeze=False,
                             gridspec_kw={'width_ratios': [len(g[0]) for g in groups]})
    w = 0.8 / len(rs)
    for ax, (group, title) in zip(axes[0], groups):
        for k, r in enumerate(rs):
            shade = 0.45 + 0.55 * (TIER_ORDER.get(r['tier'], 0) + 1) / 4
            col = (0.30 * shade, 0.45 * shade, 0.69 * shade + 0.2) if r['family'] == 'baseline' \
                else (0.87 * shade + 0.1, 0.52 * shade, 0.32 * shade)
            ax.bar([i + k * w for i in range(len(group))],
                   [r['bench'].get(t, {}).get('value') or 0 for t in group], w, label=r['run'], color=col)
        for i, t in enumerate(group):
            if TASKS[t][2]:
                ax.hlines(TASKS[t][2], i - w / 2, i + 0.8 - w / 2, colors='k', linestyles=':', lw=1)
        ax.set_xticks([i + 0.4 - w / 2 for i in range(len(group))])
        ax.set_xticklabels(group, rotation=30, ha='right', fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle('Benchmarks, every model side by side (dotted line = chance)')
    fig.tight_layout()
    fig.savefig('runs/benchmarks.png', dpi=150)
    plt.close(fig)


def load_runs(pattern='runs/*/metrics.json'):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        if m.get('data') == 'synthetic' or not m.get('val_loss'):
            continue
        m['run'] = os.path.basename(os.path.dirname(path))
        m['tier'] = m['run'].split('_', 1)[-1]
        m['bench'] = {}
        bp = os.path.join(os.path.dirname(path), 'benchmark.json')
        if os.path.exists(bp):
            with open(bp, 'r', encoding='utf-8') as f:
                m['bench'] = json.load(f).get('tasks', {})
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
             'Generated {}. Plots: scaling.png, curves.png, benchmarks.png. Every number: compare.csv.'.format(
                 time.strftime('%Y-%m-%d %H:%M')), '',
             '## Training', '',
             '| run | family | params | tokens | train FLOPs | val loss | loss per cycle | tok/s/GPU | MFU | GPU-h | done |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in _order(runs):
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
    lines += comparison_sections(runs)
    text = '\n'.join(lines) + '\n'
    print(text)
    write_csv(runs)
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
    plt.close(fig)
    plot_curves(plt, runs)
    plot_bench(plt, runs)
    print('saved runs/scaling_report.md, compare.csv, scaling.png, curves.png, benchmarks.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
