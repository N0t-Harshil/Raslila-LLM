"""Scaling report: Raslila vs the baseline, from every finished run.

Raslila v4 (the looped "cortex", arch: cortex) and the older grid versions
(v2/v3) are kept apart: each gets its own fitted line. thinking.png shows
how v4's loss and reasoning accuracy change with the number of thinking
loops, with the same-size baseline as a flat reference line.

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
    'var_chain': ('acc', True, 0.10), 'add2': ('acc', True, 0.25),
}
TIER_ORDER = {'50m': 0, '125m': 1, '150m': 2, '350m': 3, '1b': 4}


def _line(m):
    """Which curve a run belongs to: the baseline, Raslila v4, or the old grid."""
    if m['family'] != 'raslila':
        return 'baseline'
    return 'raslila v4' if (m.get('model') or {}).get('arch') == 'cortex' else 'raslila grid (v2/v3)'


def _labels(r, n):
    loops = r.get('exit_loops')
    return [str(x) for x in loops] if loops and len(loops) == n else [str(i + 1) for i in range(n)]


def _order(runs):
    return sorted(runs, key=lambda r: (TIER_ORDER.get(r['tier'], 9), r['params_total'], r['family']))


def _tasks(runs):
    return [t for t in TASKS if any(t in r['bench'] for r in runs)]


def _fmt(metric, v):
    return '{:.4f}'.format(v) if metric == 'bpb' else '{:.1%}'.format(v)


def _plain_baseline(runs, tier):
    b = [r for r in runs if r['tier'] == tier and r['family'] == 'baseline']
    b.sort(key=lambda r: (r['run'] != 'baseline_' + tier, r['run']))
    return b[0] if b else None


def equal_compute_rows(runs):
    """Measured equal-compute check: in each tier, finished baselines trained on the
    same tokens but at different depths give loss vs training FLOPs/token. Read the
    baseline loss off that line (log-linear) at each Raslila run's FLOPs/token."""
    rows = []
    fpt = lambda r: r['flops_seen'] / max(1, r['tokens_seen'])
    for tier in sorted({r['tier'] for r in runs}, key=lambda t: TIER_ORDER.get(t, 9)):
        zs = [r for r in runs if r['tier'] == tier and r['family'] == 'raslila' and r.get('finished')]
        for z in sorted(zs, key=lambda r: r['run']):
            bs = sorted([r for r in runs if r['tier'] == tier and r['family'] == 'baseline' and r.get('finished')
                         and abs(r['tokens_seen'] / max(1, z['tokens_seen']) - 1) < 0.05], key=fpt)
            if len(bs) < 2:
                continue
            x = math.log(fpt(z))
            lo = [b for b in bs if fpt(b) <= fpt(z)]
            hi = [b for b in bs if fpt(b) > fpt(z)]
            b1, b2 = (lo[-1], hi[0]) if (lo and hi) else ((bs[-2], bs[-1]) if not hi else (bs[0], bs[1]))
            x1, x2 = math.log(fpt(b1)), math.log(fpt(b2))
            lb = b1['val_loss'] + (b2['val_loss'] - b1['val_loss']) * (x - x1) / (x2 - x1)
            rows.append('| {} | {} | {:.2f}x | {:.4f} | {:.4f} | {:+.4f} | {} | {} |'.format(
                tier, z['run'], fpt(z) / fpt(bs[0]), lb, z['val_loss'], z['val_loss'] - lb,
                ' vs '.join('{} ({:.0f}M, {:.4f})'.format(b['run'], b['params_total'] / 1e6, b['val_loss']) for b in (b1, b2)),
                ('Raslila' if z['val_loss'] < lb else 'baseline') +
                ('' if (fpt(bs[0]) / 1.05 <= fpt(z) <= fpt(bs[-1]) * 1.05) else ' (extrapolated)')))
    if not rows:
        return []
    return ['', '## Equal training compute (measured, no scaling-law fit)', '',
            'Baselines of the same tier trained on the same tokens at different depths give baseline loss as a '
            'function of training FLOPs per token; it is read at each Raslila run FLOPs per token. diff = Raslila '
            'minus that baseline (negative = Raslila better at EQUAL training compute).', '',
            '| tier | Raslila run | its compute/token | baseline loss at that compute | Raslila val loss | diff '
            '| from baselines | better at equal compute |', '|---|---|---|---|---|---|---|---|'] + rows


def comparison_sections(runs):
    out = []
    tasks = _tasks(runs)
    rs = _order(runs)
    if tasks:
        out += ['', '## Benchmarks (zero-shot; gsm8k 3-shot)', '',
                'Higher is better, except *_bpb (bits per byte: lower is better). '
                'Bold = the better of the two models at that size. Raslila v4: read out after its test_loops (8); grid: after all cycles.', '',
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
                pb = _plain_baseline(rs, r['tier'])
                mate = ([pb] if r['family'] != 'baseline' else
                        [o for o in rs if o['tier'] == r['tier'] and o['family'] == 'raslila'] if r is pb else [])
                mate = [o for o in mate if o and o['bench'].get(t, {}).get('value') is not None]
                better = bool(mate) and ((v > mate[0]['bench'][t]['value']) if hib
                                         else (v < mate[0]['bench'][t]['value']))
                row.append(('**{}**' if better else '{}').format(_fmt(metric, v)))
            out.append('| {} | {} | {} | {} |'.format(
                t, metric, '-' if chance is None else '{:.0%}'.format(chance), ' | '.join(row)))

    h2h = []
    for tier in sorted({r['tier'] for r in runs}, key=lambda t: TIER_ORDER.get(t, 9)):
        b = _plain_baseline(runs, tier)
        zs = [r for r in runs if r['tier'] == tier and r['family'] == 'raslila']
        if not (b and zs):
            continue
        for z in sorted(zs, key=lambda r: r['run']):
            h2h.append(_h2h_row(tier, b, z, tasks))
    if h2h:
        out += ['', '## Head to head at the same size', '',
                'diff = Raslila minus baseline (negative = Raslila better). Under ~0.02 is noise at one size. '
                '"compute/token" = training FLOPs per token relative to the baseline (thinking costs compute).', '',
                '| tier | Raslila run | params (baseline / Raslila) | baseline val loss | Raslila val loss | diff '
                '| compute/token | benchmarks Raslila won / lost | lower val loss |',
                '|---|---|---|---|---|---|---|---|---|'] + h2h

    loop = []
    for r in rs:
        if r['family'] != 'raslila':
            continue
        per = r.get('val_loss_per_cycle') or []
        lab = _labels(r, len(per))
        if len(per) > 1:
            loop.append('| {} | validation loss | {} | {:+.4f} |'.format(
                r['run'], '  '.join('{}:{:.4f}'.format(a, v) for a, v in zip(lab, per)), per[-1] - per[0]))
        for t in tasks:
            pc = r['bench'].get(t, {}).get('per_cycle')
            if pc and len(pc) > 1:
                lab = [str(x) for x in (r['bench'][t].get('exit_loops') or range(1, len(pc) + 1))]
                loop.append('| {} | {} | {} | {:+.4f} |'.format(
                    r['run'], t, '  '.join('{}:{}'.format(a, _fmt(TASKS[t][0], v)) for a, v in zip(lab, pc)),
                    pc[-1] - pc[0]))
    if loop:
        out += ['', '## Thinking: does more looping help?', '',
                'Same model, read out after 1, 2, 4, ... loops (v4) or after each cycle (grid). '
                'Each cell is loops:value. For loss / bpb a negative change is better; for accuracy a positive one. '
                'A standard transformer cannot do this: its line would be flat. See thinking.png.', '',
                '| run | measure | by loops | first -> last change |', '|---|---|---|---|'] + loop
    out += equal_compute_rows(runs)
    return out


def _h2h_row(tier, b, z, tasks):
    wins = losses = 0
    for t in tasks:
        vb, vz = b['bench'].get(t, {}).get('value'), z['bench'].get(t, {}).get('value')
        if vb is None or vz is None or abs(vz - vb) < 1e-9:
            continue
        if (vz > vb) == TASKS[t][1]:
            wins += 1
        else:
            losses += 1
    cpt = (z['flops_seen'] / max(1, z['tokens_seen'])) / max(1e-9, b['flops_seen'] / max(1, b['tokens_seen']))
    return '| {} | {} | {:.0f}M / {:.0f}M | {:.4f} | {:.4f} | {:+.4f} | {:.2f}x | {} / {} | {}{} |'.format(
        tier, z['run'], b['params_total'] / 1e6, z['params_total'] / 1e6, b['val_loss'], z['val_loss'],
        z['val_loss'] - b['val_loss'], cpt, wins, losses,
        'Raslila' if z['val_loss'] < b['val_loss'] else 'baseline',
        '' if (b.get('finished') and z.get('finished')) else ' (partial runs)')


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
            shade = 0.45 + 0.55 * (TIER_ORDER.get(r['tier'], 0) + 1) / 5
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


def plot_thinking(plt, runs):
    """v4: loss and accuracy vs number of thinking loops; baseline = flat line."""
    looped = [r for r in _order(runs) if r['line'] == 'raslila v4' and r.get('exit_loops')]
    if not looped:
        return
    panels = [('validation loss', None)] + [(t, t) for t in ('var_chain', 'add2', 'arc_easy', 'hellaswag',
                                                              'gsm8k_bpb', 'wikitext_bpb')
                                            if any(r['bench'].get(t, {}).get('per_cycle') for r in looped)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), squeeze=False)
    for ax, (title, t) in zip(axes[0], panels):
        for r in looped:
            if t is None:
                xs, ys = r['exit_loops'], r.get('val_loss_per_cycle') or []
            else:
                b = r['bench'].get(t, {})
                xs, ys = b.get('exit_loops') or [], b.get('per_cycle') or []
            if xs and len(xs) == len(ys):
                ax.plot(xs, ys, 'o-', color='#dd8452', label=r['run'])
            base = [b for b in [_plain_baseline(runs, r['tier'])] if b]
            if base:
                bv = base[0]['val_loss'] if t is None else base[0]['bench'].get(t, {}).get('value')
                if bv is not None:
                    ax.axhline(bv, color='#4c72b0', ls='--', lw=1, label=base[0]['run'])
        if t and TASKS[t][2]:
            ax.axhline(TASKS[t][2], color='k', ls=':', lw=1, label='chance')
        ax.set_xscale('log', base=2)
        ticks = sorted({x for r in looped for x in (r.get('exit_loops') or [])})
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(x) for x in ticks])
        ax.set_xlabel('thinking loops')
        ax.set_title(title if t is None else '{} ({})'.format(t, TASKS[t][0]), fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.suptitle('Thinking time: Raslila v4 read out after more loops (dashed = same-size standard transformer)')
    fig.tight_layout()
    fig.savefig('runs/thinking.png', dpi=150)
    plt.close(fig)


def load_runs(pattern='runs/*/metrics.json'):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        if m.get('data') == 'synthetic' or not m.get('val_loss'):
            continue
        m['run'] = os.path.basename(os.path.dirname(path))
        parts = m['run'].split('_')
        m['tier'] = parts[1] if len(parts) > 1 else m['run']      # raslila_150m_topo -> 150m
        m['line'] = _line(m)
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
    lines = ['# Raslila vs baseline -- scaling report', '',
             'Generated {}. Plots: scaling.png, curves.png, benchmarks.png, thinking.png. Every number: compare.csv.'.format(
                 time.strftime('%Y-%m-%d %H:%M')), '',
             '## Training', '',
             '| run | model | params | tokens | train FLOPs | val loss | loss by loops / cycles | tok/s/GPU | MFU | GPU-h | done |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in _order(runs):
        per = m.get('val_loss_per_cycle') or []
        lines.append('| {} | {} | {:.0f}M | {:.2f}B | {:.2e} | **{:.4f}** | {} | {:,.0f} | {} | {:.1f} | {} |'.format(
            m['run'], m['line'], m['params_total'] / 1e6, m['tokens_seen'] / 1e9, m['flops_seen'],
            m['val_loss'], ' '.join('{}:{:.3f}'.format(a, v) for a, v in zip(_labels(m, len(per)), per))
            if len(per) > 1 else '-',
            m['tokens_per_second'] / max(1, m['gpus']),
            '{:.1%}'.format(m['mfu']) if m.get('mfu') else 'n/a',
            m['train_seconds'] * m['gpus'] / 3600, 'yes' if m.get('finished') else 'partial'))

    fits = {}
    for fam in ('baseline', 'raslila v4', 'raslila grid (v2/v3)'):
        pts = [(m['flops_seen'], m['val_loss']) for m in runs if m['line'] == fam and m.get('finished')
               and m['run'].count('_') == 1]                      # variants (e.g. _topo) are not in the fit
        if pts or fam == 'baseline':
            fits[fam] = (fit(pts), len(pts))
    lines += ['', '## Fit: loss = a * C^-b (finished runs only)', '']
    for fam, (f, n) in fits.items():
        if f:
            lines.append('- **{}**: a = {:.3f}, b = {:.4f} ({} runs){}'.format(
                fam, f[0], f[1], n, '  <- fewer than 3 points, not reliable' if n < 3 else ''))
        else:
            lines.append('- **{}**: needs at least 2 finished runs ({} so far)'.format(fam, n))

    fb = fits['baseline'][0]
    fr = (fits.get('raslila v4') or (None,))[0] or (fits.get('raslila grid (v2/v3)') or (None,))[0]
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
    colors = {'baseline': '#4c72b0', 'raslila v4': '#dd8452', 'raslila grid (v2/v3)': '#bbbbbb'}
    for fam in colors:
        rs = [m for m in runs if m['line'] == fam]
        if not rs:
            continue
        ax.scatter([m['flops_seen'] for m in rs], [m['val_loss'] for m in rs],
                   color=colors[fam], label=fam, zorder=3)
        for m in rs:
            ax.annotate('{:.0f}M'.format(m['params_total'] / 1e6), (m['flops_seen'], m['val_loss']),
                        textcoords='offset points', xytext=(4, 4), fontsize=8)
        f = (fits.get(fam) or (None,))[0]
        if f:
            cs = sorted(m['flops_seen'] for m in rs)
            xs = [cs[0] * (cs[-1] * 10 / cs[0]) ** (i / 50) for i in range(51)]
            ax.plot(xs, [f[0] * x ** -f[1] for x in xs], color=colors[fam], ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('training FLOPs')
    ax.set_ylabel('validation loss (nats/token)')
    ax.set_title('Raslila vs standard transformer (same data, tokens, recipe)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig('runs/scaling.png', dpi=150)
    plt.close(fig)
    plot_curves(plt, runs)
    plot_bench(plt, runs)
    plot_thinking(plt, runs)
    print('saved runs/scaling_report.md, compare.csv, scaling.png, curves.png, benchmarks.png, thinking.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
