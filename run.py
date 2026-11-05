"""Raslila: one entry point. Every command logs to logs/<command>_<time>.log.

    python run.py <command> [options]        python run.py <command> --help

THE PLAN (scaling study: does the 3D brain grid beat a standard transformer?)
    1. python run.py setup            install requirements, check packages, list GPUs
    2. python run.py check            validate the study configs (fair pairs, sizes, cost)
    3. python run.py smoke            seconds-long runs of both models (no data needed)
    4. python run.py data             build the general+math+code corpus (once, cached)
    5. python run.py speed            FIRST GO/NO-GO: Raslila vs baseline tokens/s on your GPU
    6. python run.py study            train every model ONE AFTER ANOTHER, each on all GPUs:
                                      baseline_50m, raslila_50m, baseline_125m, ... raslila_1b.
                                      After each model: benchmark it + refresh the comparison.
    7. python run.py report           comparison of every model: training, benchmarks,
                                      thinking loops, scaling fit (runs/scaling_report.md)

WATCHING / COMPARING
    python run.py status              one line per model: progress, val loss, ETA
    python run.py compare             same as report
    python run.py bench --run raslila_125m      benchmark one model (or --all)
    logs/study.log                    timeline of the whole study (start/done/scores)
    runs/<name>/train.log             full training output of that model (kept across resumes)
    runs/<name>/log.csv, metrics.json, benchmark.json   every number of that model

SINGLE RUNS / USING A MODEL
    python run.py scale --tier 125m --family raslila [--gpus 4]   one training run
    python run.py generate --run raslila_125m --prompt "..."      sample text
    python run.py live --run raslila_125m                         always-on chat session
    python run.py legacy <command>                                old v1 commands

Training resumes automatically from runs/<name>/ckpt.pt, so if anything stops
(crash, reboot, Ctrl+C) just run the same command again. Long runs over SSH:
start them inside `tmux` so a dropped connection does not kill them.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))
PY = sys.executable
TIERS = ['50m', '125m', '350m', '1b']
FAMILIES = ['raslila', 'baseline']


def banner(text):
    print('\n' + '=' * 78 + '\n' + text + '\n' + '=' * 78, flush=True)


def log_path(name):
    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    return os.path.join(ROOT, 'logs', '{}_{}.log'.format(
        name, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))


def run(cmd, name, env=None, also_log=None):
    """Run a command, streaming output to the console AND a log file
    (and, if given, appending it to `also_log` too)."""
    path = log_path(name)
    print('>>> ' + ' '.join(cmd) + '\n    (log: {})'.format(os.path.relpath(path, ROOT)), flush=True)
    full_env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', **(env or {}))
    extra = None
    if also_log:
        os.makedirs(os.path.dirname(also_log), exist_ok=True)
        extra = open(also_log, 'a', encoding='utf-8')
        extra.write('\n##### {} | {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'), ' '.join(cmd)))
    with open(path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=full_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                errors='replace', bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                if extra:
                    extra.write(line)
                    extra.flush()
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            print('\n[interrupted -- run the same command again to resume]')
            return 130
        finally:
            if extra:
                extra.close()
    if proc.returncode:
        print('\n[{} exited with code {}; see {}]'.format(name, proc.returncode, os.path.relpath(path, ROOT)))
    return proc.returncode


def n_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def train_cmd(config, gpus, extra=()):
    if gpus > 1:
        return [PY, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node={}'.format(gpus), 'scripts/train_scaling.py', '--config', config] + list(extra)
    return [PY, 'scripts/train_scaling.py', '--config', config] + list(extra)


def run_status(name):
    path = os.path.join(ROOT, 'runs', name, 'metrics.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
ENV_PROBE = r'''
import importlib, warnings
for name in ("numpy", "torch", "pyarrow", "datasets", "tokenizers", "yaml", "matplotlib"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            m = importlib.import_module(name)
        except Exception as e:
            print("FAIL", name, type(e).__name__ + ":", str(e).splitlines()[0][:200]); continue
    bad = [str(w.message).splitlines()[0][:200] for w in caught if "numpy" in str(w.message).lower()]
    print(("FAIL " + name + " NumPy warning: " + bad[0]) if bad else ("OK   " + name + " " + getattr(m, "__version__", "")))
'''


def check_env():
    out = subprocess.run([PY, '-c', ENV_PROBE], cwd=ROOT, capture_output=True, text=True,
                         encoding='utf-8', errors='replace').stdout
    print(out.rstrip())
    fails = [l for l in out.splitlines() if l.startswith('FAIL') and 'matplotlib' not in l]
    if not fails:
        return True
    if any('numpy' in l.lower() for l in fails):
        print('\nNumPy version conflict. On conda/Jupyter images keep NumPy 1 and pin pyarrow:\n'
              '  pip install "numpy<2" "pyarrow<20"    (add "datasets<4.1" if pip refuses)\n'
              'Cleanest fix: a fresh venv, then pip install -r requirements.txt')
    else:
        print('\nInstall the missing packages:  pip install -r requirements.txt')
    return False


def cmd_setup(a):
    banner('Installing requirements')
    if run([PY, '-m', 'pip', 'install', '-r', 'requirements.txt'], 'setup'):
        return 1
    banner('Packages')
    if not check_env():
        return 1
    banner('GPUs')
    import torch
    print('torch', torch.__version__, '| CUDA', torch.version.cuda)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print('GPU {}: {} ({:.0f} GB)'.format(i, p.name, p.total_memory / 1e9))
    if not torch.cuda.is_available():
        print('No CUDA GPU visible: install the CUDA build of torch (pytorch.org/get-started).')
    return 0


def cmd_check(a):
    banner('Validating scaling-study configs')
    return run([PY, 'scripts/check_scaling.py'], 'check')


def cmd_smoke(a):
    banner('Smoke test: both model families, random tokens, a few seconds each')
    for fam in FAMILIES:
        rc = run([PY, 'scripts/train_scaling.py', '--config', 'configs/scaling/smoke_{}.yaml'.format(fam),
                  '--synthetic', '--fresh'], 'smoke_' + fam)
        if rc:
            return rc
    print('\nSmoke OK. (Loss stays near ln(vocab) on random tokens -- that is expected.)')
    return 0


def cmd_data(a):
    banner('Packages')
    if not check_env():
        return 1
    banner('Building the corpus from {} (cached; re-running is a no-op)'.format(a.config))
    code = ('import sys; sys.path.insert(0, "src"); '
            'from raslila.data_mix import load_data_config, build_cache; '
            'build_cache(load_data_config({!r}), force={})'.format(a.config, bool(a.force)))
    return run([PY, '-c', code], 'data')


def cmd_speed(a):
    banner('Speed test, tier {} (largest micro-batch that fits, then timed steps)'.format(a.tier))
    cmd = [PY, 'scripts/speed_test.py', '--tier', a.tier]
    if a.compile:
        cmd.append('--compile')
    return run(cmd, 'speed_' + a.tier)


def cmd_scale(a):
    fams = FAMILIES if a.family == 'both' else [a.family]
    for fam in fams:
        cfg = 'configs/scaling/{}_{}.yaml'.format(fam, a.tier)
        banner('Training {} on {} GPU(s)'.format(cfg, a.gpus))
        extra = ['--compile'] if a.compile else []
        if a.micro_batch:
            extra += ['--micro-batch', str(a.micro_batch)]
        rc = run(train_cmd(cfg, a.gpus, extra), 'train_{}_{}'.format(fam, a.tier))
        if rc:
            return rc
    return 0


STUDY_LOG = os.path.join(ROOT, 'logs', 'study.log')


def study_log(msg):
    """One line in logs/study.log (appended, never overwritten) and on screen."""
    os.makedirs(os.path.dirname(STUDY_LOG), exist_ok=True)
    line = '[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M'), msg)
    print(line, flush=True)
    with open(STUDY_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def bench_summary(name):
    path = os.path.join(ROOT, 'runs', name, 'benchmark.json')
    if not os.path.exists(path):
        return ''
    with open(path, 'r', encoding='utf-8') as f:
        tasks = json.load(f).get('tasks', {})
    return ', '.join('{} {:.3f}'.format(t, v['value']) for t, v in tasks.items())


def bench_cmd(runs, a):
    cmd = [PY, 'scripts/benchmark.py']
    for r in runs:
        cmd += ['--run', r]
    if getattr(a, 'quick', False):
        cmd.append('--quick')
    if getattr(a, 'run_code', False):
        cmd.append('--run-code')
    if getattr(a, 'think', None):
        cmd += ['--think', str(a.think)]
    if getattr(a, 'tasks', None):
        cmd += ['--tasks', a.tasks]
    if getattr(a, 'force', False):
        cmd.append('--force')
    return cmd


def cmd_study(a):
    """Train every model one after another, each on ALL GPUs (DDP).
    Order: baseline then Raslila at each size, smallest size first, so a
    fair same-size comparison is available as early as possible. After every
    model: benchmark it and refresh the comparison report."""
    gpus = a.gpus or n_gpus()
    if gpus < 1:
        sys.exit('No GPUs visible.')
    tiers = [t for t in TIERS if t in a.tiers.split(',')]
    order = [(f, t) for t in tiers for f in ('baseline', 'raslila')]
    names = ['{}_{}'.format(f, t) for f, t in order]
    banner('Study: {} models, one after another, each on {} GPU(s)\n  {}'.format(
        len(order), gpus, ' -> '.join(names)))
    study_log('STUDY START on {} GPU(s): {}'.format(gpus, ' -> '.join(names)))
    extra = ['--compile'] if a.compile else []
    failed = []
    for n, (fam, tier) in enumerate(order, 1):
        name = '{}_{}'.format(fam, tier)
        st = run_status(name)
        if st and st.get('finished'):
            print('[{}/{}] {} already trained (val loss {:.4f})'.format(n, len(order), name, st['val_loss']))
        else:
            cfg = 'configs/scaling/{}.yaml'.format(name)
            banner('[{}/{}] TRAINING {} on {} GPU(s)'.format(n, len(order), name, gpus))
            study_log('START {} ({}/{})'.format(name, n, len(order)))
            t0 = time.time()
            rc = run(train_cmd(cfg, gpus, extra), 'train_' + name,
                     also_log=os.path.join(ROOT, 'runs', name, 'train.log'))
            if rc == 130:
                study_log('STOPPED by user during {} (re-run `python run.py study` to resume)'.format(name))
                return 130
            st = run_status(name) or {}
            if rc or not st.get('finished'):
                failed.append(name)
                study_log('FAILED {} (exit {}) after {:.1f}h -- see runs/{}/train.log; moving on'.format(
                    name, rc, (time.time() - t0) / 3600, name))
                continue
            study_log('DONE {}: val loss {:.4f} | {:.0f}M params | {:.2f}B tokens | {:,.0f} tok/s | {:.1f}h'.format(
                name, st['val_loss'], st['params_total'] / 1e6, st['tokens_seen'] / 1e9,
                st['tokens_per_second'], (time.time() - t0) / 3600))
        if not a.no_bench:
            banner('[{}/{}] BENCHMARK {}'.format(n, len(order), name))
            rc = run(bench_cmd([name], a), 'bench_' + name,
                     also_log=os.path.join(ROOT, 'runs', name, 'benchmark.log'))
            if rc == 130:
                return 130
            study_log(('BENCH {}: {}' if not rc else 'BENCH {} had errors: {}').format(name, bench_summary(name)))
        run([PY, 'scripts/scaling_report.py'], 'report')
        study_log('comparison updated: runs/scaling_report.md')
    study_log('STUDY END. failed: {}'.format(', '.join(failed) or 'none'))
    return 1 if failed else 0


def cmd_bench(a):
    if not a.run and not a.all:
        sys.exit('Give --run NAME (repeatable) or --all.')
    runs = list(a.run or [])
    if a.all:
        import glob
        runs += sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, 'runs', '*', 'ckpt.pt'))
                       if not os.path.basename(os.path.dirname(p)).startswith('smoke'))
    banner('Benchmark: ' + ', '.join(runs))
    rc = 0
    for r in dict.fromkeys(runs):
        rc = run(bench_cmd([r], a), 'bench_' + r, also_log=os.path.join(ROOT, 'runs', r, 'benchmark.log')) or rc
        if rc == 130:
            return rc
        study_log('BENCH {}: {}'.format(r, bench_summary(r)))
    run([PY, 'scripts/scaling_report.py'], 'report')
    return rc


def cmd_status(a):
    banner('Status of every model (runs/)')
    print('{:<16} {:>9} {:>13} {:>9} {:>11} {:>8}  {}'.format(
        'model', 'params', 'progress', 'val loss', 'tok/s', 'ETA', 'benchmarks'))
    for t in TIERS:
        for f in ('baseline', 'raslila'):
            name = '{}_{}'.format(f, t)
            st = run_status(name)
            if not st:
                print('{:<16} {:>9}'.format(name, 'not started'))
                continue
            done, total = st.get('steps_done', 0), max(1, st.get('steps_total', 1))
            eta = '' if st.get('finished') else '{:.1f}h'.format(
                (total - done) * st['train_seconds'] / max(1, done) / 3600)
            val = st.get('val_loss')
            print('{:<16} {:>8.0f}M {:>12} {:>9} {:>11,.0f} {:>8}  {}'.format(
                name, st['params_total'] / 1e6,
                'finished' if st.get('finished') else '{:.0%} ({}/{})'.format(done / total, done, total),
                '{:.4f}'.format(val) if val else '-', st['tokens_per_second'], eta,
                'yes' if os.path.exists(os.path.join(ROOT, 'runs', name, 'benchmark.json')) else '-'))
    if os.path.exists(STUDY_LOG):
        with open(STUDY_LOG, 'r', encoding='utf-8') as f:
            tail = f.readlines()[-8:]
        print('\nlast lines of logs/study.log:\n' + ''.join('  ' + l for l in tail).rstrip())
    print('\nFull comparison: python run.py report  ->  runs/scaling_report.md')
    return 0


def cmd_report(a):
    banner('Scaling report')
    return run([PY, 'scripts/scaling_report.py'], 'report')


def cmd_generate(a):
    cmd = [PY, 'scripts/generate_scaling.py', '--run', a.run, '--prompt', a.prompt,
           '--tokens', str(a.tokens), '--temperature', str(a.temperature)]
    if a.think:
        cmd += ['--think', str(a.think)]
    return run(cmd, 'generate')


def cmd_live(a):
    from raslila.data_mix import load_data_config
    tok = load_data_config('configs/data.yaml')['tokenizer']
    cmd = [PY, 'scripts/raslila_live.py', '--checkpoint', os.path.join('runs', a.run, 'ckpt.pt'),
           '--tokenizer', tok, '--window', '1024']
    if a.inbox:
        cmd += ['--inbox', a.inbox]
    if a.offload:
        cmd.append('--offload')
    print('>>> ' + ' '.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def cmd_legacy(a):
    return subprocess.call([PY, 'scripts/legacy_run.py'] + a.rest, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', metavar='command')

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        return p

    add('setup', cmd_setup, 'install requirements, check packages, list GPUs')
    add('check', cmd_check, 'validate the scaling-study configs')
    add('smoke', cmd_smoke, 'seconds-long runs of both models on random tokens')
    p = add('data', cmd_data, 'build the general+math+code token cache (once)')
    p.add_argument('--config', default='configs/data.yaml')
    p.add_argument('--force', action='store_true')
    p = add('speed', cmd_speed, 'tokens/s of Raslila vs baseline on this GPU')
    p.add_argument('--tier', default='125m', choices=TIERS)
    p.add_argument('--compile', action='store_true', help='also try torch.compile')
    p = add('scale', cmd_scale, 'train one tier (one or both families)')
    p.add_argument('--tier', required=True, choices=TIERS)
    p.add_argument('--family', default='both', choices=FAMILIES + ['both'])
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--micro-batch', type=int, default=None)
    p.add_argument('--compile', action='store_true')
    p = add('study', cmd_study, 'train every model one after another (each on all GPUs), '
                                'benchmark each, keep the comparison up to date')
    p.add_argument('--gpus', type=int, default=None, help='default: all visible')
    p.add_argument('--tiers', default='50m,125m,350m,1b')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--no-bench', action='store_true', help='skip benchmarking after each model')
    p.add_argument('--quick', action='store_true', help='quick benchmarks (300 examples per task)')
    p.add_argument('--run-code', action='store_true', help='include HumanEval pass@1 (executes model code)')
    p = add('bench', cmd_bench, 'benchmark trained models (general, math, code)')
    p.add_argument('--run', action='append', help='e.g. raslila_125m (repeatable)')
    p.add_argument('--all', action='store_true', help='every model with a checkpoint')
    p.add_argument('--quick', action='store_true', help='300 examples per task')
    p.add_argument('--tasks', default=None, help='comma list, e.g. hellaswag,gsm8k')
    p.add_argument('--think', type=int, default=None, help='thinking cycles for Raslila')
    p.add_argument('--run-code', action='store_true', help='include HumanEval pass@1 (executes model code)')
    p.add_argument('--force', action='store_true', help='re-score even if cached')
    add('status', cmd_status, 'one line per model: progress, val loss, ETA')
    add('report', cmd_report, 'compare every model: training, benchmarks, loops, scaling fit')
    add('compare', cmd_report, 'same as report')
    p = add('generate', cmd_generate, 'sample text from a trained run')
    p.add_argument('--run', required=True, help='e.g. raslila_125m (a folder in runs/)')
    p.add_argument('--prompt', default='The most important idea in science is')
    p.add_argument('--tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--think', type=int, default=None, help='max thinking cycles (Raslila)')
    p = add('live', cmd_live, 'always-on chat session with a trained run')
    p.add_argument('--run', required=True)
    p.add_argument('--inbox', default='data/live_inbox')
    p.add_argument('--offload', action='store_true')
    p = add('legacy', cmd_legacy, 'old Raslila v1 commands (python run.py legacy --help)')
    p.add_argument('rest', nargs=argparse.REMAINDER)

    a = ap.parse_args()
    if not getattr(a, 'command', None):
        ap.print_help()
        return 0
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    return a.fn(a) or 0


if __name__ == '__main__':
    sys.exit(main())
