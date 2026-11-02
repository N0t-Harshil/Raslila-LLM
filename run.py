"""Raslila: one entry point. Every command logs to logs/<command>_<time>.log.

    python run.py <command> [options]        python run.py <command> --help

THE PLAN (scaling study: does the 3D brain grid beat a standard transformer?)
    1. python run.py setup            install requirements, check packages, list GPUs
    2. python run.py check            validate the study configs (fair pairs, sizes, cost)
    3. python run.py smoke            seconds-long runs of both models (no data needed)
    4. python run.py data             build the general+math+code corpus (once, cached)
    5. python run.py speed            FIRST GO/NO-GO: Raslila vs baseline tokens/s on your GPU
    6. python run.py study            train every size of both models on all GPUs
                                      (50M, 125M, 350M pairs one per GPU, then 1B on all 4)
    7. python run.py report           table + scaling plot + head-to-head verdict

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


def run(cmd, name, env=None):
    """Run a command, streaming output to the console AND a log file."""
    path = log_path(name)
    print('>>> ' + ' '.join(cmd) + '\n    (log: {})'.format(os.path.relpath(path, ROOT)), flush=True)
    full_env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', **(env or {}))
    with open(path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=full_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                errors='replace', bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            print('\n[interrupted -- run the same command again to resume]')
            return 130
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


def cmd_study(a):
    """Schedule the whole study over the GPUs. Small tiers: one run per GPU in
    parallel. Tiers in --multi-gpu-tiers: one run at a time on all GPUs."""
    gpus = a.gpus or n_gpus()
    if gpus < 1:
        sys.exit('No GPUs visible.')
    tiers = [t for t in TIERS if t in a.tiers.split(',')]
    multi = set(a.multi_gpu_tiers.split(',')) if gpus > 1 else set()
    single = [(f, t) for t in tiers if t not in multi for f in FAMILIES]
    big = [(f, t) for t in tiers if t in multi for f in FAMILIES]
    todo = []
    for f, t in single + big:
        st = run_status('{}_{}'.format(f, t))
        if st and st.get('finished'):
            print('done already: {}_{} (val {:.4f})'.format(f, t, st['val_loss']))
        else:
            todo.append((f, t))
    banner('Study on {} GPU(s): {}'.format(gpus, ', '.join('{}_{}'.format(f, t) for f, t in todo) or 'nothing to do'))
    extra = ['--compile'] if a.compile else []

    def launch(fam, tier, devices):
        cfg = 'configs/scaling/{}_{}.yaml'.format(fam, tier)
        path = log_path('study_{}_{}'.format(fam, tier))
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                   CUDA_VISIBLE_DEVICES=','.join(map(str, devices)))
        f = open(path, 'w', encoding='utf-8')
        p = subprocess.Popen(train_cmd(cfg, len(devices), extra), cwd=ROOT, env=env,
                             stdout=f, stderr=subprocess.STDOUT)
        print('[{}] START {}_{} on GPU {} -> {}'.format(time.strftime('%H:%M'), fam, tier,
                                                        ','.join(map(str, devices)), os.path.relpath(path, ROOT)),
              flush=True)
        return p, f, path

    def tail(path):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = [l.strip() for l in f.readlines()[-30:] if l.startswith('step')]
            return lines[-1] if lines else ''
        except OSError:
            return ''

    failed = []
    queue = [j for j in todo if j[1] not in multi]
    running = {}                                   # gpu -> (job, proc, file, path)
    last_report = time.time()
    try:
        while queue or running:
            for g in range(gpus):
                if g not in running and queue:
                    job = queue.pop(0)
                    running[g] = (job,) + launch(job[0], job[1], [g])
            time.sleep(20)
            for g, (job, p, f, path) in list(running.items()):
                if p.poll() is not None:
                    f.close()
                    del running[g]
                    if p.returncode:
                        failed.append(job)
                        print('[{}] FAILED {}_{} (exit {}), see {}'.format(
                            time.strftime('%H:%M'), job[0], job[1], p.returncode, os.path.relpath(path, ROOT)))
                    else:
                        st = run_status('{}_{}'.format(*job)) or {}
                        print('[{}] DONE {}_{}: val loss {}'.format(time.strftime('%H:%M'), job[0], job[1],
                                                                  st.get('val_loss')), flush=True)
            if time.time() - last_report > 600 and running:
                last_report = time.time()
                for g, (job, _, _, path) in running.items():
                    print('  GPU {} {}_{}: {}'.format(g, job[0], job[1], tail(path)), flush=True)
        for job in [j for j in todo if j[1] in multi]:
            p, f, path = launch(job[0], job[1], list(range(gpus)))
            while p.poll() is None:
                time.sleep(600)
                print('  {}_{}: {}'.format(job[0], job[1], tail(path)), flush=True)
            f.close()
            if p.returncode:
                failed.append(job)
                print('FAILED {}_{}, see {}'.format(job[0], job[1], os.path.relpath(path, ROOT)))
            else:
                print('DONE {}_{}: val loss {}'.format(job[0], job[1], (run_status('{}_{}'.format(*job)) or {}).get('val_loss')))
    except KeyboardInterrupt:
        print('\nStopping all runs (they resume from their last checkpoint when you re-run `study`).')
        for job, p, f, _ in running.values():
            p.terminate()
        return 130
    cmd_report(a)
    return 1 if failed else 0


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
    p = add('study', cmd_study, 'train every tier of both families across all GPUs')
    p.add_argument('--gpus', type=int, default=None, help='default: all visible')
    p.add_argument('--tiers', default='50m,125m,350m,1b')
    p.add_argument('--multi-gpu-tiers', default='1b', help='tiers trained on all GPUs at once')
    p.add_argument('--compile', action='store_true')
    add('report', cmd_report, 'scaling table, plot and verdict from finished runs')
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
