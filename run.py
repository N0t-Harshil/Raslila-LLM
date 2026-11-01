"""One entry point for everything: setup, data, training, benchmarking, chatting.

    python run.py <command> [options]          python run.py <command> --help

FIRST TIME ON THE A100 (in order)
    python run.py setup          install requirements, show GPU
    python run.py check          validate every config (must pass)
    python run.py data           download + clean + tokenize FineWeb-Edu (~3 GB, once)
    python run.py smoke          3-step training run: catches config/OOM problems in minutes

THE EXPERIMENTS
    python run.py ab             depth_sweeps A/B: bidirectional (2) vs feedforward (1)
    python run.py compare        verdict for the A/B (decision rule fixed in advance)
    python run.py train --config config_adaptive.yaml      adaptive-depth model
    python run.py eval  --config config_adaptive.yaml      score it: loss, speed, thinking cycles

EVERYTHING ELSE
    python run.py train          train any config (default config.yaml); --resume latest
    python run.py eval           evaluate the latest checkpoint of a config
    python run.py bench          architecture benchmarks (GPT vs Raslila, ablations)
    python run.py generate       sample text from a checkpoint
    python run.py live           always-on chat session (rests when idle, input mid-answer)
    python run.py all            check -> data -> smoke -> ab -> compare -> adaptive train -> eval

Every command that runs a script also writes its full output to logs/<command>_<time>.log.
Long runs over SSH: start them inside `tmux` or `screen` (or with `nohup ... &`) so a
dropped connection does not kill training. Interrupted training resumes with
`python run.py train --config <same config> --resume latest`.
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))
PY = sys.executable
AB_A = 'checkpoints/raslila_ab_a'
AB_B = 'checkpoints/raslila_ab_b'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rel(path, start=ROOT):
    """Path relative to the project if possible (Windows cannot cross drives)."""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


def banner(text):
    print('\n' + '=' * 78 + '\n' + text + '\n' + '=' * 78, flush=True)


def run(cmd, log_name):
    """Run a command, streaming its output to the console AND a log file."""
    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(ROOT, 'logs', '{}_{}.log'.format(log_name, stamp))
    print('>>> ' + ' '.join(cmd))
    print('    (log: {})'.format(rel(log_path, ROOT)), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            print('\n[interrupted -- resume training with --resume latest]')
            return 130
    if proc.returncode != 0:
        print('\n[{} exited with code {}; see {}]'.format(
            log_name, proc.returncode, rel(log_path, ROOT)))
    return proc.returncode


def load_yaml(path):
    import yaml
    with open(os.path.join(ROOT, path), encoding='utf-8') as f:
        return yaml.safe_load(f)


def out_dir_of(config):
    return (load_yaml(config).get('training') or {}).get('out_dir', 'checkpoints/raslila')


def latest_checkpoint(out_dir):
    """Highest-step raslila_step*.pt in out_dir, or None."""
    best, best_step = None, -1
    for path in glob.glob(os.path.join(ROOT, out_dir, 'raslila_step*.pt')):
        m = re.search(r'raslila_step(\d+)\.pt$', path)
        if m and int(m.group(1)) > best_step:
            best, best_step = path, int(m.group(1))
    return best


def resolve_checkpoint(args):
    if getattr(args, 'checkpoint', None) and args.checkpoint != 'latest':
        return args.checkpoint
    ck = latest_checkpoint(out_dir_of(args.config))
    if ck is None:
        sys.exit('No checkpoint found in {} -- train first, or pass --checkpoint PATH.'
                 .format(out_dir_of(args.config)))
    return ck


def tokenizer_of(config):
    from raslila.hf_dataset import resolve
    return resolve(load_yaml(config).get('dataset')).get('tokenizer')


def ok(rc, what):
    if rc != 0:
        sys.exit('STOPPED: {} failed (exit code {}).'.format(what, rc))


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
ENV_PROBE = r'''
import importlib, warnings
for name in ("numpy", "scipy", "pandas", "torch", "pyarrow", "datasets", "tokenizers", "yaml"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            m = importlib.import_module(name)
        except Exception as e:
            print("FAIL", name, type(e).__name__ + ":", str(e).splitlines()[0][:200])
            continue
    # e.g. torch built for NumPy 1.x warns at import and breaks later
    bad = [str(w.message).splitlines()[0][:200] for w in caught if "numpy" in str(w.message).lower()]
    if bad:
        print("FAIL", name, "NumPy warning:", bad[0])
    else:
        print("OK  ", name, getattr(m, "__version__", ""))
'''


def check_env():
    """Import every package training needs in a fresh interpreter; explain conflicts."""
    out = subprocess.run([PY, '-c', ENV_PROBE], cwd=ROOT, capture_output=True,
                         text=True, encoding='utf-8', errors='replace').stdout
    print(out.rstrip())
    fails = [ln for ln in out.splitlines() if ln.startswith('FAIL')]
    if not fails:
        return True
    if any('numpy' in ln.lower() for ln in fails):
        print('\nNumPy version conflict: a package was built for a different NumPy major\n'
              'version than the one installed. Most conda/Jupyter images build torch,\n'
              'scipy and pandas against NumPy 1.x, so keep NumPy 1 and pin pyarrow:\n'
              '  pip install "numpy<2" "pyarrow<20"   (add "datasets<4.1" if pip refuses)\n'
              'Only if every package above is NumPy-2 ready instead: pip install -U "numpy>=2"\n'
              'Cleanest: a fresh venv, then  pip install -r requirements.txt')
    else:
        print('\nSome packages failed to import; install them with:\n'
              '  pip install -r requirements.txt')
    return False


def cmd_setup(args):
    banner('Installing requirements')
    ok(run([PY, '-m', 'pip', 'install', '-r', 'requirements.txt'], 'setup'), 'pip install')
    banner('Checking packages import together')
    if not check_env():
        return 1
    banner('Hardware')
    import torch
    print('torch', torch.__version__)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print('GPU {}: {} ({:.0f} GB), bf16 {}'.format(
                i, p.name, p.total_memory / 1e9, torch.cuda.is_bf16_supported()))
    else:
        print('No CUDA GPU visible. Training will run on CPU (smoke tests only).')
        print('On the A100 install the CUDA build of torch: https://pytorch.org/get-started/locally/')
    return 0


def cmd_check(args):
    banner('Validating configs')
    return run([PY, 'scripts/check_configs.py'], 'check')


def cmd_data(args):
    banner('Checking packages import together')
    if not check_env():
        return 1
    banner('Preparing the dataset for {} (downloads once, then cached)'.format(args.config))
    cmd = [PY, 'scripts/get_dataset.py', '--config', args.config]
    if args.force:
        cmd.append('--force')
    if args.max_tokens:
        cmd += ['--max-tokens', str(args.max_tokens)]
    return run(cmd, 'data')


def cmd_smoke(args):
    banner('Smoke test: 3 training steps on {}'.format(args.config))
    return run([PY, 'scripts/train_raslila.py', '--config', args.config, '--steps', '3',
                '--batch-size', '1', '--seq-len', '32', '--out-dir', 'checkpoints/smoke',
                '--save-every', '0'], 'smoke')


def cmd_train(args):
    banner('Training {}'.format(args.config))
    cmd = [PY, 'scripts/train_raslila.py', '--config', args.config]
    if args.steps:
        cmd += ['--steps', str(args.steps)]
    if args.batch_size:
        cmd += ['--batch-size', str(args.batch_size)]
    if args.save_every:
        cmd += ['--save-every', str(args.save_every)]
    if args.out_dir:
        cmd += ['--out-dir', args.out_dir]
    if args.resume:
        ck = args.resume
        if ck == 'latest':
            ck = latest_checkpoint(args.out_dir or out_dir_of(args.config))
            if ck is None:
                print('No checkpoint to resume from -- starting fresh.')
        if ck:
            cmd += ['--resume', ck]
    if args.gpus > 1:
        cmd = ['torchrun', '--standalone', '--nproc_per_node={}'.format(args.gpus)] + cmd[1:]
    return run(cmd, 'train_' + os.path.splitext(os.path.basename(args.config))[0])


def cmd_ab(args):
    banner('A/B: config.yaml (depth_sweeps 2) vs config_baseline_feedforward.yaml (1)\n'
           'Arms run one after the other, {} GPU(s) each.'.format(args.gpus))
    cmd = [PY, 'scripts/run_ab.py', '--nproc', str(args.gpus)]
    if args.steps:
        cmd += ['--steps', str(args.steps)]
    return run(cmd, 'ab')


def cmd_compare(args):
    banner('A/B verdict')
    a, b = args.run_a or AB_A, args.run_b or AB_B
    cmd = [PY, 'scripts/compare_runs.py', a, b]
    if args.min_steps:
        cmd += ['--min-steps', str(args.min_steps)]
    return run(cmd, 'compare')


def cmd_bench(args):
    banner('Architecture benchmark: {}'.format(args.mode))
    return run([PY, 'scripts/benchmark_gpt_vs_raslila.py', '--mode', args.mode,
                '--steps', str(args.steps)], 'bench_' + args.mode)


def cmd_generate(args):
    ck = resolve_checkpoint(args)
    cmd = [PY, 'scripts/generate_raslila.py', '--checkpoint', ck, '--config', args.config,
           '--tokenizer', tokenizer_of(args.config), '--prompt', args.prompt,
           '--tokens', str(args.tokens), '--temperature', str(args.temperature)]
    return run(cmd, 'generate')


def cmd_live(args):
    ck = resolve_checkpoint(args)
    cmd = [PY, 'scripts/raslila_live.py', '--checkpoint', ck, '--config', args.config,
           '--tokenizer', tokenizer_of(args.config)]
    if args.inbox:
        cmd += ['--inbox', args.inbox]
    if args.offload:
        cmd.append('--offload')
    print('>>> ' + ' '.join(cmd), flush=True)
    # Interactive: hand the terminal over directly (no log tee).
    return subprocess.call(cmd, cwd=ROOT)


def cmd_eval(args):
    """Score a checkpoint on the held-out slice: loss, perplexity, speed, and
    (adaptive models) loss per thinking depth + how often it stops early."""
    import math
    import numpy as np
    import torch
    import torch.nn.functional as F
    from raslila.raslila import Raslila, RaslilaConfig
    from raslila.hf_dataset import resolve

    ck_path = resolve_checkpoint(args)
    raw = load_yaml(args.config)
    ds = resolve(raw.get('dataset'))
    seq_len = int((raw.get('training') or {}).get('seq_len', 128))
    val_tokens = int(ds.get('val_tokens') or 0)
    cache = os.path.join(ROOT, ds['cache'])
    if not os.path.exists(cache):
        sys.exit('Token cache {} not found -- run `python run.py data` first.'.format(cache))
    corpus = np.load(cache, mmap_mode='r')
    if val_tokens > seq_len + 1:
        val = corpus[:val_tokens]
        held_out = True
    else:
        val = corpus[-2_000_000:]
        held_out = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ck_path, map_location='cpu')
    model = Raslila(RaslilaConfig(**ckpt['config']))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    use_bf16 = device.type == 'cuda'
    max_think = int(model.config.max_think_cycles or 0)

    banner('Evaluating {}\n  step {}, {:.1f}M params, {} on {}{}'.format(
        rel(ck_path, ROOT), ckpt.get('step'),
        sum(p.numel() for p in model.parameters()) / 1e6,
        'adaptive depth (max {} cycles)'.format(max_think) if max_think else
        'fixed depth ({} sweeps)'.format(model.config.depth_sweeps), device,
        '' if held_out else '\n  WARNING: dataset.val_tokens is 0, so this uses the cache tail, '
                            'which training may have seen'))

    n = max(1, min(args.windows, (len(val) - seq_len - 1) // max(1, seq_len)))
    starts = np.linspace(0, len(val) - seq_len - 1, num=n, dtype=np.int64)
    idx = starts[:, None] + np.arange(seq_len)[None, :]
    X = torch.from_numpy(np.asarray(val[idx]).astype(np.int64))
    Y = torch.from_numpy(np.asarray(val[idx + 1]).astype(np.int64))

    def mean_loss(**kw):
        total, count = 0.0, 0
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            for i in range(0, n, args.batch_size):
                x, y = X[i:i + args.batch_size].to(device), Y[i:i + args.batch_size].to(device)
                logits = model(x, **kw)['logits'].float()
                total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                         ignore_index=0, reduction='sum').item()
                count += int((y != 0).sum())
        return total / max(1, count)

    results = {'checkpoint': rel(ck_path, ROOT), 'step': ckpt.get('step'),
               'windows': int(n), 'seq_len': seq_len, 'held_out': held_out}
    full = mean_loss(halt=False)
    results['val_loss'] = full
    results['perplexity'] = math.exp(min(full, 50))
    print('val loss (full depth): {:.4f}   perplexity: {:.1f}   ({} windows x {} tokens)'.format(
        full, results['perplexity'], n, seq_len))

    # speed: forward tokens per second at the eval batch size
    x = X[:args.batch_size].to(device)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        model(x, halt=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 3
        for _ in range(reps):
            model(x, halt=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    tps = reps * x.numel() / (time.perf_counter() - t0)
    results['forward_tokens_per_s'] = tps
    print('speed: {:,.0f} tokens/s forward (batch {} x {})'.format(tps, x.size(0), seq_len))

    if max_think:
        print('\nthinking depth -> val loss (lower = better; does more thinking help?)')
        per = {}
        for c in range(1, max_think + 1):
            per[c] = mean_loss(think_cycles=c, halt=False)
            print('  {} cycle{}: {:.4f}'.format(c, 's' if c > 1 else ' ', per[c]))
        results['loss_by_cycles'] = per
        # Halting is decided on the LAST position, so measure it per window
        # (batch 1) on the token each window is about to produce.
        used, nll_halt, nll_full = [], 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            for i in range(min(n, args.halt_windows)):
                x, y = X[i:i + 1].to(device), Y[i, -1].item()
                out = model(x, halt=True)
                used.append(out['cycles_used'])
                nll_halt += -torch.log_softmax(out['logits'][0, -1].float(), -1)[y].item()
                nll_full += -torch.log_softmax(model(x, halt=False)['logits'][0, -1].float(), -1)[y].item()
        k = len(used)
        hist = {c: used.count(c) for c in range(1, max_think + 1)}
        results.update(avg_cycles_used=sum(used) / k, cycles_histogram=hist,
                       last_token_loss_halting=nll_halt / k, last_token_loss_full=nll_full / k)
        print('\nwith early stopping ({} samples): avg {:.2f} of {} cycles used'.format(
            k, sum(used) / k, max_think))
        for c, cnt in hist.items():
            print('  stopped after {} cycle{}: {:5.1%} {}'.format(
                c, 's' if c > 1 else ' ', cnt / k, '#' * int(40 * cnt / k)))
        print('  next-token loss: early-stopping {:.4f} vs always-full {:.4f} '
              '(close = stopping early costs little)'.format(nll_halt / k, nll_full / k))

    out_path = os.path.join(os.path.dirname(ck_path), 'eval_step{}.json'.format(ckpt.get('step')))
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('\nsaved: {}'.format(rel(out_path, ROOT)))
    return 0


def cmd_all(args):
    steps = args.steps
    for name, fn, extra in [
            ('check', cmd_check, {}),
            ('data', cmd_data, {'force': False, 'max_tokens': None}),
            ('smoke', cmd_smoke, {}),
            ('A/B', cmd_ab, {'steps': steps}),
            ('compare', cmd_compare, {'run_a': None, 'run_b': None, 'min_steps': steps}),
    ]:
        ns = argparse.Namespace(config='config.yaml', gpus=args.gpus, **extra)
        ok(fn(ns), name)
    ns = argparse.Namespace(config='config_adaptive.yaml', steps=steps, batch_size=None,
                            save_every=None, out_dir=None, resume=None, gpus=args.gpus)
    ok(cmd_train(ns), 'adaptive training')
    ns = argparse.Namespace(config='config_adaptive.yaml', checkpoint='latest',
                            windows=128, batch_size=8, halt_windows=256)
    ok(cmd_eval(ns), 'adaptive eval')
    banner('All done. A/B verdict above; adaptive eval saved next to its checkpoint.')
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', metavar='command')

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        return p

    add('setup', cmd_setup, 'install requirements and show the GPU')
    add('check', cmd_check, 'validate every config (must pass before a run)')
    p = add('data', cmd_data, 'download, clean and tokenize the corpus (cached)')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--force', action='store_true', help='rebuild even if cached')
    p.add_argument('--max-tokens', type=int, default=None, help='smaller slice for testing')
    p = add('smoke', cmd_smoke, '3-step training run to catch problems early')
    p.add_argument('--config', default='config.yaml')
    p = add('train', cmd_train, 'train a config')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--steps', type=int, default=None, help='total steps (default: config)')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--save-every', type=int, default=None)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--resume', default=None, help='checkpoint path, or "latest"')
    p.add_argument('--gpus', type=int, default=1)
    p = add('ab', cmd_ab, 'run the depth_sweeps A/B (both arms)')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--gpus', type=int, default=1)
    p = add('compare', cmd_compare, 'A/B verdict from the two runs')
    p.add_argument('run_a', nargs='?', default=None, help='default ' + AB_A)
    p.add_argument('run_b', nargs='?', default=None, help='default ' + AB_B)
    p.add_argument('--min-steps', type=int, default=None)
    p = add('eval', cmd_eval, 'score a checkpoint: loss, speed, thinking cycles')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--checkpoint', default='latest')
    p.add_argument('--windows', type=int, default=128, help='validation windows')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--halt-windows', type=int, default=256,
                   help='samples for the early-stopping measurement')
    p = add('bench', cmd_bench, 'architecture benchmarks (short, small models)')
    p.add_argument('--mode', default='gpt_vs_raslila',
                   choices=['gpt_vs_raslila', 'legacy_vs_modern', 'plus', 'cortex', 'recur', 'bind'])
    p.add_argument('--steps', type=int, default=200)
    p = add('generate', cmd_generate, 'sample text from a checkpoint')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--checkpoint', default='latest')
    p.add_argument('--prompt', default='The most important idea in science is')
    p.add_argument('--tokens', type=int, default=150)
    p.add_argument('--temperature', type=float, default=0.8)
    p = add('live', cmd_live, 'always-on chat session')
    p.add_argument('--config', default='config_adaptive.yaml')
    p.add_argument('--checkpoint', default='latest')
    p.add_argument('--inbox', default='data/live_inbox')
    p.add_argument('--offload', action='store_true', help='move weights to CPU while resting')
    p = add('all', cmd_all, 'check -> data -> smoke -> A/B -> compare -> adaptive -> eval')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--gpus', type=int, default=1)

    args = ap.parse_args()
    if not getattr(args, 'command', None):
        ap.print_help()
        return 0
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    return args.fn(args) or 0


if __name__ == '__main__':
    sys.exit(main())
