"""Train Raslila (classical 3D-grid model) from config.yaml.

Supports:
- Single GPU / multi-GPU (torchrun, NCCL or gloo backend)
- bf16/fp16 mixed precision
- Gradient accumulation
- Loss-adaptive LR (warmup ramp + ReduceLROnPlateau-style decay on loss)
- Full per-step logging: lr, loss, alpha, beta, grad_norm (console + train_log.csv)
- Checkpointing
- Next-token language modeling loss (full-parameter training)

Usage:
    python scripts/train_raslila.py                     # single GPU / CPU
    torchrun --nproc_per_node=N scripts/train_raslila.py  # multi-GPU
"""
import sys
import os
import time
import random
import hashlib
import argparse
import csv
import dataclasses

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

from raslila.raslila import Raslila, RaslilaConfig, TransformerNode
from raslila.hf_dataset import ensure_dataset



def available_host_memory_bytes():
    if sys.platform.startswith('linux'):
        available = None
        try:
            with open('/proc/meminfo', encoding='ascii') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        available = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
        try:
            with open('/sys/fs/cgroup/memory.max', encoding='ascii') as f:
                limit_text = f.read().strip()
            with open('/sys/fs/cgroup/memory.current', encoding='ascii') as f:
                current = int(f.read().strip())
            if limit_text != 'max':
                cgroup_available = max(0, int(limit_text) - current)
                available = cgroup_available if available is None else min(available, cgroup_available)
        except (OSError, ValueError):
            pass
        return available
    return None


def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    b3 = raw.get('raslila', {})
    tr = raw.get('training', {})
    return raw, b3, tr


def tokenize(text):
    """Byte-level tokens (0-255) mapped into the model vocab."""
    tokens = [b + 1 for b in text.encode('utf-8', errors='replace')]  # 1..256
    return tokens


TOKENIZER_FILE = os.path.join('data', 'tokenizer', 'raslila_bpe.json')


def load_bpe(path=TOKENIZER_FILE):
    """Load the trained BPE tokenizer (id 0 = <|pad|>)."""
    from tokenizers import Tokenizer
    if not os.path.exists(path):
        raise FileNotFoundError(
            'BPE tokenizer not found at {}. Train it first:\n'
            '  python scripts/train_tokenizer.py --vocab-size 199995'.format(path))
    return Tokenizer.from_file(path)


def tokenize_corpus_bpe(tok, data_dirs, cache_path=None, block=2000):
    """Encode the whole corpus to a flat integer array, cached on disk.

    Byte-level encoding of 313 MB is far too slow per epoch; BPE + a cached
    array makes window sampling trivial and fast. Uses uint16 when the vocab
    fits (<= 65535, halving cache I/O) and uint32 otherwise, so a large-vocab
    tokenizer cannot silently overflow.
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    dtype = np.uint16 if tok.get_vocab_size() <= 65535 else np.uint32
    if isinstance(data_dirs, str):
        data_dirs = [d.strip() for d in data_dirs.split(',') if d.strip()]
    files = []
    for dd in data_dirs:
        if not os.path.isdir(dd):
            raise ValueError('Data dir not found: {}'.format(dd))
        for root, _dirs, names in os.walk(dd):
            for f in sorted(names):
                if f.lower().endswith('.txt'):
                    files.append(os.path.join(root, f))
    files = sorted(set(files))
    if not files:
        raise ValueError('No .txt files found in {}'.format(', '.join(data_dirs)))
    chunks = []
    total = 0
    for path in files:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        if not text:
            continue
        ids = tok.encode(text).ids
        chunks.append(np.asarray(ids, dtype=dtype))
        total += len(ids)
    arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=dtype)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, arr)
    return arr


def legacy_cache_path(data_dir):
    """Per-corpus cache path for the `training.data_dir` fallback.

    This was a single hardcoded `data/corpus_tokens.npy`, so pointing data_dir at
    a second corpus silently reused the first one's tokens -- you would train on
    the wrong data with no warning, only a suspicious loss curve. Key the cache
    on the paths actually requested.
    """
    key = ','.join(d.strip() for d in data_dir.split(',') if d.strip())
    return os.path.join('data', 'corpus_tokens_{}.npy'.format(
        hashlib.md5(key.encode('utf-8')).hexdigest()[:8]))


def load_corpus(data_dirs, max_bytes=None):
    """Load all *.txt files from one or more data dirs (recursive) into a token array.

    Args:
        data_dirs: str or list of paths (also accepts comma-separated string)
        max_bytes: optional per-file cap (None = read entire file)
    """
    if isinstance(data_dirs, str):
        data_dirs = [d.strip() for d in data_dirs.split(',') if d.strip()]
    files = []
    for dd in data_dirs:
        if not os.path.isdir(dd):
            raise ValueError('Data dir not found: {}'.format(dd))
        for root, _dirs, names in os.walk(dd):
            for f in sorted(names):
                if f.lower().endswith('.txt'):
                    files.append(os.path.join(root, f))
    files = sorted(set(files))
    if not files:
        raise ValueError('No .txt files found in {}'.format(', '.join(data_dirs)))
    tokens = []
    for path in files:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        if max_bytes is not None:
            text = text[:max_bytes]
        tokens.extend(tokenize(text))
    return tokens


def make_batch(tokens, batch_size, seq_len, step, steps):
    """Sample random contiguous windows from the corpus.

    Works with a Python list (byte-level) or a numpy uint16/uint32 array
    Batches are built vectorised in the numpy case; this is on the hot path.
    """
    n = len(tokens)
    starts = torch.randint(0, max(1, n - seq_len - 1), (batch_size,))
    inp = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, seq_len, dtype=torch.long)
    if isinstance(tokens, np.ndarray):
        arr = tokens
        st = starts.numpy()
        offs = np.arange(seq_len)
        idx = st[:, None] + offs[None, :]
        idx = np.clip(idx, 0, max(0, len(arr) - 1))
        inp = torch.from_numpy(arr[idx].astype(np.int64))
        idx2 = np.clip(st[:, None] + offs[None, :] + 1, 0, max(0, len(arr) - 1))
        tgt = torch.from_numpy(arr[idx2].astype(np.int64))
        return inp, tgt
    for i, s in enumerate(starts.tolist()):
        chunk = tokens[s:s + seq_len + 1]
        inp[i, :len(chunk) - 1] = torch.tensor(chunk[:-1], dtype=torch.long)
        tgt[i, :len(chunk) - 1] = torch.tensor(chunk[1:], dtype=torch.long)
    return inp, tgt


def save_checkpoint(model, optimizer, scheduler, step, out_dir, is_main,
                    best_loss=None, plateau_streak=None, lr=None,
                    fsdp=False):
    """Write a checkpoint.

    FSDP note: `model.state_dict()` on a FSDP-wrapped module returns each
    parameter's LOCAL shard. A checkpoint saved that way is not a model --
    loading it back into an unwrapped module gives you 1/N of the weights with
    the rest silently zero. Gathering to full params is a collective op, so
    every rank must enter this function; only rank 0 writes.
    """
    if fsdp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, StateDictType,
            FullStateDictConfig, FullOptimStateDictConfig)
        # full_state_dict offloads to CPU, so a 20B gather does not OOM the
        # GPU it came from; rank0_only means only rank 0 materializes it.
        sd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        osd_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                  state_dict_config=sd_cfg,
                                  optim_state_dict_config=osd_cfg):
            m_sd = model.state_dict()
            o_sd = FSDP.optim_state_dict(model, optimizer)
        if not is_main:
            return
        m_cfg = model.module.config if hasattr(model, 'module') else model.config
    else:
        if not is_main:
            return
        m_sd = model.state_dict()
        o_sd = optimizer.state_dict()
        m_cfg = model.config

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'raslila_step{}.pt'.format(step))
    torch.save({
        'model': m_sd,
        'config': dataclasses.asdict(m_cfg),
        'optimizer': o_sd,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'step': step,
        'best_loss': best_loss,
        'plateau_streak': plateau_streak,
        'lr': lr,
    }, (path))
    print('Saved: {}'.format(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--steps', type=int, default=None,
                        help='total steps (default: training.max_steps from config)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='checkpoint dir (default: training.out_dir from config)')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--lr-patience', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default=None,
                        help='corpus dir (default: training.data_dir from config)')
    parser.add_argument('--seq-len', type=int, default=None,
                        help='context window length in tokens (default: training.seq_len)')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint .pt to resume training from')
    parser.add_argument('--mixed-precision', type=str, default=None,
                        choices=['no', 'fp16', 'bf16'],
                        help='override config precision (bf16 needs Ampere+; use fp16 on T4)')
    parser.add_argument('--save-every', type=int, default=None,
                        help='checkpoint interval in steps (default: training.save_every)')
    parser.add_argument('--tok-mode', type=str, default=None,
                        choices=['bytes', 'bpe'],
                        help='tokenization (default: training.tok_mode)')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='path to trained BPE tokenizer json '
                             '(default: dataset.tokenizer from config, else '
                             'data/tokenizer/raslila_bpe.json)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fsdp', action='store_true',
                        help='wrap the model in FSDP (FULL_SHARD) instead of DDP. '
                             'Required above ~5B params: shards params, grads and '
                             'optimizer state across ranks so the model fits.')
    args = parser.parse_args()

    raw, b3, tr = load_config(args.config)
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main = rank == 0

    # Seed PER RANK, and only after rank is known. Previously this ran before
    # `rank` was read and used the same seed on every process, so all ranks
    # drew the identical batch from make_batch and DDP averaged four copies of
    # one gradient: 4x the compute for 1x the data. Data parallel is only
    # data parallel if the ranks see different data.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    # Python's `random` is a separate stream from torch/numpy; replay sampling
    # uses it, so offset that too or replayed batches also repeat per rank.
    random.seed(args.seed + rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda:{}'.format(local_rank))
    else:
        device = torch.device('cpu')

    # Distributed init (gloo also works on CPU; nccl for multi-GPU)
    use_dist = world_size > 1
    if use_dist:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch.distributed.init_process_group(
            backend=backend, init_method='env://',
            rank=rank, world_size=world_size)

    steps = args.steps if args.steps is not None else int(tr.get('max_steps', 1000))
    batch_size = args.batch_size if args.batch_size is not None else int(tr.get('batch_size', 8))
    seq_len = args.seq_len if args.seq_len is not None else int(tr.get('seq_len', 32))
    if args.out_dir is None:
        args.out_dir = tr.get('out_dir', 'checkpoints/raslila')
    if args.save_every is None:
        args.save_every = max(1, int(tr.get('save_every', 500)))
    else:
        args.save_every = max(0, args.save_every)
    data_dir = args.data_dir if args.data_dir is not None else tr.get('data_dir')
    tok_mode = args.tok_mode if args.tok_mode is not None else tr.get('tok_mode', 'bytes')

    # HuggingFace corpus (the default). Takes precedence over training.data_dir,
    # which is now only the fallback for a locally-curated .txt corpus.
    # `raslila/` is imported lazily so `--tok-mode bytes` still works if the
    # `datasets` package is not installed.
    ds_cfg = raw.get('dataset')
    ds_resolved = None
    if ds_cfg:
        from raslila.hf_dataset import resolve as resolve_dataset
        ds_resolved = resolve_dataset(ds_cfg)
        if args.tokenizer is None:
            args.tokenizer = ds_resolved['tokenizer']
        # The HF path is BPE-only: byte-level ids cannot encode a streaming
        # corpus's vocabulary and the vocab check below would reject it anyway.
        if tok_mode != 'bpe':
            if is_main:
                print('dataset: is set, forcing tok_mode bytes -> bpe')
            tok_mode = 'bpe'
        data_dir = None          # do not double-load the legacy corpus
    elif args.tokenizer is None:
        args.tokenizer = TOKENIZER_FILE
    lr = args.lr if args.lr is not None else float(tr.get('learning_rate', 3e-4))
    grad_clip = float(tr.get('grad_clip', 1.0))
    accum = max(1, int(tr.get('gradient_accumulation', tr.get('accumulation_steps', 1))))
    mixed_precision = tr.get('mixed_precision', 'no')
    if args.mixed_precision is not None:
        mixed_precision = args.mixed_precision

    model_cfg = RaslilaConfig(**b3)
    # Guard against a silent data/vocab mismatch: byte tokens are 1..256, so a
    # byte-level run with a large BPE vocab would train on garbage.
    if tok_mode == 'bytes' and model_cfg.vocab_size > 257:
        raise SystemExit(
            'Config mismatch: tok_mode="bytes" produces token ids 1..256 but '
            'raslila.vocab_size={}. Either set vocab_size: 257 for byte-level, '
            'or use tok_mode: bpe.'.format(model_cfg.vocab_size))
    if tok_mode == 'bpe' and model_cfg.vocab_size <= 257:
        raise SystemExit(
            'Config mismatch: tok_mode="bpe" needs a BPE-sized vocab but '
            'raslila.vocab_size={}. Train a tokenizer and set vocab_size to its '
            'size.'.format(model_cfg.vocab_size))
    twenty_b_family = (model_cfg.hidden_dim == 1024
                       and model_cfg.depth_layers * model_cfg.grid_h * model_cfg.grid_w == 125)
    if twenty_b_family and model_cfg.node_depth > 11:
        raise SystemExit('The 4x A100-80GB configuration supports node_depth <= 11; the requested shape exceeds the verified memory budget.')
    large_20b_shape = twenty_b_family and model_cfg.node_depth == 11
    if args.fsdp and not use_dist:
        raise SystemExit('--fsdp requires torchrun with WORLD_SIZE greater than 1.')
    if large_20b_shape and (not args.fsdp or world_size != 4):
        raise SystemExit('The shipped 20B shape requires torchrun --nproc_per_node=4 and --fsdp.')
    fsdp_cpu_init = bool(use_dist and args.fsdp)
    if fsdp_cpu_init and large_20b_shape:
        model_params_est = 17_820_462_270
        dataset_bytes = int((ds_cfg or {}).get('max_tokens', 0) or 0) * 4 * world_size
        host_required = int((model_params_est * 4 * world_size + dataset_bytes) * 1.10)
        host_available = available_host_memory_bytes()
        if host_available is not None and host_available < host_required:
            raise SystemExit('FSDP CPU initialization needs about {:.1f} GB available host memory (model replicas plus token cache and 10% reserve); only {:.1f} GB is available.'.format(host_required / 1e9, host_available / 1e9))
        if is_main and host_available is not None:
            print('FSDP CPU staging: estimated {:.1f} GB host memory required; {:.1f} GB available.'.format(host_required / 1e9, host_available / 1e9))
    model = Raslila(model_cfg) if fsdp_cpu_init else Raslila(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print('Raslila nodes: {}'.format(model.n_nodes))
        print('Raslila params: {:,} ({:.1f}M)'.format(n_params, n_params / 1e6))
        print('Device: {}'.format(device))
        print('Steps: {}, batch: {}, seq_len: {}, lr: {}, accum: {}'.format(
            steps, batch_size, seq_len, lr, accum))
        print('Grid: {}x{}x{}, hidden={}'.format(
            b3.get('dim_x'), b3.get('dim_y'), b3.get('dim_z'),
            b3.get('hidden_dim')))

    if use_dist and args.fsdp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy, MixedPrecision, CPUOffload, StateDictType,
            FullStateDictConfig, FullOptimStateDictConfig,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools

        # Shard per TransformerNode. Each node is one sharding unit, so its
        # weights are all-gathered only for the duration of that node's forward
        # and immediately released -- peak memory is one node, not the model.
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16 if use_bf16 else None,
            reduce_dtype=torch.bfloat16 if use_bf16 else None,
            buffer_dtype=torch.bfloat16 if use_bf16 else None,
        ) if use_bf16 else None
        offload = CPUOffload(offload_params=True) if tr.get('fsdp_cpu_offload', False) else None

        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={TransformerNode}),
            mixed_precision=mp_policy,
            cpu_offload=offload,
            device_id=local_rank if torch.cuda.is_available() else None,
            # Original param names survive wrapping, so state_dict keys match the
            # unwrapped model and --resume / generate_raslila.py keep working.
            use_orig_params=True,
            limit_all_gathers=True,
        )
        if is_main:
            print('FSDP: FULL_SHARD over {} ranks (per-node sharding)'.format(world_size))
    elif use_dist:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    params = list(model.parameters())
    # Under FSDP, optimizer-state sharding is handled by FSDP itself (it exposes
    # sharded flat params; AdamW over those is the standard recipe).
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=float(tr.get('weight_decay', 0.01)))
    warmup = max(1, int(tr.get('warmup_steps', 200)))
    if args.warmup is not None:
        warmup = args.warmup

    # Warmup followed by optional loss-adaptive plateau decay.
    adaptive_lr = bool(tr.get('adaptive_lr', True))
    lr_factor = float(tr.get('lr_factor', 0.5))
    lr_patience = int(tr.get('lr_patience', 200))
    if args.lr_patience is not None:
        lr_patience = args.lr_patience
    lr_threshold = float(tr.get('lr_threshold', 1e-3))
    # Hard floor for the plateau decay. Without it the decay is geometric and
    # unbounded: a 4000-step run at lr_patience=500 halves the LR 8 times,
    # 6e-4 -> 4.7e-6, and the last quarter of training runs frozen. `min_lr` is
    # documented in config.yaml but was never read by this path.
    min_lr = float(tr.get('min_lr', 0.0) or 0.0)
    best_loss = float('inf')
    plateau_streak = 0

    def cur_lr():
        return optimizer.param_groups[0]['lr']

    def set_lr(new_lr):
        for g in optimizer.param_groups:
            g['lr'] = float(new_lr)

    def adjust_lr(step_idx, win_avg):
        """Warmup ramp then plateau-aware decay based on the loss."""
        nonlocal best_loss, plateau_streak
        if step_idx < warmup:
            set_lr(lr * min(1.0, (step_idx + 1) / warmup))
            return
        if not adaptive_lr:
            set_lr(lr)
            return
        if win_avg < best_loss - lr_threshold * best_loss:
            best_loss = win_avg
            plateau_streak = 0
            return
        plateau_streak += 50  # one check per 50-step window
        if plateau_streak >= lr_patience:
            set_lr(max(cur_lr() * lr_factor, min_lr))
            plateau_streak = 0

    scaler = torch.amp.GradScaler('cuda') if (torch.cuda.is_available() and
                                             mixed_precision in ('fp16',)) else None
    use_bf16 = torch.cuda.is_available() and mixed_precision in ('bf16', 'true')

    # Resume from checkpoint if requested
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('lr') is not None:
            set_lr(ckpt['lr'])
        best_loss = float(ckpt.get('best_loss', best_loss))
        plateau_streak = int(ckpt.get('plateau_streak', plateau_streak))
        start_step = int(ckpt.get('step', 0)) + 1
        if is_main:
            print('Resumed from {} at step {} (lr {:.2e})'.format(
                args.resume, start_step - 1, cur_lr()))

    # Real-text corpus: byte-level (default) or BPE (3-4x token efficiency).
    vocab = model_cfg.vocab_size
    t0 = time.perf_counter()
    window_loss = 0.0
    window_size = 0
    grad_norm = 0.0
    csv_path = None
    corpus = None
    val_corpus = None
    if ds_resolved is not None:
        # Rank 0 downloads + tokenizes; everyone else waits, then loads. This
        # is the only place a network call happens, and it happens once.
        if is_main:
            ensure_dataset(ds_resolved)
        if use_dist:
            torch.distributed.barrier()
        corpus = np.load(ds_resolved['cache'])
        val_tokens = int(ds_cfg.get('val_tokens', 0) or 0)
        if val_tokens > 0:
            if val_tokens < seq_len + 1:
                raise SystemExit('dataset.val_tokens must be at least seq_len + 1.')
            if val_tokens >= len(corpus) - seq_len - 1:
                raise SystemExit('dataset.val_tokens must leave at least seq_len + 1 training tokens.')
            val_corpus = corpus[:val_tokens]
            corpus = corpus[val_tokens:]
        if is_main:
            print('Corpus (BPE): {:,} training tokens from {} [{}]'.format(
                len(corpus), ds_resolved['name'], ds_resolved['cache']))
            if val_corpus is not None:
                print('Validation: {:,} stable head tokens'.format(len(val_corpus)))
            from tokenizers import Tokenizer
            _tok = Tokenizer.from_file(args.tokenizer)
            print('Tokenizer: {} (vocab {})'.format(args.tokenizer, _tok.get_vocab_size()))
            if len(corpus):
                print('Corpus id range: [{}, {}] (raslila.vocab_size={})'.format(
                    int(corpus.min()), int(corpus.max()), vocab))
                if int(corpus.max()) >= vocab:
                    raise SystemExit(
                        'Token id {} exceeds raslila.vocab_size={}: the corpus '
                        'was tokenized with a larger vocabulary than the model '
                        'has embeddings for.'.format(int(corpus.max()), vocab))
    elif data_dir:
        if tok_mode == 'bpe':
            bpe = load_bpe(args.tokenizer)
            cache = legacy_cache_path(data_dir)
            corpus = tokenize_corpus_bpe(bpe, data_dir, cache_path=cache)
            if is_main:
                print('Corpus (BPE): {:,} tokens from {}'.format(len(corpus), data_dir))
                print('Tokenizer: {} (vocab {})'.format(args.tokenizer, bpe.get_vocab_size()))
        else:
            corpus = load_corpus(data_dir)
            if is_main:
                print('Corpus (bytes): {} tokens from {}'.format(len(corpus), data_dir))
    elif is_main:
        print('No corpus: neither `dataset:` nor training.data_dir is set. '
              'Training will only run if --resume supplies data-dependent state.')

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        csv_path = os.path.join(args.out_dir, 'train_log.csv')
        csv_fields = ['step', 'lr', 'loss', 'val_loss', 'alpha', 'beta', 'grad_norm', 'time_s']
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', encoding='utf-8', newline='') as cf:
                csv.writer(cf).writerow(csv_fields)
        else:
            with open(csv_path, 'r', encoding='utf-8', newline='') as cf:
                reader = csv.DictReader(cf)
                old_fields = reader.fieldnames or []
                old_rows = list(reader) if 'val_loss' not in old_fields else None
            if old_rows is not None:
                with open(csv_path, 'w', encoding='utf-8', newline='') as out:
                    writer = csv.DictWriter(out, fieldnames=csv_fields)
                    writer.writeheader()
                    for old_row in old_rows:
                        writer.writerow(old_row)

    eval_every = max(0, int(tr.get('eval_every', 0) or 0))

    def evaluate_validation():
        if val_corpus is None:
            return None
        was_training = model.training
        model.eval()
        total = 0.0
        count = 0
        starts = np.linspace(0, max(0, len(val_corpus) - seq_len - 1),
                             num=min(16, max(1, len(val_corpus) // max(1, seq_len))),
                             dtype=np.int64)
        with torch.no_grad():
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset:offset + batch_size]
                if len(batch_starts) < batch_size:
                    batch_starts = np.pad(batch_starts, (0, batch_size - len(batch_starts)), mode='edge')
                offsets = np.arange(seq_len)
                ix = batch_starts[:, None] + offsets[None, :]
                x = torch.from_numpy(val_corpus[ix].astype(np.int64)).to(device)
                y = torch.from_numpy(val_corpus[ix + 1].astype(np.int64)).to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(x)['logits']
                    total += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                             y.reshape(-1), ignore_index=0).item()
                count += 1
        if was_training:
            model.train()
        return total / max(1, count)

    grow_every = int(getattr(model_cfg, 'grow_every', 0) or 0)
    grow_min_scale = float(getattr(model_cfg, 'grow_min_scale', 0.004) or 0.004)
    grow_step = int(getattr(model_cfg, 'grow_step', 2) or 2)
    grow_max_nodes = int(getattr(model_cfg, 'grow_max_nodes', 64) or 64)
    grow_best = float('inf')

    # ---- Learning from experience ----------------------------------------
    # Humans revisit what they have seen: sleep replay re-runs the day's
    # episodes at 10-20x, and it is what consolidates temporary memories and
    # prevents catastrophic forgetting. Without it we only ever see each passage
    # once, which is the standard recipe for forgetting. `replay_prob` resamples
    # text already consumed; `replay_weight` lets it count more than new text.
    replay_prob = float(getattr(model_cfg, 'replay_prob', 0.0) or 0.0)
    replay_weight = float(getattr(model_cfg, 'replay_weight', 1.0) or 1.0)
    replay_seen = 0.0     # fraction of the corpus already consumed
    replay_stats = {'hits': 0, 'steps': 0}

    for step in range(start_step, start_step + steps):
        gstep = step + 1  # 1-indexed global step for warmup math
        # Experience replay: draw from EARLIER in the corpus rather than only
        # from fresh text, so knowledge gets revisited instead of overwritten.
        replayed = False
        if corpus is not None and replay_prob > 0 and replay_seen > 0.0:
            if random.random() < replay_prob:
                start = random.randint(0, max(0, int(len(corpus) * replay_seen) - 1))
                offs = random.randint(0, max(0, len(corpus) - start - (batch_size + 1) * seq_len))
                if start + offs + (batch_size + 1) * seq_len <= len(corpus):
                    rows, nxt = [], []
                    for b in range(batch_size):
                        i = start + offs + b * seq_len
                        row = corpus[i:i + seq_len]
                        if len(row) < 2:
                            continue
                        rows.append(torch.tensor(row, dtype=torch.long))
                        nxt.append(torch.tensor(corpus[i + 1:i + 1 + seq_len],
                                               dtype=torch.long))
                    if rows:
                        input_ids = torch.stack(rows)
                        target_ids = torch.stack(nxt)
                        replayed = True
                        replay_stats['hits'] += 1
        if not replayed:
            if corpus is not None:
                input_ids, target_ids = make_batch(
                    corpus, batch_size, seq_len, step, steps)
            else:
                # Random mini-batches over the vocabulary
                input_ids = torch.randint(1, min(vocab, 4096), (batch_size, seq_len)).to(device)
                target_ids = torch.randint(1, min(vocab, 4096), (batch_size, seq_len)).to(device)
        replay_stats['steps'] += 1
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        model.train()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            out = model(input_ids)
            logits = out['logits']
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,
            )
            # MTP auxiliary loss (hierarchical forecasting): predict t+2..t+K
            # from the final representation. Mirrors full_train_step so the
            # mtp_steps/mtp_weight config actually takes effect in training.
            mref = model.module if use_dist else model
            mtp_w = float(getattr(model_cfg, 'mtp_weight', 0.0) or 0.0)
            if mtp_w > 0 and len(mref.mtp_heads) > 0:
                final_repr = out['final_repr']
                aux = []
                for k, head in enumerate(mref.mtp_heads, start=2):
                    if final_repr.size(1) <= k:
                        break
                    aux.append(F.cross_entropy(
                        head(final_repr[:, :-k]).reshape(-1, logits.size(-1)),
                        target_ids[:, k:].reshape(-1),
                        ignore_index=0,
                    ))
                if aux:
                    loss = loss + mtp_w * torch.stack(aux).mean()
            # Replayed text is weighted up: consolidating an old memory is worth
            # more than a single fresh exposure, which is what makes revisiting
            # actually counteract forgetting instead of just diluting it.
            if replayed and replay_weight != 1.0:
                loss = loss * replay_weight
            loss = loss / accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        last_step = (gstep == start_step + steps)
        if gstep % accum == 0 or last_step:
            # Warmup ramp applies at every optimizer step; plateau decay every window
            if step < warmup:
                set_lr(lr * min(1.0, gstep / warmup))
            if scaler is not None:
                scaler.unscale_(optimizer)
            # FSDP shards gradients, so each rank holds only its slice and a
            # naive clip_grad_norm_ would compute the norm of a fragment and
            # clip by the wrong amount. FSDP's own clip does the all-reduce.
            if use_dist and args.fsdp:
                grad_norm = torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_(
                    model.parameters(), grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        window_loss += loss.item() * accum
        window_size += 1

        if corpus is not None:
            # Track how much of the corpus we have consumed, so replay can draw
            # only from genuinely already-seen text.
            replay_seen = min(1.0, (step - start_step + 1) * batch_size * seq_len
                              / max(1, len(corpus)))

        # The LR schedule must run on EVERY rank, not just rank 0. It used to
        # sit inside the `is_main` block below, so ranks 1..N-1 kept whatever
        # LR they finished warmup with while rank 0 decayed on plateaus. The
        # replicas then drifted apart and the gradients DDP averaged no longer
        # came from identical models. Deciding the LR is deterministic given
        # the loss, so every rank computing it independently is safe and
        # needs no collective communication.
        if step % 50 == 0 or last_step:
            adjust_lr(step, window_loss / max(1, window_size))

        eval_due = (val_corpus is not None and eval_every > 0
                    and (gstep % eval_every == 0 or last_step))
        val_loss = evaluate_validation() if eval_due else None

        if is_main and (step % 50 == 0 or last_step):
            win_avg = window_loss / max(1, window_size)

            m = model.module if use_dist else model
            alpha = float(m.alpha.data if hasattr(m, 'alpha') else 0.0)
            beta = float(m.beta.data if hasattr(m, 'beta') else 0.0)
            dt = time.perf_counter() - t0
            gn = float(grad_norm)
            rp = '{:.0%}'.format(replay_stats['hits'] / max(1, replay_stats['steps']))
            tag = ' [replay]' if replayed else ''
            print('Step {:5d} | lr {:.2e} | loss {:.4f} | val {} | alpha {:+.3f} | beta {:+.3f} | grad_norm {:.4f} | {:.1f}s | replay {}{}'.format(
                gstep - 1, cur_lr(), win_avg, 'n/a' if val_loss is None else '{:.4f}'.format(val_loss), alpha, beta, gn, dt, rp, tag))
            if csv_path:
                with open(csv_path, 'a', encoding='utf-8') as cf:
                    cf.write('{},{:.6e},{:.6f},{},{:.6f},{:.6f},{:.6f},{:.2f}\n'.format(
                        gstep - 1, cur_lr(), win_avg, '' if val_loss is None else '{:.6f}'.format(val_loss), alpha, beta, gn, dt))

            # Cortical growth: on a plateau, add regions (a brain grows).
            # FSDP cannot grow: new parameters cannot be inserted into an
            # already-wrapped module, so this path is disabled under FSDP.
            if grow_every > 0 and (gstep % grow_every == 0) and not last_step:
                rel_improve = (grow_best - win_avg) / max(abs(grow_best), 1e-8)
                grow_best = min(grow_best, win_avg)
                if win_avg < grow_best * (1.0 - grow_min_scale) and gstep > warmup:
                    m = model.module if use_dist else model
                    added = m.grow_regions(grow_step, optimizer=optimizer)
                    if added:
                        print('  plateau at {:.4f} -> grew {} region(s), now {}'.format(
                            win_avg, added, m.n_nodes), flush=True)
            window_loss = 0.0
            window_size = 0

        # Save on the save_every boundary, but skip it when the last step already
        # saved (they collided and wrote the same ~2GB checkpoint twice).
        #
        # FSDP: the full-state-dict gather is a COLLECTIVE -- every rank must
        # enter save_checkpoint or rank 0 hangs forever. So the `is_main` gate
        # is dropped here (save_checkpoint itself only writes on rank 0), and
        # the FSDP wrapper is passed rather than model.module, because
        # model.module holds the local shard, not the full weights.
        save_due = (args.save_every > 0 and gstep % args.save_every == 0
                    and not last_step)
        if save_due and (is_main or (use_dist and args.fsdp)):
            save_checkpoint(model if (use_dist and args.fsdp)
                            else (model.module if use_dist else model),
                            optimizer, None, gstep - 1, args.out_dir, is_main,
                            best_loss=best_loss, plateau_streak=plateau_streak,
                            lr=cur_lr(), fsdp=(use_dist and args.fsdp))

    if is_main or (use_dist and args.fsdp):
        # Checkpoint filenames are ZERO-INDEXED (step 0 = after the first update),
        # so N total steps produces raslila_step{N-1}.pt.
        last_saved = (gstep - 1) if steps > 0 else start_step
        save_checkpoint(model if (use_dist and args.fsdp)
                        else (model.module if use_dist else model),
                        optimizer, None, last_saved, args.out_dir, is_main,
                        best_loss=best_loss, plateau_streak=plateau_streak,
                        lr=cur_lr(), fsdp=(use_dist and args.fsdp))
        if is_main:
            print('\nDone in {:.1f}s'.format(time.perf_counter() - t0))

    if use_dist:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()