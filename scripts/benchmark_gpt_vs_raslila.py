"""Head-to-head: GPT-style stacked transformer vs Raslila, same data/seed/steps.

Both models trained on identical Gutenberg byte windows under identical
optimizer settings. GPT gets ~the same param count but its params are spent on
a deep per-token stack (true depth), while Raslila spends them on a 3D grid.
"""
import sys
import os
import time
import argparse
import math
import json

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from raslila.raslila import Raslila, RaslilaConfig
from train_raslila import load_corpus, make_batch


class Block(nn.Module):
    """Standard pre-LN GPT decoder block (causal self-attention + MLP)."""
    def __init__(self, hidden, n_head, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.mlp(x)))
        return x


class GPT(nn.Module):
    """Plain causal decoder (layers stacked per token)."""
    def __init__(self, vocab, max_seq, hidden, n_head, n_layer, ff_dim, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(max_seq, hidden)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(hidden, n_head, ff_dim, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab)
        self.n_layer = n_layer
        self.hidden = hidden

    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        h = self.drop(h)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), 1)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_f(h))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def estimate_flops_per_token(model, seq_len):
    """Approximate forward FLOPs per token (2 FLOPs per multiply-add)."""
    attention = 0
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            attention += 4 * module.embed_dim * seq_len
        elif module.__class__.__name__ == 'CausalSelfAttention':
            attention += 4 * module.hidden_dim * seq_len
    return 2 * count_params(model) + attention


def report_cost(label, model, seq_len):
    active = count_params(model)
    flops = estimate_flops_per_token(model, seq_len)
    print('{} active parameters: {:,}'.format(label, active))
    print('{} approximate forward FLOPs/token: {:,}'.format(label, flops))


def run_validate_wins(args, seeds):
    import yaml
    config_path = args.config if os.path.isabs(args.config) else os.path.join(_ROOT, args.config)
    with open(config_path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    rc = raw.get('raslila', {})
    ds = raw.get('dataset', {})
    valid_fields = set(RaslilaConfig.__dataclass_fields__)
    unknown = sorted(set(rc) - valid_fields)
    if unknown:
        raise SystemExit('Unknown Raslila config fields: {}'.format(unknown))
    if ds.get('name') != 'Skylion007/openwebtext' or ds.get('vocab_size') != 199995 or rc.get('vocab_size') != 199995:
        raise SystemExit('--validate-wins requires the current OpenWebText dataset and matching 199995-token BPE vocabulary.')
    if (rc.get('depth_layers') != 5 or rc.get('grid_h') != 5 or rc.get('grid_w') != 5
            or rc.get('diagonal') is not True or float(rc.get('long_range_prob', -1)) != 0.3
            or int(rc.get('max_long_range', -1)) != 8):
        raise SystemExit('--validate-wins requires the current 5x5x5 layout and wiring: diagonal=true, long_range_prob=0.3, max_long_range=8.')
    cache = ds.get('cache', 'data/openwebtext_tokens.npy')
    cache_path = cache if os.path.isabs(cache) else os.path.join(_ROOT, cache)
    if not os.path.exists(cache_path):
        raise SystemExit('Current token cache not found at {}. Prepare the current dataset on the GPU server first.'.format(cache_path))
    tokenizer = ds.get('tokenizer', 'data/tokenizer/raslila_bpe.json')
    tokenizer_path = tokenizer if os.path.isabs(tokenizer) else os.path.join(_ROOT, tokenizer)
    meta_path = os.path.join(os.path.dirname(tokenizer_path), 'meta.json')
    if not os.path.exists(tokenizer_path) or not os.path.exists(meta_path):
        raise SystemExit('Current fitted BPE tokenizer and meta.json are required; prepare the dataset first.')
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    expected_source = '{}::-::{}chars'.format(ds.get('name'), int(ds.get('tokenizer_sample_chars', 0)))
    if meta.get('source') != expected_source or int(meta.get('vocab_size', 0)) != 199995:
        raise SystemExit('Tokenizer metadata does not match the current OpenWebText 300M-char, 199995-token fit.')
    try:
        from tokenizers import Tokenizer
        actual_vocab = Tokenizer.from_file(tokenizer_path).get_vocab_size()
    except Exception as exc:
        raise SystemExit('Cannot load current BPE tokenizer: {}: {}'.format(type(exc).__name__, exc))
    if actual_vocab != 199995:
        raise SystemExit('Current tokenizer has vocab {}, expected 199995.'.format(actual_vocab))
    if args.batch_size <= 0 or args.seq_len <= 0 or args.lr <= 0:
        raise SystemExit('--batch-size, --seq-len, and --lr must be positive.')
    tok = np.load(cache_path, mmap_mode='r')
    if tok.ndim != 1 or tok.dtype not in (np.dtype('uint16'), np.dtype('uint32'), np.dtype('int32'), np.dtype('int64')):
        raise SystemExit('Current token cache must be a one-dimensional integer array.')
    if len(tok) == 0:
        raise SystemExit('Current token cache is empty.')
    sample_ix = np.linspace(0, len(tok) - 1, num=min(100000, len(tok)), dtype=np.int64)
    if int(tok[sample_ix].max()) >= 199995 or int(tok[sample_ix].min()) < 0:
        raise SystemExit('Current token cache contains ids outside the 199995-token vocabulary.')
    val_count = int(ds.get('val_tokens', 0) or 0)
    if val_count <= 0:
        val_count = min(1000000, max(args.seq_len + 1, len(tok) // 100))
    if val_count < args.seq_len + 1:
        raise SystemExit('dataset.val_tokens must be at least seq_len + 1.')
    if val_count >= len(tok) - args.seq_len - 1:
        raise SystemExit('Token cache is too short for the requested held-out prefix and sequence length.')
    val_tok = tok[:val_count]
    train_tok = tok[val_count:]
    val_starts = np.linspace(0, max(0, len(val_tok) - args.seq_len - 1),
                             num=args.batch_size, dtype=np.int64)
    val_offsets = np.arange(args.seq_len)
    val_indices = val_starts[:, None] + val_offsets[None, :]
    val_inputs = torch.from_numpy(val_tok[val_indices].astype(np.int64)).cuda()
    val_targets = torch.from_numpy(val_tok[val_indices + 1].astype(np.int64)).cuda()
    tokens_per_step = args.batch_size * args.seq_len
    epoch_fraction = args.steps * tokens_per_step / max(1, len(train_tok))
    print('validation steps: {} | tokens/step: {:,} | epoch fraction: {:.6f} | held-out prefix: {:,} tokens'.format(args.steps, tokens_per_step, epoch_fraction, len(val_tok)))
    print('prior figures (-0.27, -0.041, -0.026 nats) are unreplicated on this current setup.')
    effects = {'modern_vs_legacy': [], 'bind4_vs_bind0': [], 'immediate_vs_frozen': []}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        def config(**overrides):
            d = dict(rc)
            d.update(overrides)
            d['max_seq_len'] = args.seq_len
            d['dropout'] = float(d.get('dropout', 0.1))
            return RaslilaConfig(**d)
        modern = config()
        legacy = config(use_rope=False, use_swiglu=False, use_rmsnorm=False,
                        qk_norm=False, tie_weights=False)
        bind0, bind4 = config(bind_roles=0), config(bind_roles=4)
        frozen, immediate = config(immediate_updates=False), config(immediate_updates=True)
        pairs = [('modern_vs_legacy', legacy, modern),
                 ('bind4_vs_bind0', bind0, bind4),
                 ('immediate_vs_frozen', frozen, immediate)]
        for name, cfg_a, cfg_b in pairs:
            torch.manual_seed(seed)
            a = Raslila(cfg_a).cuda()
            torch.manual_seed(seed)
            b = Raslila(cfg_b).cuda()
            oa = torch.optim.AdamW(a.parameters(), lr=args.lr)
            ob = torch.optim.AdamW(b.parameters(), lr=args.lr)
            for step in range(args.steps):
                inp, tgt = make_batch(train_tok, args.batch_size, args.seq_len, step, args.steps)
                inp, tgt = inp.cuda(non_blocking=True), tgt.cuda(non_blocking=True)
                for arm_index, (model, opt) in enumerate(((a, oa), (b, ob))):
                    torch.manual_seed(seed + step + 100000)
                    opt.zero_grad(set_to_none=True)
                    logits = model(inp)['logits']
                    loss = F.cross_entropy(logits.reshape(-1, 199995), tgt.reshape(-1))
                    loss.backward()
                    opt.step()
            a.eval(); b.eval()
            with torch.no_grad():
                la = F.cross_entropy(a(val_inputs)['logits'].reshape(-1, 199995), val_targets.reshape(-1)).item()
                lb = F.cross_entropy(b(val_inputs)['logits'].reshape(-1, 199995), val_targets.reshape(-1)).item()
            effects[name].append(lb - la)
            del a, b, oa, ob
            torch.cuda.empty_cache()
        print('seed {} complete'.format(seed), flush=True)
    for name, values in effects.items():
        mean = sum(values) / len(values)
        spread = math.sqrt(sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1))
        print('{}: mean paired effect {:+.4f} nats, seed SD {:.4f}, seeds {}'.format(
            name, mean, spread, ','.join(map(str, seeds))))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--seq-len', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--data-dir', type=str, default='data/gutenberg')
    ap.add_argument('--gpt-vocab', type=int, default=257)
    ap.add_argument('--gpt-hidden', type=int, default=256)
    ap.add_argument('--gpt-layers', type=int, default=8)
    ap.add_argument('--node-depth', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--validate-wins', action='store_true',
                    help='Measure modern vs legacy blocks, bind_roles=4 vs 0, and immediate updates vs off on current data. Prior figures are unreplicated on the current setup. Requires GPU, --steps, --seeds, and --config.')
    ap.add_argument('--seeds', type=str, default=None,
                    help='Comma-separated paired seeds required with --validate-wins.')
    ap.add_argument('--config', type=str, default='config.yaml',
                    help='Current Raslila/dataset config for --validate-wins.')
    ap.add_argument('--mode', type=str, default='gpt_vs_raslila',
                    choices=['gpt_vs_raslila', 'legacy_vs_modern', 'plus', 'cortex', 'recur', 'bind'],
                    help='gpt_vs_raslila: GPT stack vs modern Raslila; '
                         'legacy_vs_modern: old-style vs modernized; '
                         'plus: LNS/gated-message ablation; '
                         'cortex: MTP/workspace/topk ablation; '
                         'recur: frozen vs immediate cascades vs priority; '
                         'bind: baseline vs role-binding vs binding+local')
    args = ap.parse_args()

    if args.validate_wins:
        if args.steps is None or args.steps <= 0 or not args.seeds:
            ap.error('--validate-wins requires --steps > 0 and --seeds (comma-separated).')
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except ValueError:
            ap.error('--seeds must be comma-separated integers.')
        if len(seeds) < 5 or len(set(seeds)) != len(seeds):
            ap.error('--validate-wins requires at least five unique paired seeds.')
        if not torch.cuda.is_available():
            raise SystemExit('--validate-wins requires a CUDA GPU; refusing to run on this CPU-only host.')
        return run_validate_wins(args, seeds)

    if args.steps is None:
        args.steps = 300
    torch.manual_seed(args.seed)
    tok = load_corpus(args.data_dir)
    print('corpus tokens:', len(tok))
    print('-' * 70)

    if args.mode == 'bind':
        def bind_cfg(**kw):
            d = dict(dim_x=3, dim_y=3, dim_z=2, hidden_dim=256, num_heads=8,
                     num_layers=2, node_depth=args.node_depth, ff_dim=1024,
                     edge_hidden_dim=128, edge_layers=2, local_radius=1,
                     long_range_prob=0.1, max_long_range=3, vocab_size=257,
                     max_seq_len=args.seq_len, causal=True,
                     immediate_updates=True)
            d.update(kw)
            return RaslilaConfig(**d)
        arms = [
            ('base       ', bind_cfg()),
            ('bind4      ', bind_cfg(bind_roles=4)),
            ('bind8      ', bind_cfg(bind_roles=8)),
            ('bind4+local', bind_cfg(bind_roles=4, bind_local=True)),
        ]
        models = [(n, Raslila(c)) for n, c in arms]
        for n, m in models:
            print('{} params: {:>10,}'.format(n.strip(), count_params(m)))
            report_cost(n.strip(), m, args.seq_len)
        print('-' * 70)
        opts = [torch.optim.AdamW(m.parameters(), lr=args.lr) for _, m in models]
        for _, m in models:
            m.train()
        t0 = time.time()
        losses = {}
        for step in range(args.steps):
            inp, tgt = make_batch(tok, args.batch_size, args.seq_len, step, args.steps)
            for (n, m), o in zip(models, opts):
                r = m.full_train_step(inp, tgt, optimizer=o, lr=args.lr)
                losses[n] = r['main_loss']
            if step % 50 == 0 or step == args.steps - 1:
                el = time.time() - t0
                print('step {:>4} | {} | {:.0f}s'.format(
                    step, ' | '.join('{} {:7.4f}'.format(n.strip(), losses[n])
                                     for n, _ in models), el))
                sys.stdout.flush()
        print('-' * 70)
        for n, _ in models:
            print('{} main-loss final: {:.4f}'.format(n.strip(), losses[n]))
        return

    if args.mode == 'recur':
        def recur_cfg(**kw):
            d = dict(dim_x=3, dim_y=3, dim_z=2, hidden_dim=256, num_heads=8,
                     num_layers=2, node_depth=args.node_depth, ff_dim=1024,
                     edge_hidden_dim=128, edge_layers=2, local_radius=1,
                     long_range_prob=0.1, max_long_range=3, vocab_size=257,
                     max_seq_len=args.seq_len, causal=True)
            d.update(kw)
            return RaslilaConfig(**d)
        arms = [
            ('frozen2   ', recur_cfg()),
            ('immediate2', recur_cfg(immediate_updates=True)),
            ('+priority ', recur_cfg(immediate_updates=True, priority_order=True)),
            ('frozen4   ', recur_cfg(num_layers=4)),
        ]
        models = [(n, Raslila(c)) for n, c in arms]
        for n, m in models:
            print('{} params: {:>10,}'.format(n.strip(), count_params(m)))
            report_cost(n.strip(), m, args.seq_len)
        print('-' * 70)
        opts = [torch.optim.AdamW(m.parameters(), lr=args.lr) for _, m in models]
        for _, m in models:
            m.train()
        t0 = time.time()
        losses = {}
        for step in range(args.steps):
            inp, tgt = make_batch(tok, args.batch_size, args.seq_len, step, args.steps)
            for (n, m), o in zip(models, opts):
                r = m.full_train_step(inp, tgt, optimizer=o, lr=args.lr)
                losses[n] = r['main_loss']
            if step % 50 == 0 or step == args.steps - 1:
                el = time.time() - t0
                print('step {:>4} | {} | {:.0f}s'.format(
                    step, ' | '.join('{} {:7.4f}'.format(n.strip(), losses[n])
                                     for n, _ in models), el))
                sys.stdout.flush()
        print('-' * 70)
        for n, _ in models:
            print('{} main-loss final: {:.4f}'.format(n.strip(), losses[n]))
        return

    if args.mode == 'cortex':
        def cortex_cfg(**kw):
            d = dict(dim_x=3, dim_y=3, dim_z=2, hidden_dim=256, num_heads=8,
                     num_layers=2, node_depth=args.node_depth, ff_dim=1024,
                     edge_hidden_dim=128, edge_layers=2, local_radius=1,
                     long_range_prob=0.1, max_long_range=3, vocab_size=257,
                     max_seq_len=args.seq_len, causal=True)
            d.update(kw)
            return RaslilaConfig(**d)
        arms = [
            ('base      ', cortex_cfg()),
            ('+mtp      ', cortex_cfg(mtp_steps=2, mtp_weight=0.3)),
            ('+workspace', cortex_cfg(workspace_slots=4)),
            ('+topk     ', cortex_cfg(readout_topk=4)),
        ]
        models = [(n, Raslila(c)) for n, c in arms]
        for n, m in models:
            print('{} params: {:>10,}'.format(n.strip(), count_params(m)))
            report_cost(n.strip(), m, args.seq_len)
        print('-' * 70)
        # full_train_step applies each arm's own config (MTP auxiliary loss etc.)
        opts = [torch.optim.AdamW(m.parameters(), lr=args.lr) for _, m in models]
        for _, m in models:
            m.train()
        t0 = time.time()
        losses = {}
        for step in range(args.steps):
            inp, tgt = make_batch(tok, args.batch_size, args.seq_len, step, args.steps)
            for (n, m), o in zip(models, opts):
                r = m.full_train_step(inp, tgt, optimizer=o, lr=args.lr)
                losses[n] = r['main_loss']
            if step % 50 == 0 or step == args.steps - 1:
                el = time.time() - t0
                print('step {:>4} | {} | {:.0f}s'.format(
                    step, ' | '.join('{} {:7.4f}'.format(n.strip(), losses[n])
                                     for n, _ in models), el))
                sys.stdout.flush()
        print('-' * 70)
        for n, _ in models:
            print('{} main-loss final: {:.4f}'.format(n.strip(), losses[n]))
        return

    if args.mode == 'plus':
        def plus_cfg(**kw):
            d = dict(dim_x=3, dim_y=3, dim_z=2, hidden_dim=256, num_heads=8,
                     num_layers=2, node_depth=args.node_depth, ff_dim=1024,
                     edge_hidden_dim=128, edge_layers=2, local_radius=1,
                     long_range_prob=0.1, max_long_range=3, vocab_size=257,
                     max_seq_len=args.seq_len, causal=True)
            d.update(kw)
            return RaslilaConfig(**d)
        arms = [
            ('base      ', plus_cfg()),
            ('+lns      ', plus_cfg(lns=True)),
            ('+gate     ', plus_cfg(gated_messages=True)),
            ('+both     ', plus_cfg(lns=True, gated_messages=True)),
        ]
        models = [(n, Raslila(c)) for n, c in arms]
        for n, m in models:
            print('{} params: {:>10,}'.format(n.strip(), count_params(m)))
            report_cost(n.strip(), m, args.seq_len)
        print('-' * 70)
        opts = [torch.optim.AdamW(m.parameters(), lr=args.lr) for _, m in models]
        for _, m in models:
            m.train()
        t0 = time.time()
        losses = {}
        for step in range(args.steps):
            inp, tgt = make_batch(tok, args.batch_size, args.seq_len, step, args.steps)
            for (n, m), o in zip(models, opts):
                o.zero_grad(set_to_none=True)
                l = m(inp)['logits']
                loss = F.cross_entropy(l.view(-1, 257), tgt.view(-1))
                loss.backward(); o.step()
                losses[n] = loss.item()
            if step % 50 == 0 or step == args.steps - 1:
                el = time.time() - t0
                print('step {:>4} | {} | {:.0f}s'.format(
                    step, ' | '.join('{} {:7.4f}'.format(n.strip(), losses[n])
                                     for n, _ in models), el))
                sys.stdout.flush()
        print('-' * 70)
        for n, _ in models:
            print('{} final: {:.4f}'.format(n.strip(), losses[n]))
        return

    if args.mode == 'legacy_vs_modern':
        # NOTE: each mode's baseline reflects the config that was current when
        # that ablation was run (e.g. 'bind' includes immediate_updates=True,
        # this one predates it). Compare arms WITHIN a mode only; cross-mode
        # numbers are not directly comparable.
        cfg_a = RaslilaConfig(
            dim_x=3, dim_y=3, dim_z=2, hidden_dim=256, num_heads=8, num_layers=2,
            node_depth=args.node_depth, ff_dim=1024, edge_hidden_dim=128, edge_layers=2,
            local_radius=1, long_range_prob=0.1, max_long_range=3, vocab_size=257,
            max_seq_len=args.seq_len, causal=True,
            use_rope=False, use_swiglu=False, use_rmsnorm=False,
            qk_norm=False, tie_weights=False,
        )
        cfg_b = RaslilaConfig(
            dim_x=3, dim_y=3, dim_z=2, hidden_dim=256, num_heads=8, num_layers=2,
            node_depth=args.node_depth, ff_dim=1024, edge_hidden_dim=128, edge_layers=2,
            local_radius=1, long_range_prob=0.1, max_long_range=3, vocab_size=257,
            max_seq_len=args.seq_len, causal=True,
        )
        model_a = Raslila(cfg_a)
        model_b = Raslila(cfg_b)
        print('Legacy Raslila params: {:>10,}'.format(count_params(model_a)))
        print('Modern Raslila params: {:>10,}'.format(count_params(model_b)))
        print('-' * 70)

        opt_a = torch.optim.AdamW(model_a.parameters(), lr=args.lr)
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=args.lr)
        model_a.train(); model_b.train()
        t0 = time.time()
        for step in range(args.steps):
            inp, tgt = make_batch(tok, args.batch_size, args.seq_len, step, args.steps)
            opt_a.zero_grad(set_to_none=True)
            la = model_a(inp)['logits']
            loss_a = F.cross_entropy(la.view(-1, 257), tgt.view(-1))
            loss_a.backward(); opt_a.step()

            opt_b.zero_grad(set_to_none=True)
            lb = model_b(inp)['logits']
            loss_b = F.cross_entropy(lb.view(-1, 257), tgt.view(-1))
            loss_b.backward(); opt_b.step()

            if step % 50 == 0 or step == args.steps - 1:
                el = time.time() - t0
                print('step {:>4} | Legacy {:7.4f} | Modern {:7.4f} | {:.0f}s'.format(
                    step, loss_a.item(), loss_b.item(), el))
                sys.stdout.flush()

        print('-' * 70)
        print('Legacy final: {:.4f}'.format(loss_a.item()))
        print('Modern final: {:.4f}'.format(loss_b.item()))
        print('Modern - Legacy: {:.3f} nats (negative = modern BETTER)'.format(
            loss_b.item() - loss_a.item()))
        return

    gpt = GPT(args.gpt_vocab, args.seq_len, args.gpt_hidden, 4, args.gpt_layers,
              args.gpt_hidden * 4)
    cfg = RaslilaConfig(
        dim_x=3, dim_y=3, dim_z=2, hidden_dim=256, num_heads=8, num_layers=2,
        node_depth=args.node_depth, ff_dim=1024, edge_hidden_dim=128, edge_layers=2,
        local_radius=1, long_range_prob=0.1, max_long_range=3, vocab_size=257,
        max_seq_len=args.seq_len, causal=True,
    )
    raslila = Raslila(cfg)
    print('GPT params:   {:>10,}'.format(count_params(gpt)))
    print('Raslila params: {:>10,}'.format(count_params(raslila)))
    report_cost('GPT', gpt, args.seq_len)
    report_cost('Raslila', raslila, args.seq_len)
    print('-' * 70)

    opt_g = torch.optim.AdamW(gpt.parameters(), lr=args.lr)
    opt_b = torch.optim.AdamW(raslila.parameters(), lr=args.lr)

    gpt.train(); raslila.train()
    t0 = time.time()
    for step in range(args.steps):
        inp, tgt = make_batch(tok, args.batch_size, args.seq_len, step, args.steps)
        opt_g.zero_grad(set_to_none=True)
        lg = gpt(inp)
        loss_g = F.cross_entropy(lg.view(-1, args.gpt_vocab), tgt.view(-1))
        loss_g.backward(); opt_g.step()

        opt_b.zero_grad(set_to_none=True)
        lb = raslila(inp)['logits']
        loss_b = F.cross_entropy(lb.view(-1, args.gpt_vocab), tgt.view(-1))
        loss_b.backward(); opt_b.step()

        if step % 50 == 0 or step == args.steps - 1:
            el = time.time() - t0
            print('step {:>4} | GPT loss {:7.4f} | Raslila loss {:7.4f} | {:.0f}s'.format(
                step, loss_g.item(), loss_b.item(), el))
            sys.stdout.flush()

    print('-' * 70)
    print('GPT final:     {:.4f}'.format(loss_g.item()))
    print('Raslila final:   {:.4f}'.format(loss_b.item()))
    print('Raslila - GPT: {:.3f} nats (negative = raslila BETTER)'.format(
        loss_b.item() - loss_g.item()))


if __name__ == '__main__':
    main()