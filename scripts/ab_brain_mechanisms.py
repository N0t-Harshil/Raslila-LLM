"""A/B the brain mechanisms added for bidirectional thinking + learning from experience.

Each variant changes ONE thing against a fixed baseline, so the effect is
attributable. Same seed, same data, same steps, byte vocab for a controlled
comparison.
"""
import os
import sys
import time

sys.path.insert(0, 'src')
import numpy as np
import torch
import torch.nn.functional as F

from raslila.raslila import Raslila, RaslilaConfig

BASE = dict(
    depth_layers=3, grid_h=3, grid_w=3,
    hidden_dim=64, num_heads=4, ff_dim=128, edge_hidden_dim=32,
    vocab_size=257, bind_roles=4, dropout=0.0,
    causal=True, use_rope=True, use_swiglu=True, use_rmsnorm=True, qk_norm=True,
    immediate_updates=True, long_range_prob=0.1,
    # baseline = plain feedforward, additive, no memory
    depth_sweeps=1, inhibit=False, state_dim=0,
)

STEPS = int(os.environ.get('AB_STEPS', '120'))
BATCH = 4
SEQ = 48


"""A/B the brain mechanisms added for bidirectional thinking + learning from experience.

Each variant changes ONE thing against a fixed baseline, so the effect is
attributable. Same seed, same data, same steps, same BPE tokens as production.

Two bugs this benchmark has already caught, both of which would have made the
comparison meaningless:

1. It originally fed uniformly RANDOM bytes. Random bytes have no structure, so
   the loss pins at ln(257) and every variant ties. A tie that comes from the
   data having no structure is not a result about the model.
2. It then assumed a 257-vocab byte corpus, but the real cached corpus is BPE-200k
   (ids up to 199994). Training a 257-vocab model on 200k-vocab data is exactly the
   silent data/vocab mismatch that has bitten this project before.
"""
import os
import sys
import time

sys.path.insert(0, 'src')
import numpy as np
import torch
import torch.nn.functional as F

from raslila.raslila import Raslila, RaslilaConfig

VOCAB = 199995

BASE = dict(
    depth_layers=3, grid_h=3, grid_w=3,
    hidden_dim=64, num_heads=4, ff_dim=128, edge_hidden_dim=32,
    vocab_size=VOCAB, bind_roles=4, dropout=0.0,
    causal=True, use_rope=True, use_swiglu=True, use_rmsnorm=True, qk_norm=True,
    immediate_updates=True, long_range_prob=0.1,
    # baseline = plain feedforward, additive, no memory
    depth_sweeps=1, inhibit=False, state_dim=0,
)


def corpus(n=1_500_000):
    """Real English text as BPE ids (production tokenization)."""
    # data/corpus_tokens.npy was deleted with the 2026-10 corpus swap;
    # data/openwebtext_tokens.npy is the production cache now.
    for path in ('data/openwebtext_tokens.npy', 'data/corpus_tokens.npy'):
        if os.path.exists(path):
            break
    else:
        raise SystemExit(
            'no token cache found. Build one first:\n'
            '  python scripts/get_dataset.py --config config.yaml')
    ids = np.load(path)
    hi = int(ids.max())
    if hi >= VOCAB:
        raise SystemExit('corpus has id {} but vocab is {}: mismatch'.format(hi, VOCAB))
    return np.asarray(ids[:n], dtype=np.int64)


def run(variant, data, seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = dict(BASE)
    cfg.update(variant)
    model = Raslila(RaslilaConfig(**cfg))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
    warmup = 15
    losses = []
    t0 = time.perf_counter()
    pos = 0
    for step in range(STEPS):
        if cfg['state_dim'] > 0 and step > 0:
            model.reset_state()
        if pos + (BATCH + 1) * SEQ > len(data):
            pos = 0
        rows, nxt = [], []
        for b in range(BATCH):
            i = pos + b * SEQ
            rows.append(torch.tensor(data[i:i + SEQ], dtype=torch.long))
            nxt.append(torch.tensor(data[i + 1:i + 1 + SEQ], dtype=torch.long))
        pos += BATCH * SEQ
        x = torch.stack(rows)
        t = torch.stack(nxt)
        model.train()
        lr = 6e-4 * min(1.0, (step + 1) / warmup)
        for g in opt.param_groups:
            g['lr'] = lr
        logits = model(x)['logits']
        loss = F.cross_entropy(logits.view(-1, VOCAB), t.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    dt = (time.perf_counter() - t0) / STEPS
    # Final loss = mean of last 15 steps (less noisy than the last single step)
    tail = float(np.mean(losses[-15:]))
    return tail, n_params, dt, losses


def main():
    torch.set_num_threads(8)
    data = corpus()
    print('A/B: {} steps, batch {} x seq {}, BPE vocab {}'.format(
        STEPS, BATCH, SEQ, VOCAB))
    print('seed fixed, one variable changed at a time\n')
    print('{:<34s} {:>11s} {:>12s} {:>8s} {:>9s}'.format(
        'variant', 'params', 'final loss', 'vs base', 's/step'))
    print('-' * 80)

    variants = [
        ('baseline (feedforward, additive)', {}),
        ('+ feedback sweep (depth_sweeps=2)', {'depth_sweeps': 2}),
        ('+ feedback sweep x3 (sweeps=3)', {'depth_sweeps': 3}),
        ('+ inhibition (signed edges)', {'inhibit': True}),
        ('+ persistent state (dim 16)', {'state_dim': 16}),
        ('+ state 32 & sweeps 2 & inhibit', {'state_dim': 32, 'depth_sweeps': 2,
                                             'inhibit': True}),
    ]
    base_loss = None
    for name, kw in variants:
        loss, npar, dt, curve = run(kw, data)
        if base_loss is None:
            base_loss = loss
        delta = '' if kw == {} else '{:+.4f}'.format(loss - base_loss)
        print('{:<34s} {:>11,} {:>12.4f} {:>8s} {:>9.2f}'.format(
            name, npar, loss, delta, dt))
    print()
    print('Lower final loss is better. "vs base" is the absolute nats difference;')
    print('negative means that change HELPED.')


if __name__ == '__main__':
    main()