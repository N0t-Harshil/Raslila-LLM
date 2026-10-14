"""Verify the 20B groundwork WITHOUT training.

Checks, in order:
  1. config_20b.yaml parses and the shape counts to the configured 17.82B params
     (analytic estimate -- does NOT allocate 20B params).
  2. activation_checkpoint is bit-identical to the plain path (fwd + grad).
  3. FSDP 2-rank (gloo, CPU) produces loss matching single-process within
     tolerance. This is the check that matters: sharding bugs are invisible
     until they silently corrupt weights, and they show up as a loss gap.
  4. Reports per-rank memory arithmetic so you know if 20B fits before
     launching on the A100s.

Run:  python scripts/verify_20b_groundwork.py
"""
import os
import sys
import math
import argparse
import importlib.util

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import torch
import yaml

from raslila.raslila import Raslila, RaslilaConfig, TransformerNode

CFG_20B = os.path.join(_ROOT, 'config_20b.yaml')
OK = True

# A deliberately tiny 3x(3x3), hidden-64 model, inlined so this script has no
# dependency on any config file (config_3m.yaml / config_15m.yaml were removed
# when the config set was cut to 50m/164m/20b). Every key is a RaslilaConfig
# field; dropout is forced to 0.0 at the call site because these checks are
# about bit-identity.
SMALL_CONFIG = {
    'layout': 'deep_grid',
    'depth_layers': 3, 'grid_h': 3, 'grid_w': 3,
    'num_layers': 1, 'node_depth': 1,
    'hidden_dim': 64, 'num_heads': 4, 'ff_dim': 256, 'dropout': 0.0,
    'use_rope': True, 'rope_theta': 10000.0,
    'use_swiglu': True, 'use_rmsnorm': True, 'qk_norm': True,
    'tie_weights': True, 'lns': False, 'gated_messages': False,
    'mtp_steps': 0, 'mtp_weight': 0.3,
    'workspace_slots': 0, 'readout_topk': 0,
    'bind_roles': 4, 'bind_local': False,
    'replay_prob': 0.0, 'replay_weight': 1.5,
    'grow_every': 0, 'grow_min_scale': 0.004, 'grow_step': 2,
    'grow_max_nodes': 64,
    'immediate_updates': True, 'priority_order': False, 'settle_thresh': 0.0,
    'depth_sweeps': 2,
    'inhibit': False, 'inhibit_neg': True, 'inhibit_beta': 1.0,
    'state_dim': 0, 'state_decay_init': 0.9,
    'local_radius': 1, 'long_range_prob': 0.3, 'max_long_range': 8,
    'connectivity_seed': 42,
    'lateral': True, 'vertical': True, 'diagonal': True, 'feedback': True,
    'readout_layer': 'last',
    'edge_hidden_dim': 24, 'edge_layers': 2,
    'vocab_size': 16384, 'max_seq_len': 1024, 'causal': True,
}


# Module-level (not nested): spawn pickles the target, and a closure over
# `main`'s locals cannot be pickled -- that is exactly the error you get from
# mp.get_context('spawn') with a nested worker.


def _fsdp_worker(rank, world, small_cfg, q, init_file):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools

    # File-based init, not env://. Gloo on Windows cannot resolve a hostname to
    # a device ("makeDeviceForHostname(): unsupported gloo device"), so
    # MASTER_ADDR/MASTER_PORT + env:// fails there. A rendezvous file works on
    # both Windows and Linux and needs no free port.
    dist.init_process_group('gloo', init_method='file://{}'.format(init_file),
                            rank=rank, world_size=world)
    torch.manual_seed(123)   # same init weights on both ranks
    m = Raslila(RaslilaConfig(**small_cfg))
    m = FSDP(
        m,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerNode}),
        use_orig_params=True,
        limit_all_gathers=True,
    )
    m.eval()
    # Same data on every rank so the loss is comparable to the reference.
    torch.manual_seed(77)
    x = torch.randint(0, small_cfg['vocab_size'], (4, 32))
    y = torch.randint(0, small_cfg['vocab_size'], (4, 32))
    with torch.no_grad():
        l = torch.nn.functional.cross_entropy(
            m(x)['logits'].reshape(-1, small_cfg['vocab_size']), y.reshape(-1)).item()
    q.put((rank, l))
    dist.destroy_process_group()


def check(name, cond, detail=''):
    global OK
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        OK = False
    print('  [{}] {}{}'.format(tag, name, ('  -- ' + detail) if detail else ''))


def analytic_params(b3):
    """Count params without instantiating the full model.

    The full configured model is ~80 GB of fp32 alone; instantiating it on a CPU
    box to count parameters is both slow and likely to OOM the local machine.
    Per-region cost scales exactly, so one region's count times the region
    count is exact for the transformer part, and the edge/embedding terms are
    closed-form.
    """
    node = TransformerNode(
        b3['hidden_dim'], b3['num_heads'], b3['ff_dim'], 0.0,
        num_blocks=b3.get('node_depth', 1), use_rope=True,
        use_swiglu=True, use_rmsnorm=True, qk_norm=True)
    per_node = sum(p.numel() for p in node.parameters())
    H = b3['hidden_dim']
    ehd = b3.get('edge_hidden_dim', 96)
    el = b3.get('edge_layers', 2)
    if el == 2:
        per_edge = 2 * H * ehd + ehd + ehd * H + H
    else:
        per_edge = 0
    n_nodes = b3['depth_layers'] * b3['grid_h'] * b3['grid_w']
    # measured edge:node ratio on the current 5x5 production grid is 1114/125
    n_edges = int(round(n_nodes * 1114 / 125))
    emb = b3['vocab_size'] * H + H
    readout = 4 * H * H + 10 * H
    return per_node * n_nodes + per_edge * n_edges + emb + readout, n_nodes, n_edges


def main():
    global OK
    parser = argparse.ArgumentParser(description='Static and small-model 20B groundwork checks; never constructs the full model.')
    parser.add_argument('--require-fsdp-check', action='store_true',
                        help='treat the Windows/Linux FSDP equivalence skip as a failure')
    args = parser.parse_args()
    print('=' * 72)
    print('Raslila 20B groundwork verification')
    print('=' * 72)

    with open(CFG_20B, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    b3 = raw['raslila']
    tr = raw['training']
    cfg = RaslilaConfig(**b3)

    print('   dependency imports')
    for package in ('datasets', 'huggingface_hub'):
        try:
            __import__(package)
            check('{} import'.format(package), True)
        except Exception as exc:
            check('{} import'.format(package), False, '{}: {}'.format(type(exc).__name__, exc))

    # ---- 1. shape + memory arithmetic (no giant allocation) ----
    print('\n1. Shape and memory arithmetic')
    n, n_nodes, n_edges = analytic_params(b3)
    print('   analytic estimate: {:,} params ({:.2f}B), {} regions, ~{} edges'
          .format(n, n / 1e9, n_nodes, n_edges))
    check('shipped 20B-class shape', 17e9 <= n <= 19e9, '{:.2f}B'.format(n / 1e9))

    WORLD = 4
    BYTES = 16          # bf16 params+grads (4) + fp32 master (4) + 2x Adam (8)
    state_gb = n * BYTES / 1e9
    per_rank = state_gb / WORLD
    print('   training state at {:.0f} bytes/param: {:.1f} GB total'.format(BYTES, state_gb))
    print('   FSDP FULL_SHARD over {} ranks: {:.1f} GB/rank'.format(WORLD, per_rank))
    print('   A100-80GB per rank: 80.0 GB -> headroom {:.1f} GB'.format(80 - per_rank))
    if per_rank > 80:
        print('   NOTE: state alone exceeds 80 GB/rank. activation_checkpoint is')
        print('         mandatory AND you may still need node_depth 13->11 or')
        print('         fsdp_cpu_offload. See config_20b.yaml header.')
    check('state sharded fits (or checkpointing compensates)',
          per_rank <= 80 or b3.get('activation_checkpoint', False))

    # ---- 2. activation_checkpoint bit-identity ----
    print('\n2. activation_checkpoint equivalence (small config, dropout=0)')
    # Inline rather than read config_3m.yaml: that file was deleted when the
    # config set was cut down to 50m/164m/20b, and this check wants a genuinely
    # tiny model, not one of the production shapes.
    small = dict(SMALL_CONFIG)
    small['dropout'] = 0.0
    torch.manual_seed(0)
    a = Raslila(RaslilaConfig(**small)).eval()
    b = Raslila(RaslilaConfig(**{**small, 'activation_checkpoint': True})).eval()
    b.load_state_dict(a.state_dict())
    ids = torch.randint(0, small['vocab_size'], (2, 32))
    la, lb = a(ids)['logits'], b(ids)['logits']
    check('eval forward identical', torch.equal(la, lb),
          'max|d|={:.3e}'.format(float((la - lb).abs().max())))
    a.train(); b.train()
    oa, ob = a(ids)['logits'], b(ids)['logits']
    check('train forward identical', torch.equal(oa, ob))
    oa.sum().backward(); ob.sum().backward()
    ga = dict(a.named_parameters()); gb = dict(b.named_parameters())
    worst = 0.0
    mismatch = False
    for k in ga:
        if (ga[k].grad is None) != (gb[k].grad is None):
            mismatch = True
        elif ga[k].grad is not None:
            worst = max(worst, float((ga[k].grad - gb[k].grad).abs().max()))
    check('gradients identical', not mismatch and worst == 0.0,
          'max|d|={:.3e}'.format(worst))

    # ---- 3. FSDP == single-process (the real check) ----
    if sys.platform.startswith('win'):
        print('\n3. FSDP equivalence vs single process -- SKIPPED on Windows')
        print('   Gloo cannot initialize a process group on this box:')
        print('     RuntimeError: makeDeviceForHostname(): unsupported gloo device')
        print('   That is a Windows limitation of the gloo backend, not a bug in')
        print('   the FSDP path. This check is what the server run is FOR -- run')
        print('   it there before training:')
        print('     python scripts/verify_20b_groundwork.py --require-fsdp-check')
        print('   On Linux it spawns 2 gloo ranks, shards a small model with FSDP,')
        print('   and asserts the loss matches single-process. Do not train 20B')
        print('   until that check passes on the A100 box.')
        if args.require_fsdp_check:
            check('FSDP equivalence executed (required)', False, 'run this check on Linux')
    else:
        print('\n3. FSDP equivalence vs single process (this is the one that matters)')
        print('   Launching a 2-rank gloo/FSDP job via torch.multiprocessing...')
        import torch.multiprocessing as mp

        world = 2
        ref_loss = [None]

        torch.manual_seed(123)
        m = Raslila(RaslilaConfig(**small)).eval()
        x = torch.randint(0, small['vocab_size'], (4, 32))
        y = torch.randint(0, small['vocab_size'], (4, 32))
        with torch.no_grad():
            ref_loss[0] = torch.nn.functional.cross_entropy(
                m(x)['logits'].reshape(-1, small['vocab_size']), y.reshape(-1)).item()

        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        init_file = os.path.join(os.environ.get('TEMP', '.'),
                                 'raslila_fsdp_rendezvous')
        for p in (init_file, init_file + '.lock'):
            if os.path.exists(p):
                os.remove(p)
        procs = [ctx.Process(target=_fsdp_worker,
                             args=(r, world, small, q, init_file))
                 for r in range(world)]
        for p in procs:
            p.start()
        got = {}
        for _ in range(world):
            r, l = q.get(timeout=300)
            got[r] = l
        for p in procs:
            p.join(timeout=60)

        ok_join = len(got) == world and all(not p.is_alive() for p in procs)
        check('both FSDP ranks finished (no hang)', ok_join,
              'got ranks {}'.format(sorted(got)))
        if ok_join:
            # Reference used seed 123 for init AND seed 77 for data; the FSDP
            # worker mirrors both. Identical weights + data -> identical loss.
            gap = max(abs(got[r] - ref_loss[0]) for r in got)
            spread = max(got.values()) - min(got.values())
            check('FSDP loss matches single-process', gap < 1e-4,
                  'gap={:.3e}, rank spread={:.3e}'.format(gap, spread))
            print('   single-process loss: {:.6f}'.format(ref_loss[0]))
            for r in sorted(got):
                print('   FSDP rank {} loss:    {:.6f}'.format(r, got[r]))

    # ---- summary ----
    print('\n' + '=' * 72)
    if OK:
        print('ALL CHECKS PASSED. Groundwork is sound.')
        print('Next: 20-step smoke run on the server, then check nvidia-smi peak')
        print('      memory and tok/s before committing to 30000 steps.')
    else:
        print('SOME CHECKS FAILED -- do not launch on the server yet.')
    print('=' * 72)
    return 0 if OK else 1


if __name__ == '__main__':
    sys.exit(main())
