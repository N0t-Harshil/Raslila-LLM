"""Download + tokenize a HuggingFace corpus for Raslila.

Standalone front-end to src/raslila/hf_dataset.py. Training calls the same code
automatically when the cache is missing, so you only need this to pre-fetch or
to force a rebuild.

Usage:
    python scripts/get_dataset.py --config config.yaml
    python scripts/get_dataset.py --config config_50m.yaml --force
    python scripts/get_dataset.py --max-tokens 50000000     # quick smoke fetch
    python scripts/get_dataset.py --list                    # show the defaults
"""
import argparse
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import yaml

from raslila.hf_dataset import DEFAULTS, ensure_dataset, resolve


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config.yaml',
                    help='config whose `dataset:` section to use')
    ap.add_argument('--name', default=None, help='override dataset.name')
    ap.add_argument('--split', default=None, help='override dataset.split')
    ap.add_argument('--max-tokens', type=int, default=None,
                    help='override dataset.max_tokens (lower = faster test fetch)')
    ap.add_argument('--cache', default=None, help='override dataset.cache')
    ap.add_argument('--force', action='store_true',
                    help='rebuild even if the cache already exists')
    ap.add_argument('--list', action='store_true',
                    help='print the effective dataset settings and exit')
    args = ap.parse_args()

    cfg = None
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        cfg = cfg.get('dataset')
    elif args.config != 'config.yaml':
        raise SystemExit('Config not found: {}'.format(args.config))

    overrides = {k: v for k, v in {
        'name': args.name, 'split': args.split,
        'max_tokens': args.max_tokens, 'cache': args.cache,
    }.items() if v is not None}

    ds = resolve(cfg)
    ds.update(overrides)

    if args.list:
        print('effective dataset settings ({})'.format(args.config))
        for key in sorted(set(DEFAULTS) | set(ds)):
            print('  {:<24} {}'.format(key, ds.get(key)))
        return

    ensure_dataset(ds, force=args.force)


if __name__ == '__main__':
    main()
