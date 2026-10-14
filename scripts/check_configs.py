"""Validate active Raslila configs against the model and trainer surface."""
import ast
import dataclasses
import os
import re
import sys

import yaml

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from raslila.raslila import RaslilaConfig

CONFIGS = ['config.yaml', 'config_baseline_feedforward.yaml', 'config_50m.yaml', 'config_20b.yaml']
TRAINER = os.path.join(ROOT, 'scripts', 'train_raslila.py')


def load(name):
    with open(os.path.join(ROOT, name), encoding='utf-8') as f:
        return yaml.safe_load(f)


def training_reads(tree):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                found.add(node.args[0].value)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'tr':
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                found.add(key.value)
    return found


def main():
    failures = []
    field_names = {f.name for f in dataclasses.fields(RaslilaConfig)}
    with open(TRAINER, encoding='utf-8') as f:
        trainer_text = f.read()
    read_keys = training_reads(ast.parse(trainer_text))
    for name in CONFIGS:
        raw = load(name)
        for key in raw.get('raslila', {}):
            if key not in field_names:
                failures.append('{}: unknown raslila key {}'.format(name, key))
        for key in raw.get('training', {}):
            if key not in read_keys:
                failures.append('{}: unread training key {}'.format(name, key))
        if raw['raslila'].get('vocab_size') != raw.get('dataset', {}).get('vocab_size'):
            failures.append('{}: raslila.vocab_size != dataset.vocab_size'.format(name))
        r = raw['raslila']
        if name == 'config_20b.yaml' and r.get('node_depth') != 11:
            failures.append('{}: shipped 4x A100 shape requires node_depth 11'.format(name))
        if not (r.get('diagonal') is True and r.get('long_range_prob') == 0.3
                and r.get('max_long_range') == 8):
            failures.append('{}: brain-like wiring invariant failed'.format(name))
        print('{}: parsed; vocab={}; wiring=valid'.format(name, r.get('vocab_size')))
    a, b = load('config.yaml'), load('config_baseline_feedforward.yaml')
    diffs = []
    for section in set(a) | set(b):
        av, bv = a.get(section), b.get(section)
        if isinstance(av, dict) and isinstance(bv, dict):
            diffs.extend('{}.{}'.format(section, k) for k in set(av) | set(bv) if av.get(k) != bv.get(k))
        elif av != bv:
            diffs.append(section)
    if diffs != ['raslila.depth_sweeps']:
        failures.append('A/B difference is {}, expected only raslila.depth_sweeps'.format(sorted(diffs)))
    else:
        print('A/B differs only by raslila.depth_sweeps (2 vs 1)')
    if failures:
        for failure in failures:
            print('FAIL: ' + failure)
        print('Unread training keys found: {}'.format(sorted({k for n in CONFIGS for k in load(n).get('training', {}) if k not in read_keys})))
        return 1
    print('All active config checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
