"""Training data for the scaling study: general knowledge + maths + code.

One corpus, shared by every model in the study (Raslila and the baseline, all
sizes), so their losses are directly comparable.

    configs/data.yaml  ->  data/mix/tokens.bin (+ tokens.bin.json sidecar)
                           data/mix/tokenizer.json

SOURCES (weights in configs/data.yaml):
    HuggingFaceFW/fineweb-edu    educational web text      general knowledge
    wikimedia/wikipedia 20231101.en  encyclopedia           facts
    HuggingFaceTB/finemath finemath-4plus  step-by-step maths  maths
    codeparrot/codeparrot-clean  deduplicated Python        code

Documents are interleaved by weight (deficit scheduling), so every prefix of
the stream -- including the validation head -- already has the target mix.
Each source has its own cleaning thresholds (code and LaTeX are symbol-heavy).

TOKENIZER: byte-level BPE, NFC-normalised, with digits split one per token
(maths: "1234" -> 1 2 3 4, so arithmetic sees place value consistently, as in
LLaMA / DeepSeek), <|pad|> = 0 (ignored by the loss), <|endoftext|> = 1.

CACHE: a flat binary file of uint16 token ids, written incrementally while
streaming. Peak RAM stays small however many tokens are requested (the old
np.concatenate path needed ~2x the cache in RAM, impossible at 20B tokens).
"""
import hashlib
import json
import os
import time

import numpy as np
import yaml

from .hf_dataset import clean, _open_stream, _require_datasets, PAD_TOKEN, EOS_TOKEN

RECIPE = 'mix-v1-nfc-digits'

DEFAULTS = {
    'max_tokens': 21_000_000_000,
    'val_tokens': 20_000_000,
    'cache': 'data/mix/tokens.bin',
    'tokenizer': 'data/mix/tokenizer.json',
    'vocab_size': 32768,
    'tokenizer_sample_chars': 2_000_000_000,
    'split_digits': True,
    'shuffle_seed': 1234,
    'shuffle_buffer': 10_000,
    'min_chars': 200,
    'min_alpha_frac': 0.6,
    'dedup': True,
    'split': 'train',
    'text_field': 'text',
    'sources': [],
}

_INHERITED = ('split', 'text_field', 'shuffle_seed', 'shuffle_buffer',
              'min_chars', 'min_alpha_frac', 'dedup')


def load_data_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    return resolve(raw.get('data', raw))


def resolve(cfg):
    ds = dict(DEFAULTS)
    ds.update({k: v for k, v in (cfg or {}).items() if v is not None})
    full = []
    for src in ds['sources']:
        one = {k: ds[k] for k in _INHERITED}
        one.update({'config': None, 'weight': 1.0})
        one.update({k: v for k, v in src.items() if v is not None})
        full.append(one)
    if not full:
        raise SystemExit('data config has no `sources`')
    tot = sum(float(s['weight']) for s in full)
    for s in full:
        s['weight'] = float(s['weight']) / tot
    ds['sources'] = full
    for k in ('max_tokens', 'val_tokens', 'vocab_size', 'tokenizer_sample_chars'):
        ds[k] = int(float(ds[k]))
    return ds


def signature(ds):
    parts = [RECIPE, str(ds['vocab_size']), str(ds['split_digits'])]
    for s in ds['sources']:
        parts.append('{}/{}:{}@{:.4f}[{},{}]'.format(
            s['name'], s.get('config') or '-', s.get('text_field'), s['weight'],
            s.get('min_chars'), s.get('min_alpha_frac')))
    return ';'.join(parts)


def _iter_source(src, seen, stats):
    stream = _open_stream(src)
    if src.get('shuffle_seed') is not None and hasattr(stream, 'shuffle'):
        stream = stream.shuffle(seed=int(src['shuffle_seed']),
                                buffer_size=int(src.get('shuffle_buffer') or 10_000))
    field = src.get('text_field') or 'text'
    rec = stats['per_source'].setdefault(src['name'], {
        'docs': 0, 'chars': 0, 'dropped_junk': 0, 'dropped_dup': 0})
    for row in stream:
        text = row.get(field) if isinstance(row, dict) else row
        if not isinstance(text, str) or not text:
            continue
        text = clean(text, src)
        if text is None:
            rec['dropped_junk'] += 1
            continue
        if seen is not None and src.get('dedup', True):
            h = hashlib.blake2b(text.encode('utf-8', 'ignore'), digest_size=8).digest()
            if h in seen:
                rec['dropped_dup'] += 1
                continue
            seen.add(h)
        rec['docs'] += 1
        rec['chars'] += len(text)
        yield text


def iter_mixed(ds, max_chars=None, log=None, stats=None):
    """Weighted interleave of all sources; next doc comes from the source
    furthest below its share of characters so far."""
    sources = ds['sources']
    stats = stats if stats is not None else {}
    stats.setdefault('per_source', {})
    seen = set() if ds.get('dedup', True) else None
    its = [None] * len(sources)
    produced = [0] * len(sources)
    alive = [True] * len(sources)
    total = docs = 0
    t0 = time.time()
    while any(alive):
        i = min((j for j in range(len(sources)) if alive[j]),
                key=lambda j: produced[j] / sources[j]['weight'])
        if its[i] is None:
            if log:
                log('  opening {} ({}) weight {:.0%}'.format(
                    sources[i]['name'], sources[i].get('config') or '-', sources[i]['weight']))
            its[i] = _iter_source(sources[i], seen, stats)
        try:
            text = next(its[i])
        except StopIteration:
            alive[i] = False
            if log:
                log('  source {} exhausted'.format(sources[i]['name']))
            continue
        produced[i] += len(text)
        total += len(text)
        docs += 1
        if log and docs % 100_000 == 0:
            log('  ... {:,} docs, {:.2f}B chars, {:.1f} MB/s | {}'.format(
                docs, total / 1e9, total / 1e6 / max(1e-6, time.time() - t0),
                ' '.join('{}={:.0%}'.format(s['name'].split('/')[-1], produced[k] / max(1, total))
                         for k, s in enumerate(sources))))
        yield text
        if max_chars is not None and total >= max_chars:
            return


def ensure_tokenizer(ds, log=print):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
    path = ds['tokenizer']
    meta = path + '.meta.json'
    sig = signature(ds) + '::{}chars'.format(ds['tokenizer_sample_chars'])
    if os.path.exists(path) and os.path.exists(meta):
        with open(meta, 'r', encoding='utf-8') as f:
            if json.load(f).get('signature') == sig:
                log('Tokenizer: reusing {}'.format(path))
                return path
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    log('Fitting BPE vocab {} on {:,} chars of the mix ...'.format(
        ds['vocab_size'], ds['tokenizer_sample_chars']))
    t0 = time.time()
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.NFC()
    pre = [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)]
    if ds['split_digits']:
        pre.insert(0, pre_tokenizers.Digits(individual_digits=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(pre)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=ds['vocab_size'],
                                  special_tokens=[PAD_TOKEN, EOS_TOKEN],
                                  min_frequency=2, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(iter_mixed(ds, max_chars=ds['tokenizer_sample_chars'], log=log), trainer)
    tok.save(path)
    with open(meta, 'w', encoding='utf-8') as f:
        json.dump({'signature': sig, 'vocab_size': tok.get_vocab_size(),
                   'pad_id': tok.token_to_id(PAD_TOKEN), 'eos_id': tok.token_to_id(EOS_TOKEN)}, f, indent=2)
    log('Tokenizer fitted in {:.0f}s -> {} (vocab {})'.format(time.time() - t0, path, tok.get_vocab_size()))
    return path


def _sidecar(cache):
    return cache + '.json'


def cache_status(ds):
    """(ok, info) -- ok means a complete cache for exactly this config exists."""
    side = _sidecar(ds['cache'])
    if not (os.path.exists(ds['cache']) and os.path.exists(side)):
        return False, None
    with open(side, 'r', encoding='utf-8') as f:
        info = json.load(f)
    ok = info.get('signature') == signature(ds) and info.get('complete') and \
        info.get('max_tokens') == ds['max_tokens']
    return bool(ok), info


def build_cache(ds, log=print, force=False):
    ok, info = cache_status(ds)
    if ok and not force:
        log('Corpus: {:,} tokens cached at {}'.format(info['tokens'], ds['cache']))
        return info
    _require_datasets()
    tok_path = ensure_tokenizer(ds, log=log)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    if tok.get_vocab_size() > 65535:
        raise SystemExit('vocab > 65535 does not fit the uint16 cache')
    eos = tok.token_to_id(EOS_TOKEN)
    cache = ds['cache']
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    tmp = cache + '.partial'
    stats = {'per_source': {}}
    total = docs = 0
    t0 = time.time()
    log('Corpus: streaming the mix -> {} (target {:,} tokens)'.format(cache, ds['max_tokens']))
    batch = []
    with open(tmp, 'wb') as out:
        def flush():
            nonlocal total, docs
            for enc in tok.encode_batch(batch):
                if not enc.ids:
                    continue
                ids = np.asarray(enc.ids + [eos], dtype=np.uint16)
                room = ds['max_tokens'] - total
                if room <= 0:
                    break
                ids = ids[:room]
                out.write(ids.tobytes())
                total += len(ids)
                docs += 1
            batch.clear()

        last = time.time()
        for text in iter_mixed(ds, log=log, stats=stats):
            batch.append(text)
            if len(batch) >= 2048:
                flush()
                if time.time() - last > 60:
                    last = time.time()
                    rate = total / max(1e-6, last - t0)
                    log('  tokens {:,} / {:,} ({:.1%}), {:,.0f} tok/s, ETA {:.1f} h'.format(
                        total, ds['max_tokens'], total / ds['max_tokens'], rate,
                        (ds['max_tokens'] - total) / max(1.0, rate) / 3600))
                if total >= ds['max_tokens']:
                    break
        if batch and total < ds['max_tokens']:
            flush()
    os.replace(tmp, cache)
    info = {'signature': signature(ds), 'complete': True, 'max_tokens': ds['max_tokens'],
            'tokens': total, 'docs': docs, 'dtype': 'uint16', 'tokenizer': tok_path,
            'vocab_size': tok.get_vocab_size(), 'seconds': round(time.time() - t0),
            'per_source': stats['per_source']}
    with open(_sidecar(cache), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    log('Corpus: {:,} tokens from {:,} docs in {:.1f} h -> {}'.format(
        total, docs, (time.time() - t0) / 3600, cache))
    for name, r in stats['per_source'].items():
        log('  {:<34} {:>10,} docs {:>8.2f}B chars  (dropped {:,} junk, {:,} dup)'.format(
            name, r['docs'], r['chars'] / 1e9, r['dropped_junk'], r['dropped_dup']))
    if total < ds['max_tokens']:
        log('WARNING: sources ran dry at {:,} tokens (< {:,} requested).'.format(total, ds['max_tokens']))
    return info


def open_tokens(ds):
    """Memory-mapped token array of the finished cache."""
    ok, info = cache_status(ds)
    if not ok:
        raise SystemExit('No complete token cache at {} for this data config.\n'
                         'Build it first:  python run.py data'.format(ds['cache']))
    return np.memmap(ds['cache'], dtype=np.uint16, mode='r'), info
