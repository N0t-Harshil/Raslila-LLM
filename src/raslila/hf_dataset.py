"""HuggingFace corpus acquisition for Raslila.

Replaces the hand-curated Gutenberg corpus (448 books, 85,151,573 tokens) with a
HuggingFace dataset that fetches itself the first time training starts.

WHY THE SWAP.  The 85M-token Gutenberg corpus was the binding constraint on this
project, not the architecture: it is 4.7x short of Chinchilla-optimal for the 20B
config, and even for the 50M config it was 2.7 epochs over a 30k-step run with
the loss flat by step 14k.  openwebtext is ~9.5B tokens of modern web text.

PIPELINE, all steps cached so a re-run is a no-op:

    1. stream the dataset             (HF streaming -- never materialises 40GB)
    2. fit the BPE tokenizer           (only if missing, or fitted on another corpus)
    3. tokenize to <cache>             (uint32 at the 199,995-token vocab)

Under distributed training RANK 0 builds the cache and every other rank waits on
a barrier before loading, so the download happens exactly once.

Memory: the token cache is built with np.concatenate + np.save, so peak RAM is
roughly 2x the cache size (6 GB of chunks + 6 GB of the joined array at
dataset.max_tokens = 1.5e9).  Lower max_tokens if that is a problem.
"""
import json
import os
import time

import numpy as np

DEFAULTS = {
    'name': 'Skylion007/openwebtext',
    'config': None,             # HF dataset config/subset name, or None
    'split': 'train',
    'text_field': 'text',
    'max_tokens': 1_500_000_000,
    'cache': 'data/openwebtext_tokens.npy',
    'tokenizer': 'data/tokenizer/raslila_bpe.json',
    'vocab_size': 199_995,
    'train_tokenizer': True,
    'tokenizer_sample_chars': 300_000_000,
    'shuffle_seed': None,
}

PAD_TOKEN = '<|pad|>'


def resolve(cfg):
    """Merge a config's `dataset:` mapping over DEFAULTS."""
    merged = dict(DEFAULTS)
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _require_datasets():
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            'The `datasets` package is required to fetch a HuggingFace corpus.\n'
            '  pip install datasets\n'
            'or install everything the project needs:\n'
            '  pip install -r requirements.txt\n'
            'Missing: {}'.format(exc))


def _open_stream(ds):
    """Return an iterable dataset, tolerating datasets with no `train` split."""
    from datasets import load_dataset
    kwargs = {'split': ds['split'], 'streaming': True}
    try:
        if ds.get('config'):
            return load_dataset(ds['name'], ds['config'], **kwargs)
        return load_dataset(ds['name'], **kwargs)
    except (ValueError, KeyError):
        pass
    # Fall back to whatever split the dataset actually ships.
    got = load_dataset(ds['name'], ds.get('config'), streaming=True)
    if hasattr(got, 'keys') and not hasattr(got, 'column_names'):
        first = sorted(got.keys())[0]
        return got[first]
    return got


def iter_texts(ds, max_chars=None, log=None):
    """Yield non-empty documents from the configured text field.

    If `text_field` is absent (datasets disagree on the name), fall back to the
    first string column rather than failing after a long download.
    """
    src = _open_stream(ds)
    field = ds.get('text_field') or 'text'
    total = 0
    seen = 0
    for row in src:
        if isinstance(row, dict):
            text = row.get(field)
            if text is None:
                for value in row.values():
                    if isinstance(value, str):
                        text = value
                        break
        else:
            text = row if isinstance(row, str) else None
        if not text:
            continue
        seen += 1
        total += len(text)
        if log and seen % 5000 == 0:
            log('  ... {:,} docs, {:,} chars'.format(seen, total))
        yield text
        if max_chars is not None and total >= max_chars:
            return


def _source_tag(ds):
    """Identity of a fitted tokenizer.

    Sample size is part of the tag on purpose. Measured on openwebtext with
    min_frequency=2 and a 199,995 target: 60M chars only reaches 184,355 merges
    (92.2%), 300M chars reaches the full 199,995. Folding the sample size in
    means a smoke-test fit is correctly discarded by the real run instead of
    being silently reused.
    """
    return '{}::{}::{}chars'.format(
        ds['name'], ds.get('config') or '-', int(ds['tokenizer_sample_chars']))


def _read_meta_source(tok_path):
    meta_path = os.path.join(os.path.dirname(tok_path) or '.', 'meta.json')
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f).get('source')
    except (OSError, ValueError):
        return None


def _write_meta(tok_path, vocab_size, pad_id, source):
    meta_path = os.path.join(os.path.dirname(tok_path) or '.', 'meta.json')
    os.makedirs(os.path.dirname(meta_path) or '.', exist_ok=True)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'vocab_size': vocab_size, 'pad_id': pad_id,
                   'pad_token': PAD_TOKEN, 'source': source}, f, indent=2)


def ensure_tokenizer(ds, log=print):
    """Return a BPE tokenizer path, fitting one on this corpus if needed.

    `train_tokenizer: false` keeps whatever exists (and errors if nothing does).
    `train_tokenizer: true` reuses a tokenizer already fitted on this same
    dataset and otherwise fits a fresh one -- so switching corpora re-fits
    automatically, and re-running does not.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    path = ds['tokenizer']
    tag = _source_tag(ds)
    src = _read_meta_source(path)

    if not ds.get('train_tokenizer', True):
        if not os.path.exists(path):
            raise SystemExit(
                'dataset.train_tokenizer is false but no tokenizer exists at {}.\n'
                'Set dataset.train_tokenizer: true, or run:\n'
                '  python scripts/train_tokenizer.py --vocab-size {}'.format(
                    path, ds['vocab_size']))
        if src != tag:
            log('WARNING: keeping {}, which was fitted on {!r}, but this run '
                'trains on {!r}. Set dataset.train_tokenizer: true to re-fit.'
                .format(path, src, tag))
        return path

    if os.path.exists(path) and src == tag:
        log('Tokenizer: already fitted on {} ({})'.format(tag, path))
        return path

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    log('Fitting BPE vocab={} on a {:,}-char sample of {} ...'.format(
        ds['vocab_size'], int(ds['tokenizer_sample_chars']), tag))
    t0 = time.time()

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=int(ds['vocab_size']),
        special_tokens=[PAD_TOKEN],   # -> id 0, keeps ignore_index=0 valid
        show_progress=False,
        min_frequency=2,
    )
    tok.train_from_iterator(
        iter_texts(ds, max_chars=int(ds['tokenizer_sample_chars'])), trainer)
    tok.save(path)
    _write_meta(path, tok.get_vocab_size(), tok.token_to_id(PAD_TOKEN), tag)
    log('Tokenizer fitted in {:.0f}s -> {} (vocab {})'.format(
        time.time() - t0, path, tok.get_vocab_size()))
    return path


def build_cache(ds, log=print, force=False):
    """Return the token array, downloading and caching it first if necessary."""
    cache = ds['cache']
    if os.path.exists(cache) and not force:
        arr = np.load(cache, mmap_mode='r')
        log('Corpus: {:,} tokens cached at {} (skipping download)'.format(
            len(arr), cache))
        return arr

    _require_datasets()
    tok_path = ensure_tokenizer(ds, log=log)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    if tok.get_vocab_size() > int(ds['vocab_size']):
        raise SystemExit(
            'Tokenizer at {} has vocab {} but raslila.vocab_size is {}. '
            'Raise vocab_size or re-fit with a smaller dataset.vocab_size.'
            .format(tok_path, tok.get_vocab_size(), ds['vocab_size']))

    dtype = np.uint16 if tok.get_vocab_size() <= 65535 else np.uint32
    max_tokens = int(ds['max_tokens'])
    log('Corpus: streaming {} (split {}) -> up to {:,} tokens, {} dtype'.format(
        ds['name'], ds['split'], max_tokens,
        'uint16' if dtype == np.uint16 else 'uint32'))

    chunks = []
    total = 0
    docs = 0
    t0 = time.time()
    for text in iter_texts(ds, log=log):
        ids = tok.encode(text).ids
        if not ids:
            continue
        chunks.append(np.asarray(ids, dtype=dtype))
        total += len(ids)
        docs += 1
        if total >= max_tokens:
            break
    arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=dtype)
    del chunks
    if len(arr) > max_tokens:
        arr = arr[:max_tokens]

    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    np.save(cache, arr)
    log('Corpus: {:,} tokens from {:,} docs in {:.0f}s -> {} ({:.0f} MB)'.format(
        len(arr), docs, time.time() - t0, cache,
        os.path.getsize(cache) / 1e6))
    return arr


def ensure_dataset(cfg, log=print, force=False):
    """Config mapping (may be None) -> token array, building it if needed."""
    if not cfg:
        return None
    return build_cache(resolve(cfg), log=log, force=force)
