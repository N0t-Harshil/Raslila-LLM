"""Train a byte-level BPE tokenizer for Raslila.

Byte-level tokenization (1 byte = 1 token) is principled but brutally
token-inefficient: English averages ~4 characters per BPE token. That is a 3-4x
tax on effective epochs at fixed compute, and it is the largest single reason
our generations still look like garbage.

This trains a GPT-2-style byte-level BPE on the real corpus, reserving id 0 for
<pad|> so the training loop's ignore_index=0 keeps working unchanged.

Usage:
    python scripts/train_tokenizer.py --vocab-size 199995
"""
import os
import sys
import json
import glob
import time
import argparse

sys.path.insert(0, 'src')

OUT_DIR = 'data/tokenizer'
TOKENIZER_FILE = os.path.join(OUT_DIR, 'raslila_bpe.json')
PAD_TOKEN = '<|pad|>'


def iter_texts(data_dirs, max_chars=None, min_chars=2000):
    """Yield the body text of each file (Gutenberg boilerplate already stripped)."""
    total = 0
    for d in data_dirs:
        for path in sorted(glob.glob(os.path.join(d, '*.txt'))):
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    txt = f.read()
            except OSError:
                continue
            if len(txt) < min_chars:
                continue
            if max_chars and total >= max_chars:
                return
            total += len(txt)
            yield txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab-size', type=int, default=199995)
    ap.add_argument('--data-dirs', type=str, default='data/gutenberg',
                    help='comma-separated dirs of .txt files. Legacy path: the '
                         'production tokenizer is fitted automatically by '
                         'scripts/get_dataset.py on the HuggingFace corpus.')
    ap.add_argument('--max-chars', type=int, default=300_000_000,
                    help='characters of training text. Measured on openwebtext '
                         'with min_frequency=2: 60M reaches only 184,355 merges, '
                         '300M reaches the full 199,995.')
    args = ap.parse_args()

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    os.makedirs(OUT_DIR, exist_ok=True)
    dirs = [d.strip() for d in args.data_dirs.split(',') if d.strip()]

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    print('training BPE vocab={} on {} ...'.format(args.vocab_size, dirs), flush=True)
    t0 = time.time()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[PAD_TOKEN],          # -> id 0
        show_progress=False,
        min_frequency=2,
    )
    tok.train_from_iterator(iter_texts(dirs, max_chars=args.max_chars), trainer=trainer)
    print('trained in {:.0f}s, vocab={}'.format(time.time() - t0, tok.get_vocab_size()), flush=True)

    tok.save(TOKENIZER_FILE)
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'vocab_size': tok.get_vocab_size(),
                   'pad_id': tok.token_to_id(PAD_TOKEN),
                   'pad_token': PAD_TOKEN}, f, indent=2)

    # Measure the win: tokens per character vs byte-level (1 char = 1 token).
    sample = []
    for t in iter_texts(dirs, max_chars=2_000_000):
        sample.append(t)
    text = ''.join(sample)
    n_tok = len(tok.encode(text).ids)
    n_char = len(text)
    print('compression: {:.2f} chars/token ({:,.0f} chars -> {:,.0f} tokens)'.format(
        n_char / n_tok, n_char, n_tok), flush=True)
    print('vocab_size for config.yaml raslila.vocab_size: {}'.format(tok.get_vocab_size()))
    print('saved: {}'.format(TOKENIZER_FILE))


if __name__ == '__main__':
    main()
