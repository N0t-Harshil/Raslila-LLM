import os
import sys

raise SystemExit('This legacy Gutenberg builder is superseded by scripts/get_dataset.py.')

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
from train_raslila import load_bpe, tokenize_corpus_bpe

tok = load_bpe()
arr = tokenize_corpus_bpe(tok, 'data/corpus', cache_path='data/corpus_tokens.npy')
print('TOKENS {:,}'.format(len(arr)))
print('MAXID  {}'.format(int(arr.max())))
print('CACHE  data/corpus_tokens.npy ({:.0f} MB)'.format(
    os.path.getsize('data/corpus_tokens.npy') / 1e6))
