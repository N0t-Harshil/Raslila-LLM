"""Data Pipeline — P0 Blocker.

Tokenizer, dataset streaming, dataloader with prefetch.
No training until this exists and classical baseline passes.
"""
import os
import json
import numpy as np
from typing import List, Iterator, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass
class DataConfig:
    vocab_size: int = 8192
    max_seq_len: int = 512
    train_split: float = 0.98
    val_split: float = 0.01
    test_split: float = 0.01
    seed: int = 42
    num_workers: int = 0  # 0 for debugging, >0 for production
    prefetch_factor: int = 2
    pin_memory: bool = True


class BPETokenizer:
    """
    Minimal BPE tokenizer. For production, use tokenizers library (Rust).
    This is a functional implementation for the MVE.
    """
    
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.vocab: dict = {}  # token -> id
        self.id_to_token: dict = {}  # id -> token
        self.merges: list = []  # list of (token1, token2) merge rules
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self._initialized = False
    
    def train(self, texts: List[str], min_freq: int = 2) -> 'BPETokenizer':
        """Train BPE on corpus. Simplified implementation."""
        from collections import Counter
        
        # Initialize with bytes (256) + special tokens (4) = 260
        vocab = {bytes([i]): i for i in range(256)}
        vocab.update({f'<SPECIAL_{i}>': 256 + i for i in range(4)})  # placeholder
        
        # Count word frequencies
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            word_freqs.update(words)
        
        # Build vocabulary by adding most frequent words
        # This is a very simplified BPE — real implementation merges byte pairs
        sorted_words = sorted(word_freqs.items(), key=lambda x: -x[1])
        
        for word, freq in sorted_words:
            if len(vocab) >= self.vocab_size:
                break
            if freq >= min_freq and word not in vocab:
                vocab[word] = len(vocab)
        
        # Ensure special tokens have fixed IDs
        for tok, tid in self.special_tokens.items():
            vocab[tok] = tid
        
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self._initialized = True
        return self
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs. Simplified: word-level with fallback."""
        if not self._initialized:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        tokens = []
        if add_special:
            tokens.append(self.special_tokens['<BOS>'])
        
        words = text.split()
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Fallback: encode as bytes
                for b in word.encode('utf-8'):
                    tokens.append(self.vocab.get(bytes([b]), self.special_tokens['<UNK>']))
        
        if add_special:
            tokens.append(self.special_tokens['<EOS>'])
        return tokens
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        for tid in token_ids:
            if skip_special and tid in self.special_tokens.values():
                continue
            tok = self.id_to_token.get(tid, '<UNK>')
            if isinstance(tok, bytes):
                tokens.append(tok.decode('utf-8', errors='ignore'))
            else:
                tokens.append(tok)
        return ' '.join(tokens)
    
    def save(self, path: str):
        """Save tokenizer to JSON."""
        data = {
            'vocab_size': self.vocab_size,
            'vocab': {str(k): v for k, v in self.vocab.items()},
            'special_tokens': self.special_tokens,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer = cls(data['vocab_size'])
        tokenizer.vocab = {k.encode() if isinstance(k, str) and k.startswith('b\'') else k: v 
                           for k, v in data['vocab'].items()}
        # Fix: vocab keys are strings in JSON, need to handle bytes
        fixed_vocab = {}
        for k, v in data['vocab'].items():
            try:
                fixed_vocab[eval(k)] = v  # bytes literals
            except:
                fixed_vocab[k] = v
        tokenizer.vocab = fixed_vocab
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.special_tokens = data['special_tokens']
        tokenizer._initialized = True
        return tokenizer


class StreamingDataset:
    """
    Memory-efficient streaming dataset for large corpora.
    Yields (input_ids, target_ids) pairs for next-token prediction.
    """
    
    def __init__(self, 
                 file_paths: List[str],
                 tokenizer: BPETokenizer,
                 max_seq_len: int = 512,
                 buffer_size: int = 10000):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self._buffer: List[int] = []
        self._file_idx = 0
        self._file_handle = None
        self._eof = False
        
        # Pre-load all tokens for __len__ support (for small datasets)
        self._all_tokens = []
        for path in file_paths:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            self._all_tokens.extend(tokenizer.encode(text, add_special=False))
    
    def __len__(self):
        # Number of sequences of length max_seq_len we can get
        return max(1, len(self._all_tokens) // self.max_seq_len)
    
    def __getitem__(self, idx):
        # Get sequence starting at idx * max_seq_len
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1
        if end > len(self._all_tokens):
            # Wrap around or pad
            seq = self._all_tokens[start:] + [0] * (end - len(self._all_tokens))
        else:
            seq = self._all_tokens[start:end]
        
        input_ids = np.array(seq[:-1], dtype=np.int32)
        target_ids = np.array(seq[1:], dtype=np.int32)
        return input_ids, target_ids
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        while len(self._buffer) < self.max_seq_len + 1:
            self._fill_buffer()
            if self._eof and len(self._buffer) < self.max_seq_len + 1:
                raise StopIteration
        
        # Get sequence
        seq = self._buffer[:self.max_seq_len + 1]
        self._buffer = self._buffer[1:]  # Sliding window by 1
        
        input_ids = np.array(seq[:-1], dtype=np.int32)
        target_ids = np.array(seq[1:], dtype=np.int32)
        return input_ids, target_ids
    
    def _fill_buffer(self):
        if self._eof:
            return
        
        while len(self._buffer) < self.buffer_size and not self._eof:
            if self._file_handle is None:
                if self._file_idx >= len(self.file_paths):
                    self._eof = True
                    return
                self._file_handle = open(self.file_paths[self._file_idx], 'r', encoding='utf-8', errors='ignore')
            
            chunk = self._file_handle.read(8192)
            if not chunk:
                self._file_handle.close()
                self._file_handle = None
                self._file_idx += 1
                continue
            
            tokens = self.tokenizer.encode(chunk, add_special=False)
            self._buffer.extend(tokens)


class DataCollator:
    """Collate function for DataLoader with padding."""
    
    def __init__(self, pad_token_id: int = 0, max_seq_len: int = 512):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
    
    def __call__(self, batch: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        input_ids, target_ids = zip(*batch)
        
        # Find max length in batch
        max_len = min(max(len(x) for x in input_ids), self.max_seq_len)
        
        # Pad all to max_len
        padded_inputs = []
        padded_targets = []
        for inp, tgt in zip(input_ids, target_ids):
            if len(inp) > max_len:
                inp = inp[:max_len]
                tgt = tgt[:max_len]
            else:
                pad_len = max_len - len(inp)
                inp = np.pad(inp, (0, pad_len), constant_values=self.pad_token_id)
                tgt = np.pad(tgt, (0, pad_len), constant_values=self.pad_token_id)
            padded_inputs.append(inp)
            padded_targets.append(tgt)
        
        return np.stack(padded_inputs), np.stack(padded_targets)


def create_dataloaders(config: DataConfig,
                       train_files: List[str],
                       val_files: List[str],
                       test_files: Optional[List[str]] = None,
                       batch_size: int = 32) -> Tuple:
    """Create train/val/test dataloaders."""
    from torch.utils.data import DataLoader, IterableDataset
    
    # Train tokenizer on sample of training data
    print("Training tokenizer...")
    sample_texts = []
    for f in train_files[:3]:  # Sample first 3 files
        with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
            sample_texts.append(fp.read(100000))  # 100KB sample
    
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.train(sample_texts)
    print(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
    
    # Create datasets
    train_dataset = StreamingDataset(train_files, tokenizer, config.max_seq_len)
    val_dataset = StreamingDataset(val_files, tokenizer, config.max_seq_len)
    test_dataset = StreamingDataset(test_files, tokenizer, config.max_seq_len) if test_files else None
    
    collator = DataCollator(pad_token_id=tokenizer.special_tokens['<PAD>'], 
                            max_seq_len=config.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    ) if test_dataset else None
    
    return train_loader, val_loader, test_loader, tokenizer


def prepare_fineweb_edu(data_dir: str = "data/fineweb_edu",
                        num_shards: int = 10) -> Tuple[List[str], List[str], List[str]]:
    """
    Download and prepare FineWeb-Edu dataset.
    Returns (train_files, val_files, test_files).
    """
    import urllib.request
    import gzip
    
    os.makedirs(data_dir, exist_ok=True)
    
    # FineWeb-Edu sample URLs (using first few shards)
    base_url = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/data/train-"
    shard_files = []
    
    for i in range(min(num_shards, 10)):
        shard_name = f"train-{i:05d}-of-00100.parquet"
        url = base_url + shard_name
        local_path = os.path.join(data_dir, shard_name)
        shard_files.append(local_path)
        
        if not os.path.exists(local_path):
            print(f"Downloading {shard_name}...")
            urllib.request.urlretrieve(url, local_path)
    
    # Convert parquet to text files (simplified)
    # In practice, use datasets library
    train_files = []
    val_files = []
    test_files = []
    
    for i, shard in enumerate(shard_files):
        # Split: 98% train, 1% val, 1% test
        if i < int(num_shards * 0.98):
            train_files.append(shard)
        elif i < int(num_shards * 0.99):
            val_files.append(shard)
        else:
            test_files.append(shard)
    
    return train_files, val_files, test_files


def create_synthetic_data(num_samples: int = 10000, 
                          seq_len: int = 128,
                          vocab_size: int = 1000,
                          output_dir: str = "data/synthetic") -> Tuple[List[str], List[str], List[str]]:
    """Create synthetic data for testing without internet."""
    os.makedirs(output_dir, exist_ok=True)
    
    rng = np.random.default_rng(42)
    
    train_files = []
    val_files = []
    test_files = []
    
    # Create simple repetitive patterns for memorization test
    for split_name, files_list, n in [('train', train_files, int(num_samples * 0.98)),
                                       ('val', val_files, int(num_samples * 0.01)),
                                       ('test', test_files, int(num_samples * 0.01))]:
        for i in range(n):
            # Create document with some structure
            doc_tokens = []
            for _ in range(rng.integers(seq_len // 2, seq_len)):
                # Create patterns: "A B A B" or "X Y Z X Y Z"
                pattern_len = rng.integers(2, 5)
                pattern = rng.integers(10, vocab_size, size=pattern_len)
                repeats = rng.integers(2, 5)
                doc_tokens.extend(np.tile(pattern, repeats))
            
            # Truncate
            doc_tokens = doc_tokens[:seq_len]
            
            # Convert to text (token IDs as words)
            doc_text = ' '.join(f'tok_{t}' for t in doc_tokens)
            
            filepath = os.path.join(output_dir, f'{split_name}_{i:05d}.txt')
            with open(filepath, 'w') as f:
                f.write(doc_text + '\n')
            files_list.append(filepath)
    
    return train_files, val_files, test_files


def get_dataloaders_for_mve(config: DataConfig, batch_size: int = 32):
    """Get dataloaders for Minimal Viable Experiment using synthetic data."""
    train_files, val_files, test_files = create_synthetic_data(
        num_samples=50000,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size
    )
    return create_dataloaders(config, train_files, val_files, test_files, batch_size)


# Classical baseline model (GPT-2 style) for comparison
class ClassicalBaselineConfig:
    """Config for classical GPT-2 style baseline."""
    vocab_size: int = 8192
    max_seq_len: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = False
    
    @property
    def n_params(self) -> int:
        # Rough estimate
        embed = self.vocab_size * self.n_embd * 2  # input + output (tied)
        attn = self.n_layers * (4 * self.n_embd * self.n_embd)  # QKV + O
        ffn = self.n_layers * (2 * self.n_embd * 4 * self.n_embd)  # 2 linear layers, 4x expansion
        norm = self.n_layers * 2 * self.n_embd * 2  # 2 LayerNorms per layer
        return embed + attn + ffn + norm


def count_parameters(model) -> int:
    """Count trainable parameters in PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)