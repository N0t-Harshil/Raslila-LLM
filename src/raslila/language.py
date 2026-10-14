"""Complete Language Interface - Text ↔ Quantum Hamiltonian ↔ Tokens.

Implements:
- Tokenization (BPE-style or character-level)
- Text → Hamiltonian parameter mapping (embeddings → property vectors)
- Hamiltonian → token probabilities (readout marginals)
- Autoregressive generation
- Training on language data
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import Counter
import re


@dataclass
class Tokenizer:
    """Simple BPE-style tokenizer."""
    vocab: Dict[str, int]
    merges: List[Tuple[str, str]]
    vocab_size: int

    @classmethod
    def from_texts(cls, texts: List[str], vocab_size: int = 1000,
                   min_freq: int = 2) -> 'Tokenizer':
        """Train tokenizer on corpus."""
        # Character-level base vocab
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)

        vocab = {ch: i for i, (ch, _) in enumerate(char_counts.most_common())}
        merges = []

        # Simple BPE merges (placeholder - real BPE is more complex)
        # For now, use character + common bigrams
        words = ' '.join(texts).split()
        word_counts = Counter(words)

        for word, _ in word_counts.most_common(vocab_size - len(vocab)):
            if len(word) > 1 and word not in vocab:
                vocab[word] = len(vocab)
                # Add character merges
                for i in range(len(word)-1):
                    merges.append((word[i], word[i+1]))

        return cls(vocab=vocab, merges=merges, vocab_size=len(vocab))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        # Simple word-piece encoding
        tokens = []
        words = text.lower().split()
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Fallback to characters
                for ch in word:
                    tokens.append(self.vocab.get(ch, self.vocab.get(' ', 0)))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return ' '.join(inv_vocab.get(tid, '<UNK>') for tid in token_ids)


class RaslilaEmbedding:
    """
    Maps tokens/text to Hamiltonian property vectors.

    Each token has a learned 4D embedding [usage, meaning, reflection, temporality]
    that biases the edge property vectors. This is the "text → Hamiltonian" bridge.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 4, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Token embeddings: each token -> 4D bias vector
        self.token_embeddings = rng.normal(0, 0.1, size=(vocab_size, embed_dim))
        # Position embeddings for sequence position
        self.max_seq_len = 512
        self.pos_embeddings = rng.normal(0, 0.05, size=(self.max_seq_len, embed_dim))
        # Projection from token+pos to property vector per edge
        self.edge_projection = rng.normal(0, 0.1, size=(embed_dim * 2, embed_dim))

    def token_to_property_bias(self, token_id: int, position: int = 0) -> np.ndarray:
        """Get 4D property bias for a token at position."""
        token_emb = self.token_embeddings[token_id % self.vocab_size]
        pos_emb = self.pos_embeddings[position % self.max_seq_len]
        combined = np.concatenate([token_emb, pos_emb])
        return np.tanh(combined @ self.edge_projection)

    def text_to_property_bias(self, token_ids: List[int]) -> np.ndarray:
        """Aggregate property bias from sequence of tokens."""
        if not token_ids:
            return np.zeros(self.embed_dim)
        biases = [self.token_to_property_bias(tid, i)
                  for i, tid in enumerate(token_ids)]
        return np.mean(biases, axis=0)

    def update_embeddings(self, token_id: int, grad: np.ndarray, lr: float = 0.01):
        """Update token embedding via gradient (for end-to-end training)."""
        self.token_embeddings[token_id] -= lr * grad[:self.embed_dim]
        self.edge_projection -= lr * np.outer(
            np.concatenate([self.token_embeddings[token_id],
                           self.pos_embeddings[0]]), grad)


class QuantumReadout:
    """
    Maps quantum state probabilities to token probabilities.

    Uses readout qubits (subset of universes) to compute marginal
    probabilities for each token in vocabulary.
    """

    def __init__(self, n_qubits: int, vocab_size: int,
                 readout_qubits: Tuple[int, ...] = (0, 2, 4),
                 n_universes: int = 3):
        self.n_qubits = n_qubits
        self.vocab_size = vocab_size
        self.readout_qubits = readout_qubits
        self.n_readout = len(readout_qubits)
        self.n_universes = n_universes

        # Token mapping: each token -> bit pattern on readout qubits
        # 3 readout qubits = 8 patterns, map to tokens cyclically
        self.token_to_bits: Dict[int, List[int]] = {}
        for tid in range(vocab_size):
            bits = [(tid >> (self.n_readout - 1 - i)) & 1
                    for i in range(self.n_readout)]
            self.token_to_bits[tid] = bits

    def marginals_to_token_probs(self, state_probs: np.ndarray,
                                  temperature: float = 1.0) -> np.ndarray:
        """Convert quantum marginals to token probability distribution."""
        logits = np.zeros(self.vocab_size)

        for tid in range(self.vocab_size):
            bits = tuple(self.token_to_bits[tid])
            m = self._marginal_prob(state_probs, bits)
            logits[tid] = np.log(m + 1e-12) / temperature

        # Softmax
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def _marginal_prob(self, probs: np.ndarray, bits: Tuple[int, ...]) -> float:
        """Sum probabilities matching given bits on readout qubits."""
        total = 0.0
        for idx in range(2**self.n_qubits):
            # Check if qubits match bits (qubit 0 = MSB)
            match = True
            for q_idx, q in enumerate(self.readout_qubits):
                bit = (idx >> (self.n_qubits - 1 - q)) & 1
                if bit != bits[q_idx]:
                    match = False
                    break
            if match:
                total += probs[idx]
        return total

    def sample_token(self, state_probs: np.ndarray,
                     temperature: float = 1.0, top_k: int = 0) -> int:
        """Sample token from quantum state."""
        token_probs = self.marginals_to_token_probs(state_probs, temperature)
        if top_k > 0:
            top_indices = np.argpartition(token_probs, -top_k)[-top_k:]
            mask = np.zeros_like(token_probs, dtype=bool)
            mask[top_indices] = True
            token_probs = np.where(mask, token_probs, 0)
            token_probs /= token_probs.sum()
        return int(np.random.choice(self.vocab_size, p=token_probs))


class RaslilaLanguageModel:
    """
    Complete language model interface for Raslila.

    Text → [Tokenizer] → Token IDs
         → [Embedding] → Property biases per edge
         → [MLP] → J_ij couplings
         → [Hamiltonian] → Quantum state
         → [Readout] → Token probabilities
         → [Sampler] → Next token
    """

    def __init__(self, model: 'RaslilaModel', tokenizer: Tokenizer,
                 embedding: RaslilaEmbedding, readout: QuantumReadout):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.readout = readout

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def forward(self, text: str, return_state: bool = False) -> Dict:
        """Full forward pass: text → token probabilities."""
        token_ids = self.encode(text)
        bias = self.embedding.text_to_property_bias(token_ids)

        # Temporarily modify model's property vector function
        orig_pv = self.model.property_vector
        self.model.property_vector = lambda text, edge=None, step=None: (
            orig_pv(text, edge, step) + bias
        )

        try:
            result = self.model.forward(text)
        finally:
            self.model.property_vector = orig_pv

        probs = self.readout.marginals_to_token_probs(result['probs'])
        result['token_probs'] = probs
        result['next_token_id'] = int(np.argmax(probs))
        result['next_token'] = self.tokenizer.decode([result['next_token_id']])

        if return_state:
            result['quantum_state'] = result['state']
        return result

    def generate(self, prompt: str, max_new_tokens: int = 20,
                 temperature: float = 1.0, top_k: int = 0) -> str:
        """Autoregressive generation."""
        generated = self.encode(prompt)
        for _ in range(max_new_tokens):
            current_text = self.decode(generated)
            result = self.model.forward(current_text)
            next_token = self.readout.sample_token(
                result['probs'], temperature, top_k)
            generated.append(next_token)
            if next_token == self.tokenizer.vocab.get('.', -1):
                break
        return self.decode(generated)

    def compute_nll_loss(self, text: str, target_token: str) -> float:
        """Compute negative log-likelihood loss for a (text, target) pair."""
        target_id = self.tokenizer.vocab.get(target_token, 0)
        result = self.forward(text)
        token_probs = result['token_probs']
        return -np.log(token_probs[target_id] + 1e-12)

    def train_step(self, text: str, target_token: str,
                   lr_embed: float = 0.01, **train_kwargs) -> Dict:
        """Single training step on (context, target) pair."""
        # Forward
        loss = self.compute_nll_loss(text, target_token)
        # Backward: gradient through readout → Hamiltonian → MLP → Embedding
        # This is complex; for now use model's train_step with target bits
        target_bits = self.readout.token_to_bits[
            self.tokenizer.vocab.get(target_token, 0)]
        # Convert bits to token index for model's train_step
        model_target = target_bits  # simplified
        model_loss = self.model.train_step(text, target_token, **train_kwargs)

        return {
            'nll_loss': loss,
            'model_loss': model_loss,
            'token_probs': self.model.forward(text)['token_probs']
        }


def create_language_model(n_layers: int = 2, universes: Optional[List[str]] = None,
                          vocab_size: int = 256,
                          texts: Optional[List[str]] = None) -> RaslilaLanguageModel:
    """Factory function to create complete language model."""
    from .model import RaslilaModel

    # Build tokenizer
    if texts:
        tokenizer = Tokenizer.from_texts(texts, vocab_size=vocab_size)
    else:
        # Default character-level
        vocab = {chr(i): i for i in range(32, 127)}
        vocab['<UNK>'] = len(vocab)
        vocab['<PAD>'] = len(vocab)
        tokenizer = Tokenizer(vocab=vocab, merges=[], vocab_size=len(vocab))

    # Build model
    model = RaslilaModel(n_layers=n_layers, universes=universes)

    # Build embedding
    embedding = RaslilaEmbedding(tokenizer.vocab_size)

    # Build readout
    readout = QuantumReadout(model.n_qubits, tokenizer.vocab_size,
                             readout_qubits=model.readout_qubits,
                             n_universes=model.n_universes)

    return RaslilaLanguageModel(model, tokenizer, embedding, readout)


def train_on_corpus(lm: RaslilaLanguageModel, corpus: List[str],
                    epochs: int = 1, lr: float = 0.01) -> List[float]:
    """Train language model on corpus (next-token prediction)."""
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_samples = 0
        for text in corpus:
            tokens = lm.encode(text)
            if len(tokens) < 2:
                continue
            # Train on each (context, next_token) pair
            for i in range(len(tokens) - 1):
                context = lm.decode(tokens[:i+1])
                target = lm.decode([tokens[i+1]])
                result = lm.train_step(context, target, lr_mlp=lr)
                epoch_loss += result['nll_loss']
                n_samples += 1
        avg_loss = epoch_loss / max(n_samples, 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}: avg NLL = {avg_loss:.4f}")
    return losses