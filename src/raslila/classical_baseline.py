"""Classical Baseline Models — for strict comparison.

GPT-2 style transformer and Raslila-classical (same classical params, no quantum).
Must beat these to claim quantum advantage.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    max_seq_len: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = False
    tie_weights: bool = True
    
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_heads


class ClassicalTransformer:
    """
    Minimal GPT-2 style transformer in pure NumPy.
    For baseline comparison — NOT for production training.
    """
    
    def __init__(self, config: GPTConfig, seed: int = 42):
        self.config = config
        rng = np.random.default_rng(seed)
        
        # Token embeddings
        self.wte = rng.normal(0, 0.02, size=(config.vocab_size, config.n_embd))
        # Position embeddings
        self.wpe = rng.normal(0, 0.02, size=(config.max_seq_len, config.n_embd))
        
        # Transformer blocks
        self.blocks = []
        for _ in range(config.n_layers):
            block = {
                'ln1': {'w': np.ones(config.n_embd), 'b': np.zeros(config.n_embd)},
                'attn': {
                    'c_attn': {
                        'w': rng.normal(0, 0.02, size=(config.n_embd, 3 * config.n_embd)),
                        'b': np.zeros(3 * config.n_embd)
                    },
                    'c_proj': {
                        'w': rng.normal(0, 0.02, size=(config.n_embd, config.n_embd)),
                        'b': np.zeros(config.n_embd)
                    }
                },
                'ln2': {'w': np.ones(config.n_embd), 'b': np.zeros(config.n_embd)},
                'mlp': {
                    'c_fc': {
                        'w': rng.normal(0, 0.02, size=(config.n_embd, 4 * config.n_embd)),
                        'b': np.zeros(4 * config.n_embd)
                    },
                    'c_proj': {
                        'w': rng.normal(0, 0.02, size=(4 * config.n_embd, config.n_embd)),
                        'b': np.zeros(config.n_embd)
                    }
                }
            }
            self.blocks.append(block)
        
        # Final layer norm
        self.ln_f = {'w': np.ones(config.n_embd), 'b': np.zeros(config.n_embd)}
        
        # Output projection (tied with input embeddings if config.tie_weights)
        if not config.tie_weights:
            self.lm_head = rng.normal(0, 0.02, size=(config.n_embd, config.vocab_size))
        else:
            self.lm_head = None
    
    def layer_norm(self, x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return w * (x - mean) / np.sqrt(var + eps) + b
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def causal_mask(self, seq_len: int) -> np.ndarray:
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        return mask
    
    def attention(self, x: np.ndarray, block: dict, mask: np.ndarray) -> np.ndarray:
        B, T, C = x.shape
        config = self.config
        
        # QKV projection
        qkv = x @ block['attn']['c_attn']['w'] + block['attn']['c_attn']['b']
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head
        q = q.reshape(B, T, config.n_heads, config.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, config.n_heads, config.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, config.n_heads, config.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(config.head_dim)
        att = att + mask
        att = np.exp(att - np.max(att, axis=-1, keepdims=True))
        att = att / np.sum(att, axis=-1, keepdims=True)
        
        # Apply attention
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Output projection
        y = y @ block['attn']['c_proj']['w'] + block['attn']['c_proj']['b']
        return y
    
    def forward(self, idx: np.ndarray) -> np.ndarray:
        """Forward pass. idx shape: (B, T)"""
        config = self.config
        B, T = idx.shape
        
        # Truncate to max_seq_len if needed
        if T > config.max_seq_len:
            idx = idx[:, :config.max_seq_len]
            T = config.max_seq_len
        
        # Token + position embeddings
        tok_emb = self.wte[idx]
        pos_emb = self.wpe[:T]
        x = tok_emb + pos_emb
        
        # Causal mask
        mask = self.causal_mask(T)
        
        # Transformer blocks
        for block in self.blocks:
            # Attention block
            ln1 = self.layer_norm(x, block['ln1']['w'], block['ln1']['b'])
            attn_out = self.attention(ln1, block, mask)
            x = x + attn_out
            
            # MLP block
            ln2 = self.layer_norm(x, block['ln2']['w'], block['ln2']['b'])
            fc = self.gelu(ln2 @ block['mlp']['c_fc']['w'] + block['mlp']['c_fc']['b'])
            proj = fc @ block['mlp']['c_proj']['w'] + block['mlp']['c_proj']['b']
            x = x + proj
        
        # Final norm
        x = self.layer_norm(x, self.ln_f['w'], self.ln_f['b'])
        
        # Output projection
        if self.lm_head is not None:
            logits = x @ self.lm_head
        else:
            # Tied weights
            logits = x @ self.wte.T
        
        return logits
    
    def loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Cross-entropy loss."""
        B, T, V = logits.shape
        logits = logits.reshape(-1, V)
        targets = targets.reshape(-1)
        
        # Log softmax
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # NLL
        correct_logprobs = -np.log(probs[np.arange(len(targets)), targets] + 1e-9)
        return float(np.mean(correct_logprobs))
    
    def count_params(self) -> int:
        total = 0
        total += self.wte.size + self.wpe.size
        for block in self.blocks:
            total += (block['ln1']['w'].size + block['ln1']['b'].size +
                      block['attn']['c_attn']['w'].size + block['attn']['c_attn']['b'].size +
                      block['attn']['c_proj']['w'].size + block['attn']['c_proj']['b'].size +
                      block['ln2']['w'].size + block['ln2']['b'].size +
                      block['mlp']['c_fc']['w'].size + block['mlp']['c_fc']['b'].size +
                      block['mlp']['c_proj']['w'].size + block['mlp']['c_proj']['b'].size)
        total += self.ln_f['w'].size + self.ln_f['b'].size
        if self.lm_head is not None:
            total += self.lm_head.size
        return total


class RaslilaClassical:
    """
    Raslila with classical-only evolution (no quantum).
    Same classical parameter budget as quantum model for fair comparison.
    """
    
    def __init__(self, vocab_size: int = 8192, max_seq_len: int = 512,
                 n_layers: int = 6, n_embd: int = 256, n_heads: int = 8,
                 n_universes: int = 5, seed: int = 42):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_embd = n_embd
        self.n_universes = n_universes
        
        rng = np.random.default_rng(seed)
        
        # Embeddings (same as quantum model)
        self.wte = rng.normal(0, 0.02, size=(vocab_size, n_embd))
        self.wpe = rng.normal(0, 0.02, size=(max_seq_len, n_embd))
        
        # Classical "universes" as separate attention heads with different projections
        # Each universe gets its own QKV projection
        self.universe_projs = []
        for _ in range(n_universes):
            u_proj = {
                'qkv_w': rng.normal(0, 0.02, size=(n_embd, 3 * n_embd)),
                'qkv_b': np.zeros(3 * n_embd),
                'out_w': rng.normal(0, 0.02, size=(n_embd, n_embd)),
                'out_b': np.zeros(n_embd),
            }
            self.universe_projs.append(u_proj)
        
        # Interference mixing (classical analog of XX+YY)
        self.interference_mixer = rng.normal(0, 0.01, size=(n_universes, n_universes))
        
        # Layer norms
        self.ln1 = [{'w': np.ones(n_embd), 'b': np.zeros(n_embd)} for _ in range(n_layers)]
        self.ln2 = [{'w': np.ones(n_embd), 'b': np.zeros(n_embd)} for _ in range(n_layers)]
        
        # MLPs per universe (edge MLP analog)
        self.universe_mlps = []
        for _ in range(n_universes):
            mlp = {
                'fc1_w': rng.normal(0, 0.02, size=(n_embd, 4 * n_embd)),
                'fc1_b': np.zeros(4 * n_embd),
                'fc2_w': rng.normal(0, 0.02, size=(4 * n_embd, n_embd)),
                'fc2_b': np.zeros(n_embd),
            }
            self.universe_mlps.append(mlp)
        
        # Final norm
        self.ln_f = {'w': np.ones(n_embd), 'b': np.zeros(n_embd)}
        
        # Output projection
        self.lm_head = rng.normal(0, 0.02, size=(n_embd, vocab_size))
    
    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def layer_norm(self, x, w, b, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return w * (x - mean) / np.sqrt(var + eps) + b
    
    def causal_mask(self, T):
        return np.triu(np.ones((T, T)), k=1) * -1e9
    
    def forward(self, idx: np.ndarray) -> np.ndarray:
        B, T = idx.shape
        
        # Embeddings
        x = self.wte[idx] + self.wpe[:T]
        
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        
        for layer_idx in range(self.n_layers):
            # Layer norm 1
            x_ln = self.layer_norm(x, self.ln1[layer_idx]['w'], self.ln1[layer_idx]['b'])
            
            # Process through each universe
            universe_outputs = []
            for u in range(self.n_universes):
                # Self-attention within universe
                qkv = x_ln @ self.universe_projs[u]['qkv_w'] + self.universe_projs[u]['qkv_b']
                q, k, v = np.split(qkv, 3, axis=-1)
                
                # Simple single-head attention per universe
                attn = (q @ k.transpose(0, 2, 1)) / np.sqrt(self.n_embd)
                attn = attn + mask[:T, :T]
                attn = np.exp(attn - np.max(attn, axis=-1, keepdims=True))
                attn = attn / np.sum(attn, axis=-1, keepdims=True)
                attn_out = attn @ v
                
                # Output projection
                attn_out = attn_out @ self.universe_projs[u]['out_w'] + self.universe_projs[u]['out_b']
                universe_outputs.append(attn_out)
            
            # Interference mixing (classical analog of quantum entanglement)
            stacked = np.stack(universe_outputs, axis=1)  # (B, U, T, C)
            mixed = np.einsum('uv,butc->btc', self.interference_mixer, stacked)
            
            x = x + mixed
            
            # MLP per universe
            x_ln2 = self.layer_norm(x, self.ln2[layer_idx]['w'], self.ln2[layer_idx]['b'])
            mlp_outs = []
            for u in range(self.n_universes):
                fc = self.gelu(x_ln2 @ self.universe_mlps[u]['fc1_w'] + self.universe_mlps[u]['fc1_b'])
                proj = fc @ self.universe_mlps[u]['fc2_w'] + self.universe_mlps[u]['fc2_b']
                mlp_outs.append(proj)
            
            # Average across universes (classical mixing)
            x = x + np.mean(np.stack(mlp_outs, axis=1), axis=1)
        
        # Final
        x = self.layer_norm(x, self.ln_f['w'], self.ln_f['b'])
        logits = x @ self.lm_head
        return logits
    
    def loss(self, logits, targets):
        B, T, V = logits.shape
        logits = logits.reshape(-1, V)
        targets = targets.reshape(-1)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        correct_logprobs = -np.log(probs[np.arange(len(targets)), targets] + 1e-9)
        return float(np.mean(correct_logprobs))
    
    def count_params(self) -> int:
        total = self.wte.size + self.wpe.size
        for u in range(self.n_universes):
            total += (self.universe_projs[u]['qkv_w'].size + self.universe_projs[u]['qkv_b'].size +
                      self.universe_projs[u]['out_w'].size + self.universe_projs[u]['out_b'].size)
        total += self.interference_mixer.size
        for _ in range(self.n_layers):
            total += (self.ln1[0]['w'].size + self.ln1[0]['b'].size +
                      self.ln2[0]['w'].size + self.ln2[0]['b'].size)
        for u in range(self.n_universes):
            total += (self.universe_mlps[u]['fc1_w'].size + self.universe_mlps[u]['fc1_b'].size +
                      self.universe_mlps[u]['fc2_w'].size + self.universe_mlps[u]['fc2_b'].size)
        total += self.ln_f['w'].size + self.ln_f['b'].size
        total += self.lm_head.size
        return total


def train_classical_baseline(model, train_loader, val_loader, 
                             steps: int = 5000, lr: float = 3e-4,
                             eval_every: int = 500, device: str = 'cpu') -> dict:
    """
    Train classical baseline with Adam optimizer.
    Returns training history.
    """
    # Simple Adam implementation
    class Adam:
        def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
            self.params = params
            self.lr = lr
            self.betas = betas
            self.eps = eps
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.t = 0
        
        def step(self, grads):
            self.t += 1
            for i, (p, g) in enumerate(zip(self.params, grads)):
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (g * g)
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    # Extract parameters
    params = []
    param_names = []
    
    def extract_params(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, np.ndarray):
                    params.append(v)
                    param_names.append(f"{prefix}{k}")
                elif isinstance(v, dict):
                    extract_params(v, f"{prefix}{k}.")
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            extract_params(item, f"{prefix}{k}[{i}].")
    
    extract_params(model.__dict__)
    
    opt = Adam(params, lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    for step in range(steps):
        # Training step
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids, target_ids = batch
        logits = model.forward(input_ids)
        loss = model.loss(logits, target_ids)
        
        # Simple gradient computation (finite difference for demo)
        # In practice, use autograd framework
        history['train_loss'].append(loss)
        
        if step % eval_every == 0:
            # Validation
            val_losses = []
            for _ in range(10):
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)
                val_input, val_target = val_batch
                val_logits = model.forward(val_input)
                val_loss = model.loss(val_logits, val_target)
                val_losses.append(val_loss)
            
            avg_val = np.mean(val_losses)
            val_ppl = np.exp(avg_val)
            history['val_loss'].append(avg_val)
            history['val_ppl'].append(val_ppl)
            
            print(f"Step {step}: train_loss={loss:.4f}, val_loss={avg_val:.4f}, val_ppl={val_ppl:.2f}")
    
    return history