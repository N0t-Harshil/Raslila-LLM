"""Test Raslila model."""
import sys, time, os
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
from raslila.raslila import Raslila, RaslilaConfig
import torch

print("=" * 60)
print("TEST RASLILA MODEL")
print("=" * 60)

config = RaslilaConfig(
    dim_x=4, dim_y=4, dim_z=2,  # 32 nodes
    hidden_dim=512,
    num_heads=8,
    num_layers=2,
    ff_dim=2048,
    dropout=0.1,
    edge_hidden_dim=256,
    edge_layers=2,
    local_radius=1,
    long_range_prob=0.1,
    max_long_range=3,
    vocab_size=32000,
    max_seq_len=1024,
    embed_dim=512
)

model = Raslila(config)
print(f"Nodes: {model.n_nodes}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Test forward
x = torch.randint(0, 32000, (2, 64))
start = time.perf_counter()
with torch.no_grad():
    out = model(x)
elapsed = time.perf_counter() - start
print(f"Forward: {elapsed:.3f}s")
print(f"Output: {model(x)['logits'].shape}")
print("Raslila working!")