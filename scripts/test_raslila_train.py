"""Test Raslila training."""
import sys, time, os
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
from raslila.raslila import Raslila, RaslilaConfig
import torch

print("=" * 60)
print("TEST RASLILA MODEL TRAINING")
print("=" * 60)

config = RaslilaConfig(
    dim_x=3, dim_y=3, dim_z=2,  # 18 nodes
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    ff_dim=1024,
    dropout=0.1,
    edge_hidden_dim=128,
    edge_layers=2,
    local_radius=1,
    long_range_prob=0.1,
    max_long_range=3,
    vocab_size=8000,
    max_seq_len=256,
    embed_dim=256
)

model = Raslila(config)
print(f"Nodes: {model.n_nodes}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

pairs = [
    ("grief sorrow", "grief"), ("sadness tears", "grief"),
    ("logic reason", "logic"), ("if then", "logic"),
    ("dream sleep", "dream"), ("vision future", "dream"),
    ("love kindness", "empathy"), ("empathy care", "empathy"),
    ("memory recall", "memory"), ("past history", "memory"),
    ("clear think", "clarity"), ("focus mind", "clarity"),
]

print("Training 10 steps...")
start = time.perf_counter()
for step in range(10):
    for prompt, target in pairs:
        r = model.train_step(prompt, target, lr_mlp=0.03)
    if step % 3 == 0:
        r = model.train_step("test", "test", lr_mlp=0.001)
        print(f"Step {step}: loss={r['loss_after']:.4f} a={model.alpha.item():+.3f} b={model.beta.item():+.3f}")

print(f"\nDone in {time.perf_counter()-start:.1f}s")
print(f"Final: a={model.alpha.item():+.3f}, b={model.beta.item():+.3f}")

for p in ["grief sorrow", "logic reason", "dream sleep", "love kindness"]:
    r = model.forward(p)
    print(f"  '{p}' -> '{r['token']}' (conf: {r['confidence']:.3f})")

result = model.forward("logic reason", return_entanglement=True)
if result.get('entanglement'):
    ent = result['entanglement']
    print("\nEntanglement:")
    for k, v in ent.von_neumann.items():
        print(f"  {k}: {v:.3f}")