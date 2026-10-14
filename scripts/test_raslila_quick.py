"""Test Raslila training."""
import sys, time, os
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
from raslila.raslila import Raslila, RaslilaConfig
import torch

config = RaslilaConfig(
    dim_x=3, dim_y=3, dim_z=2,
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

print("Training 5 steps...")
start = time.perf_counter()
for step in range(5):
    for prompt, target in pairs:
        r = model.train_step(prompt, target, lr_mlp=0.03)
    a_val = model.alpha.item()
    b_val = model.beta.item()
    print("Step {}: loss={:.4f} a={:+.3f} b={:+.3f}".format(step, r['loss_after'], a_val, b_val))

print("\nDone in {:.1f}s".format(time.perf_counter()-start))
print("Final: a={:+.3f}, b={:+.3f}".format(model.alpha.item(), model.beta.item()))

for p in ["grief sorrow", "logic reason", "dream sleep", "love kindness"]:
    r = model.forward(p)
    print("  '{}' -> '{}' (conf: {:.3f})".format(p, r['token'], r['confidence']))

result = model.forward("logic reason", return_entanglement=True)
if result.get('entanglement'):
    ent = result['entanglement']
    print("\nEntanglement:")
    for k, v in ent.von_neumann.items():
        print("  {}: {:.3f}".format(k, v))