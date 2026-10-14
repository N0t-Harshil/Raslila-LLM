"""Ultra-fast real data test."""
import sys, time, os
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
from raslila import EnhancedRaslilaModel, ModelConfig

print("=" * 60)
print("MICRO REAL DATA TEST")
print("=" * 60)

config = ModelConfig(
    n_layers=2,
    universes=['Logical','Emotional','Temporal'],
    normalization_method='spectral',
    use_enhanced_memory=True,
    use_parameter_shift=True,
    use_annealing=False,
    use_sparse_trotter=False,
    mlp_hidden_dim=4200,
)
model = EnhancedRaslilaModel(config)
print(f"Params: {model.count_parameters()['total_trainable']}")

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
    print(f"Step {step}: loss={r['loss_after']:.4f} a={model.ham.alpha:+.3f} b={model.ham.beta:+.3f}")

print(f"\nDone in {time.perf_counter()-start:.1f}s")
print(f"Final: a={model.ham.alpha:+.3f}, b={model.ham.beta:+.3f}")

for p in ["grief sorrow", "logic reason", "dream sleep", "love kindness"]:
    r = model.forward(p)
    print(f"  '{p}' -> '{r['token']}' (conf: {r['confidence']:.3f})")

result = model.forward("logic reason", return_entanglement=True)
if result.get('entanglement'):
    ent = result['entanglement']
    print("\nEntanglement:")
    for k, v in ent.von_neumann.items():
        print(f"  {k}: {v:.3f}")