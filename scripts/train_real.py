"""Train on simple real text - no HF download needed."""
import sys
import time
import os

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
from raslila import EnhancedRaslilaModel, ModelConfig

print("=" * 60)
print("TRAIN ON CURATED REAL TEXT")
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

# Real text samples from classic literature
pairs = [
    # Grief
    ("Alice was beginning to get very tired of sitting by her sister on the bank", "grief"),
    ("She had peeped into the book her sister was reading but it had no pictures", "grief"),
    ("Once or twice she had peeped into the book her sister was reading", "grief"),
    ("So she was considering in her own mind", "grief"),
    ("Suddenly a White Rabbit with pink eyes ran close by her", "grief"),
    
    # Logic
    ("Sherlock Holmes sat in his armchair with his eyes closed", "logic"),
    ("He was thinking about the case of the missing jewel", "logic"),
    ("The game is afoot Watson he said", "logic"),
    ("Elementary my dear Watson the butler did it", "logic"),
    ("True false true false true false", "logic"),
    
    # Dream
    ("In a hole in the ground there lived a hobbit", "dream"),
    ("Not a nasty dirty wet hole filled with the ends of worms", "dream"),
    ("It was a hobbit hole and that means comfort", "dream"),
    ("Call me Ishmael some years ago never mind how long precisely", "dream"),
    ("I thought I would sail about a little and see the watery part", "dream"),
    
    # Empathy
    ("It is a truth universally acknowledged that a single man", "empathy"),
    ("In possession of a good fortune must be in want of a wife", "empathy"),
    ("Love kindness compassion care understanding", "empathy"),
    ("Empathy empathy empathy empathy", "empathy"),
    ("Kind heart understanding love", "empathy"),
    
    # Memory
    ("All children except one grow up", "memory"),
    ("They soon know that they will grow up", "memory"),
    ("Memory recall remember past yesterday history", "memory"),
    ("Time heals grief time heals", "memory"),
    ("Past history memory recall", "memory"),
    
    # Clarity
    ("Clear think focus understand clarity", "clarity"),
    ("Focus mind clear mind clear", "clarity"),
    ("Clear think focus understand", "clarity"),
    ("Logic reason deduction clarity", "clarity"),
    ("Focus mind clear mind", "clarity"),
    
    # Longing
    ("Hope dream wish desire longing", "longing"),
    ("Longing yearning desire wish", "longing"),
    ("Hope dream wish desire", "longing"),
    
    # Logic
    ("Logic reason deduction proof", "logic"),
    ("If then therefore implies logic", "logic"),
    ("True false true false logic", "logic"),
    ("Reason think analyze deduce", "logic"),
    ("Think reason understand logic", "logic"),
]

print(f"Training pairs: {len(pairs)} (balanced across 8 classes)")

# Train with a simple decaying LR schedule
lr_schedule = [(0.05, 3), (0.03, 3), (0.02, 3), (0.01, 3), (0.005, 3), (0.002, 3), (0.001, 3)]

print("\nTraining...")
start = time.perf_counter()
for lr, epochs in lr_schedule:
    for _epoch in range(epochs):
        epoch_losses = [model.train_step(p, t, lr_mlp=lr)['loss_after'] for p, t in pairs]
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  LR={lr:.4f} Epoch: loss={avg_loss:.4f} a={model.ham.alpha:+.3f} b={model.ham.beta:+.3f}")

print(f"\nFinal: a={model.ham.alpha:+.3f}, b={model.ham.beta:+.3f}")

# Test on held-out prompts
print("\n--- Evaluation ---")
test_prompts = [
    "grief sorrow loss pain",
    "logic reason deduction proof",
    "dream sleep vision future",
    "love kindness compassion",
    "memory recall remember past",
    "clear think focus understand",
]
for p in test_prompts:
    r = model.forward(p)
    print(f"  '{p}' -> '{r['token']}' (conf: {r['confidence']:.3f})")

# Entanglement
result = model.forward("logic reason", return_entanglement=True)
if result.get('entanglement'):
    ent = result['entanglement']
    print("\nEntanglement:")
    for k, v in ent.von_neumann.items():
        print(f"  {k}: {v:.3f}")
    for (a,b), mi in ent.mutual_info.items():
        print(f"  {a}-{b}: {mi:.4f}")

print(f"\nParams: {model.count_parameters()['total_trainable']}")
print(f"Time: {time.perf_counter()-start:.1f}s")