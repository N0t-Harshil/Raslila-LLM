"""Generate text from a trained Raslila checkpoint (byte-level).

Usage:
    python scripts/generate_raslila.py --checkpoint checkpoints/raslila/raslila_step1000.pt \
        --prompt "Once upon a time" --tokens 200 --temperature 0.8
"""
import os
import sys
import argparse

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import torch
import torch.nn.functional as F
import yaml

from raslila.raslila import Raslila, RaslilaConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/raslila/raslila_step1000.pt')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tokenizer', type=str, default='data/tokenizer/raslila_bpe.json',
                        help='BPE tokenizer matching training; omit for byte-level')
    parser.add_argument('--window', type=int, default=None,
                        help='context window in tokens (default: training.seq_len from '
                             '--config, i.e. the length the model was actually trained '
                             'on; falls back to the architectural max_seq_len)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Byte-level output can contain any Unicode; force UTF-8 console output
    # (Windows cp1252 otherwise crashes on replacement chars).
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    with open(args.config, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    b3 = dict(raw['raslila'])
    # `raslila:` holds architecture only; `training.seq_len` holds the context
    # length the weights were actually fitted at. They differ, and using the
    # architectural maximum instead is silently harmful (see `window` below).
    train_cfg = raw.get('training') or {}
    # The checkpoint records the exact training config; prefer it so old
    # (different-vocab) checkpoints still load.
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'config' in ckpt:
        b3 = ckpt['config']
    else:
        # Old checkpoints: infer vocab from the saved embedding table.
        b3['vocab_size'] = ckpt['model']['token_embedding.weight'].shape[0]

    # BPE if the model is bigger than the byte vocab (257); otherwise byte-level.
    bpe = None
    if int(b3.get('vocab_size', 257)) > 257:
        from tokenizers import Tokenizer
        path = args.tokenizer
        if not os.path.exists(path):
            raise SystemExit('Model uses BPE vocab {} but tokenizer {} is missing. '
                             'Pass --tokenizer <path>.'.format(b3['vocab_size'], path))
        bpe = Tokenizer.from_file(path)

    def encode(text):
        if bpe is not None:
            return bpe.encode(text).ids
        return [b + 1 for b in text.encode('utf-8', errors='replace')]

    def decode(ids):
        if bpe is not None:
            return bpe.decode(list(ids))
        return bytes(t - 1 for t in ids).decode('utf-8', errors='replace')

    model_cfg = RaslilaConfig(**b3)
    model = Raslila(model_cfg)
    model.load_state_dict(ckpt['model'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    ids = encode(args.prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    # Generate at the context length the model was TRAINED at, not at the
    # architectural maximum. `max_seq_len` sizes the position machinery; it is
    # not a claim that the model was fitted at that length. Past the trained
    # window, RoPE is extrapolating to rotations never seen in training.
    # Measured on the 15M checkpoint (trained at seq_len 128): held-out loss
    # 5.06 @128 -> 5.32 @256 -> 5.47 @512 -> 5.52 @1024. That 0.46 nat penalty
    # buys nothing, so the default is the trained length.
    arch_max = max(int(model_cfg.max_seq_len), 1)
    requested = args.window or train_cfg.get('seq_len') or arch_max
    window = min(max(int(requested), 1), arch_max)
    if int(requested) > arch_max:
        print('Warning: window {} exceeds max_seq_len {}; clamped to {}.'.format(
            requested, arch_max, window))

    with torch.no_grad():
        for _ in range(args.tokens):
            out = model(input_ids[:, -window:])
            logits = out['logits']
            last = logits[0, -1, :]
            # Never sample the pad token (id 0) — it is the CE ignore_index.
            mask = torch.full_like(last, -1e9)
            mask[1:] = last[1:]
            probs = F.softmax(mask / args.temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(0)
            input_ids = torch.cat([input_ids, nxt.unsqueeze(0).unsqueeze(0)], dim=1)

    gen = decode(input_ids[0, len(ids):].tolist())
    print('\nPrompt: {!r}\n'.format(args.prompt))
    print('Generated ({} tokens):\n{}{}\n'.format(args.tokens, args.prompt, gen))


if __name__ == '__main__':
    main()