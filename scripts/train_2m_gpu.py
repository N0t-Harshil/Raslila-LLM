"""Train 2M parameter model on 4xH100."""
import sys
import os
import time
import math

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import numpy as np

from raslila.gpu_hamiltonian import GPUHamConfig, GPUHamiltonian, RaslilaGPUModel
from raslila.data import DataConfig, get_dataloaders_for_mve

def setup_distributed(backend='nccl'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend=backend, init_method='env://')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-qubits', type=int, default=20)
    parser.add_argument('--n-universes', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--mlp-hidden-dim', type=int, default=4096)
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--max-seq-len', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation-steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--output-dir', type=str, default='checkpoints/raslila_2m')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    is_main = rank == 0

    if is_main:
        print(f"World size: {world_size}, Rank: {rank}")
        os.makedirs(args.output_dir, exist_ok=True)

    # Model config for ~2M params
    # Token embeddings: 32000 * 512 = 16.4M (too large)
    # Use smaller embed_dim: 32000 * 64 = 2.04M
    embed_dim = 64
    
    config = GPUHamConfig(
        n_qubits=20,
        n_universes=4,
        n_layers=4,
        vocab_size=32000,
        max_seq_len=1024,
        embed_dim=embed_dim,
        mlp_hidden_dim=4096,
        normalization_method='spectral',
        use_enhanced_memory=True,
        use_parameter_shift=True,
        use_annealing=False,
        use_sparse_trotter=False,
    )
    
    model = RaslilaGPUModel(config).to(device)
    
    # Wrap with DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Model params: {param_count:,} ({param_count/1e6:.1f}M)")

    # Data
    data_config = DataConfig(vocab_size=32000, max_seq_len=1024)
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders_for_mve(
        DataConfig(vocab_size=32000, max_seq_len=1024), batch_size=2
    )
    
    train_sampler = DistributedSampler(train_loader.dataset, shuffle=True)
    val_sampler = DistributedSampler(val_loader, shuffle=False)
    
    train_loader = DataLoader(train_loader.dataset, batch_size=2, sampler=train_sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_loader.dataset, batch_size=2, sampler=val_sampler,
                            num_workers=4, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, fused=True)
    scaler = GradScaler()
    
    # Training loop
    model.train()
    step = 0
    best_ppl = float('inf')
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    for step in range(50000):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_loader.sampler.set_epoch(step // len(train_loader))
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch[0].to(device, non_blocking=True)
        target_ids = batch[1].to(device, non_blocking=True)
        
        with autocast(dtype=torch.bfloat16):
            out = model(input_ids)
            logits = out['logits']
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[1][:, 1:].contiguous().to(device)
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0
            )
            loss = loss / 8  # accumulation
        
        scaler.scale(loss).backward()
        
        if (step + 1) % 8 == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        if step % 50 == 0 and is_main:
            print(f"Step {step}: loss={loss.item()*8:.4f}")
        
        if step % 500 == 0 and step > 0:
            model.eval()
            val_losses = []
            for _ in range(20):
                try:
                    val_batch = next(val_iter)
                except:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)
                val_input, val_target = val_batch[0].to(device), val_batch[1].to(device)
                with torch.no_grad(), autocast(dtype=torch.bfloat16):
                    out = model(val_input)
                    shift_logits = out['logits'][:, :-1, :].contiguous()
                    shift_labels = val_target[:, 1:].contiguous()
                    val_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1), ignore_index=0
                    )
                    val_losses.append(val_loss.item())
            model.train()
            if is_main:
                avg_loss = np.mean(val_losses)
                print(f"Step {step}: val_loss={avg_loss:.4f}, ppl={np.exp(avg_loss):.2f}")
        
        step += 1

    cleanup()

if __name__ == "__main__":
    main()