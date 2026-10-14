"""Train Raslila from YAML config."""
import sys
import os
import math
import yaml
import argparse

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
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


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config_dict, device):
    """Create model from config dict."""
    model_cfg = config_dict['model']
    ham_cfg = config_dict['hamiltonian']
    emb_cfg = config_dict['embeddings']
    mlp_cfg = config_dict['mlp']
    readout_cfg = config_dict['readout']
    
    config = GPUHamConfig(
        n_qubits=model_cfg['n_qubits'],
        n_universes=model_cfg['n_universes'],
        n_layers=model_cfg['n_layers'],
        vocab_size=emb_cfg['vocab_size'],
        max_seq_len=emb_cfg['max_seq_len'],
        embed_dim=emb_cfg['embed_dim'],
        mlp_hidden_dim=config_dict['mlp']['hidden_dim'],
        normalization_method=ham_cfg['normalization_method'],
        krylov_dim=ham_cfg['krylov_dim'],
        spectral_norm=ham_cfg['spectral_norm'],
        interference_qubits=tuple(ham_cfg['interference_qubits']),
    )
    
    model = RaslilaGPUModel(config)
    
    # Override token embedding dimension
    model.token_embedding = nn.Embedding(
        emb_cfg['vocab_size'], 
        emb_cfg['embed_dim'], 
        padding_idx=0
    ).to(next(model.parameters()).device)
    
    # Update token list from config
    model.tokens = readout_cfg['tokens']
    model.readout_qubits = tuple(readout_cfg['qubit_indices'])
    
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)

    backend = config.get('distributed', {}).get('backend', 'nccl')

    rank, world_size, local_rank = setup_distributed(backend)
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    is_main = rank == 0
    
    if is_main:
        print(f"Config loaded from {args.config}")
        print(f"Distributed backend: {backend}")
        print(f"World size: {world_size}, Rank: {rank}")
    
    # Create model
    model = create_model(config, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Wrap with DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # Data
    data_cfg = config['data']
    data_config = DataConfig(
        vocab_size=data_cfg['vocab_size'],
        max_seq_len=data_cfg['max_seq_len']
    )
    
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders_for_mve(
        data_config, 
        batch_size=data_cfg['train_batch_size']
    )
    
    train_sampler = DistributedSampler(
        train_loader.dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=data_cfg['train_batch_size'],
        sampler=train_sampler,
        num_workers=data_cfg['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_loader.dataset,
        batch_size=data_cfg['val_batch_size'],
        sampler=val_sampler,
        num_workers=data_cfg['num_workers'],
        pin_memory=True
    )
    
    # Optimizer
    train_cfg = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True
    )
    
    # Scheduler
    warmup = train_cfg['warmup_steps']
    total = train_cfg['max_steps']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min(1.0, step/warmup) * 
        (0.5 * (1 + math.cos(math.pi * max(0, step-warmup) / max(1, total-warmup))))
    )
    
    scaler = GradScaler()
    
    # Training loop
    accumulation = train_cfg['accumulation_steps']
    eval_every = train_cfg['eval_every']
    save_every = train_cfg['save_every']
    output_dir = train_cfg['output']['dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.train()
    step = 0
    best_ppl = float('inf')
    train_iter = iter(train_loader)
    
    if is_main:
        print(f"Starting training for {train_cfg['max_steps']} steps...")
        print(f"Global batch size: {data_cfg['train_batch_size'] * world_size * accumulation}")
    
    try:
        for step in range(train_cfg['max_steps']):
            model.train()
            
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
                shift_labels = target_ids[:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=0
                )
                loss = loss / accumulation
            
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # Logging
            if step % 50 == 0 and is_main:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step:5d}: loss={loss.item()*accumulation:.4f}, lr={current_lr:.2e}")
            
            # Evaluation
            if step % eval_every == 0 and step > 0:
                model.eval()
                val_losses = []
                val_iter = iter(val_loader)
                for _ in range(20):
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)
                    
                    val_input, val_target = val_batch[0].to(device), val_batch[1].to(device)
                    with torch.no_grad(), autocast(dtype=torch.bfloat16):
                        val_out = model(val_input)
                        val_logits = val_out['logits']
                        shift_logits = val_logits[:, :-1, :].contiguous()
                        shift_labels = val_batch[1][:, 1:].contiguous().to(device)
                        val_loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=0
                        )
                        val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                val_ppl = np.exp(avg_val_loss) if avg_val_loss < 100 else float('inf')
                
                if is_main:
                    print(f"\n  Step {step}: Val PPL={val_ppl:.2f}, Val Loss={avg_val_loss:.4f}")
                    
                    if val_ppl < best_ppl:
                        best_ppl = val_ppl
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'step': step,
                            'val_ppl': val_ppl,
                        }, os.path.join(output_dir, "best_model.pt"))
                        print(f"  Saved best model: PPL={val_ppl:.2f}")
                
                model.train()
            
            # Checkpointing
            if step % save_every == 0 and step > 0 and is_main:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': step,
                    'best_ppl': best_ppl,
                }, os.path.join(output_dir, f"step_{step}.pt"))
            
            step += 1
            
            if step >= train_cfg['max_steps']:
                break
    
    except KeyboardInterrupt:
        if is_main:
            print("\nInterrupted, saving checkpoint...")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': step,
            }, os.path.join(output_dir, f"interrupted_step_{step}.pt"))
    
    cleanup()


if __name__ == "__main__":
    main()