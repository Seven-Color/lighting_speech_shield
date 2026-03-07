"""
训练脚本 - v2 100帧版本 (优化版)

优化:
- 混合精度训练 (AMP)
- 梯度累积 (支持大 effective batch)
- Cosine Annealing 学习率调度
- 更好的日志输出
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from lighting_speech_shield.model_v2 import ComplexMaskNet
from lighting_speech_shield.dataset import SpeechNoiseDataset

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


def complex_mse_loss(pred, target):
    return nn.MSELoss()(pred[..., 0], target[..., 0]) + nn.MSELoss()(pred[..., 1], target[..., 1])


def train_model(data_dir, epochs=10, batch_size=2, num_frames=100, save_dir='checkpoints',
                use_amp=True, gradient_accumulation=4, base_lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"AMP enabled: {use_amp and AMP_AVAILABLE and device.type == 'cuda'}")
    
    dataset = SpeechNoiseDataset(data_dir, num_frames=num_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} samples, Frames: {num_frames}")
    print(f"Batch size: {batch_size}, Accumulation: {gradient_accumulation}, Effective: {batch_size * gradient_accumulation}")
    
    model = ComplexMaskNet(base_channels=12, use_attention=True).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    
    # Cosine Annealing with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # Mixed precision scaler
    scaler = GradScaler() if (use_amp and AMP_AVAILABLE and device.type == 'cuda') else None
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    accumulation_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for input_spec, target_mask in pbar:
            input_spec = input_spec.to(device)
            target_mask = target_mask.to(device)
            
            # Forward with optional AMP
            if scaler is not None:
                with autocast():
                    pred_mask = model(input_spec)
                    loss = complex_mse_loss(pred_mask, target_mask)
                    loss = loss / gradient_accumulation
                
                scaler.scale(loss).backward()
                accumulation_counter += 1
                
                if accumulation_counter >= gradient_accumulation:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    accumulation_counter = 0
            else:
                pred_mask = model(input_spec)
                loss = complex_mse_loss(pred_mask, target_mask)
                loss = loss / gradient_accumulation
                loss.backward()
                accumulation_counter += 1
                
                if accumulation_counter >= gradient_accumulation:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulation_counter = 0
            
            total_loss += loss.item() * gradient_accumulation
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation:.4f}', 'lr': f'{current_lr:.6f}'})
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path / 'best_model_v2.pth')
            print(f"  [OK] Saved best model")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Lighting Speech Shield v2')
    parser.add_argument('--data_dir', type=str, default='data/synthetic', help='Data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    parser.add_argument('--gradient_accumulation', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        save_dir=args.save_dir,
        use_amp=not args.no_amp,
        gradient_accumulation=args.gradient_accumulation,
        base_lr=args.lr
    )
