"""
训练脚本 - v2 版本

训练 Lighting Speech Shield v2 (Conv3D 复数 Mask)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from lighting_speech_shield.model_v2 import ComplexMaskNet
from lighting_speech_shield.dataset import SpeechNoiseDataset


def complex_mse_loss(pred_mask, target_mask):
    """
    复数 MSE 损失
    pred_mask: (B, F, T, 2) 复数
    target_mask: (B, F, T, 2) 复数
    """
    # 分离实部虚部
    pred_real = pred_mask[..., 0]
    pred_imag = pred_mask[..., 1]
    target_real = target_mask[..., 0]
    target_imag = target_mask[..., 1]
    
    # MSE
    loss = nn.MSELoss()(pred_real, target_real) + nn.MSELoss()(pred_imag, target_imag)
    return loss


def train_model(data_dir, epochs=50, batch_size=8, lr=0.001, num_frames=3, save_dir='checkpoints'):
    """训练模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据集
    dataset = SpeechNoiseDataset(data_dir, num_frames=num_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} samples, Frames: {num_frames}")
    
    # 模型
    model = ComplexMaskNet(
        num_freq=257,
        num_channels=3,
        base_channels=16
    ).to(device)
    
    # 损失函数
    criterion = complex_mse_loss
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练循环
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for input_spec, target_mask in pbar:
            input_spec = input_spec.to(device)  # (B, F, T, C, 2)
            target_mask = target_mask.to(device)  # (B, F, T, 2)
            
            # Forward
            pred_mask = model(input_spec)  # (B, F, T, 2)
            
            # Loss
            loss = criterion(pred_mask, target_mask)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_file = save_path / f'best_model_v2.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_file)
            print(f"  [OK] Saved best model: {save_file}")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    import sys
    
    # 默认参数
    data_dir = "data/synthetic"
    epochs = 50
    batch_size = 8
    num_frames = 3
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    if len(sys.argv) > 3:
        num_frames = int(sys.argv[3])
    
    # 训练
    train_model(data_dir, epochs=epochs, batch_size=batch_size, num_frames=num_frames)
