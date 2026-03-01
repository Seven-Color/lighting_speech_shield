"""
训练脚本

训练 Lighting Speech Shield 模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from lighting_speech_shield.model import LightingSpeechShield
from lighting_speech_shield.stft import STFTProcessor, apply_mask


class SpeechNoiseDataset(Dataset):
    """语音降噪数据集"""
    
    def __init__(self, data_dir, n_fft=512, hop_length=160):
        self.data_dir = Path(data_dir)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft_processor = STFTProcessor(16000, n_fft, hop_length)
        
        # 加载元数据
        import json
        with open(self.data_dir / 'metadata.json') as f:
            self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        data = np.load(self.data_dir / item['filename'])
        
        clean = torch.from_numpy(data['clean']).float()  # (C, T)
        noisy = torch.from_numpy(data['noisy']).float()  # (C, T)
        
        # STFT 变换
        clean_spec = self.stft_processor.forward(clean.unsqueeze(0), return_complex=True)  # (1, T, F, C, 2)
        noisy_spec = self.stft_processor.forward(noisy.unsqueeze(0), return_complex=True)
        
        # 计算 clean mask（理想二值 mask）
        clean_mag = torch.abs(clean_spec[..., 0] + 1j * clean_spec[..., 1])
        noisy_mag = torch.abs(noisy_spec[..., 0] + 1j * noisy_spec[..., 1])
        
        # IBM: 如果 clean > noisy，则为 1，否则为 0
        mask = (clean_mag > noisy_mag).float()  # (1, T, F, C, 1)
        
        # 只取参考通道的 mask
        mask = mask[:, :, :, 0:1, :]  # (1, T, F, 1, 1)
        mask = mask[:, :, :, 0, 0]  # (1, T, F)
        mask = mask.mean(dim=1, keepdim=False)  # 平均时间维度 → (1, F)
        mask = mask.transpose(0, 1)  # (F, 1)
        
        # 模型输入：取 3 帧（当前 + 未来 2 帧）
        # 简化：复制当前帧 3 次
        input_spec = noisy_spec[:, 0:1, :, :, :].repeat(1, 3, 1, 1, 1)  # (1, 3, F, C, 2)
        input_spec = input_spec.permute(0, 2, 1, 3, 4)  # (1, F, 3, C, 2)
        
        return input_spec.squeeze(0), mask  # ((F, 3, C, 2), (F, 1))


def train_model(data_dir, epochs=50, batch_size=16, lr=0.001, save_dir='checkpoints'):
    """训练模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # 数据集
    dataset = SpeechNoiseDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 模型
    model = LightingSpeechShield(
        input_channels=6,
        base_channels=32,
        num_layers=4,
        num_heads=4,
        future_frames=2
    ).to(device)
    
    # 损失函数：MSE + 频谱约束
    mse_loss = nn.MSELoss()
    
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
            input_spec = input_spec.to(device)  # (B, F, 3, C, 2)
            target_mask = target_mask.to(device)  # (B, F, 1)
            
            # Forward
            pred_mask = model(input_spec)  # (B, F, 1)
            
            # Loss
            loss = mse_loss(pred_mask, target_mask)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_file = save_path / f'best_model_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_file)
            print(f"  ✅ 保存最佳模型：{save_file} (loss={avg_loss:.4f})")
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_file = save_path / f'checkpoint_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_file)
    
    print(f"\n训练完成！最佳 loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    import sys
    
    # 默认参数
    data_dir = "data/synthetic"
    epochs = 50
    batch_size = 16
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    
    # 训练
    train_model(data_dir, epochs=epochs, batch_size=batch_size)
