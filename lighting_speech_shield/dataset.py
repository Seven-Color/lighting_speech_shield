"""
数据集模块

语音降噪数据集加载
"""

import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset

from .stft import STFTProcessor


class SpeechNoiseDataset(Dataset):
    """语音降噪数据集 - v2 复数 mask 版本"""
    
    def __init__(self, data_dir, n_fft=512, hop_length=160, num_frames=3):
        self.data_dir = Path(data_dir)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.stft_processor = STFTProcessor(16000, n_fft, hop_length)
        
        # 加载元数据
        with open(self.data_dir / 'metadata.json') as f:
            self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        data = np.load(self.data_dir / item['filename'])
        
        clean = torch.from_numpy(data['clean']).float()  # (C, T)
        noisy = torch.from_numpy(data['noisy']).float()  # (C, T)
        
        # STFT 变换: (1, T, F, C, 2)
        clean_spec = self.stft_processor.forward(clean.unsqueeze(0), return_complex=True)
        noisy_spec = self.stft_processor.forward(noisy.unsqueeze(0), return_complex=True)
        
        # clean_spec: (1, T, F, C, 2)
        
        # 计算复数 mask (只对参考通道)
        # 复数 mask = clean_spec / noisy_spec
        # 避免除零
        noisy_mag = torch.sqrt(noisy_spec[..., 0]**2 + noisy_spec[..., 1]**2).clamp(min=1e-8)
        
        # 复数 mask: (clean / |noisy|) * exp(1j * angle(noisy))
        mask_real = clean_spec[..., 0] / noisy_mag
        mask_imag = clean_spec[..., 1] / noisy_mag
        
        # 限制 mask 幅度在 [0, 2]
        mask_mag = torch.sqrt(mask_real**2 + mask_imag**2).clamp(max=2.0)
        mask_angle = torch.atan2(mask_imag, mask_real)
        
        mask_real = mask_mag * torch.cos(mask_angle)
        mask_imag = mask_mag * torch.sin(mask_angle)
        
        # Stack 成复数: (1, T, F, C, 2)
        complex_mask = torch.stack([mask_real, mask_imag], dim=-1)
        
        # 只取参考通道 (CH=0) 和目标帧数
        mask = complex_mask[:, :self.num_frames, :, 0:1, :]  # (1, T, F, 1, 2)
        mask = mask.squeeze(3).permute(0, 2, 1, 3)  # (1, F, T, 2)
        
        # 输入: noisy_spec 取目标帧
        input_spec = noisy_spec[:, :self.num_frames, :, :, :]  # (1, T, F, C, 2)
        input_spec = input_spec.permute(0, 2, 1, 3, 4)  # (1, F, T, C, 2)
        
        return input_spec.squeeze(0), mask.squeeze(0)  # ((F, T, C, 2), (F, T, 2))


if __name__ == "__main__":
    # 测试数据集
    dataset = SpeechNoiseDataset("data/synthetic", num_frames=3)
    print(f"Dataset size: {len(dataset)}")
    
    x, y = dataset[0]
    print(f"Input shape: {x.shape}")  # (F, T, C, 2) = (257, 3, 3, 2)
    print(f"Mask shape: {y.shape}")   # (F, T, 2) = (257, 3, 2)
    print(f"Mask range: [{y.min():.4f}, {y.max():.4f}]")
    print("[OK] Dataset test passed!")
