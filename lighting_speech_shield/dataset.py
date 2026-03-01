"""
数据集模块 - v2 100帧版本

输入: (B, F=257, T=100, CH=3, 2) -> 合并实虚部 -> (B, F=257, T=100, CH=6)
输出: (B, F=257, T=100, 2) - 复数mask
"""

import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset

from .stft import STFTProcessor


class SpeechNoiseDataset(Dataset):
    def __init__(self, data_dir, n_fft=512, hop_length=160, num_frames=100):
        self.data_dir = Path(data_dir)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.stft_processor = STFTProcessor(16000, n_fft, hop_length)
        
        with open(self.data_dir / 'metadata.json') as f:
            self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        data = np.load(self.data_dir / item['filename'])
        
        clean = torch.from_numpy(data['clean']).float()  # (C, T)
        noisy = torch.from_numpy(data['noisy']).float()  # (C, T)
        
        # STFT: (1, T, F, C, 2)
        clean_spec = self.stft_processor.forward(clean.unsqueeze(0), return_complex=True)
        noisy_spec = self.stft_processor.forward(noisy.unsqueeze(0), return_complex=True)
        
        # 扩展到目标帧数
        T_orig = clean_spec.shape[1]
        while clean_spec.shape[1] < self.num_frames:
            clean_spec = torch.cat([clean_spec, clean_spec], dim=1)
            noisy_spec = torch.cat([noisy_spec, noisy_spec], dim=1)
        
        # 裁剪
        clean_spec = clean_spec[:, :self.num_frames, :, :, :]
        noisy_spec = noisy_spec[:, :self.num_frames, :, :, :]
        
        # 参考通道 C=0
        noisy_c0 = noisy_spec[:, :, :, 0, :]  # (1, T, F, 2)
        clean_c0 = clean_spec[:, :, :, 0, :]  # (1, T, F, 2)
        
        # 计算复数mask
        noisy_mag = torch.sqrt(noisy_c0[..., 0]**2 + noisy_c0[..., 1]**2).clamp(min=1e-8)
        
        mask_real = clean_c0[..., 0] / noisy_mag
        mask_imag = clean_c0[..., 1] / noisy_mag
        
        # 限制幅度
        mask_mag = torch.sqrt(mask_real**2 + mask_imag**2).clamp(max=2.0)
        mask_angle = torch.atan2(mask_imag, mask_real)
        mask_real = mask_mag * torch.cos(mask_angle)
        mask_imag = mask_mag * torch.sin(mask_angle)
        
        # 合并实虚部: (1, T, F, 2) -> (F, T, 2)
        mask = torch.stack([mask_real.squeeze(0), mask_imag.squeeze(0)], dim=-1)  # (T, F, 2)
        mask = mask.permute(1, 0, 2)  # (F, T, 2)
        
        # 输入: (1, T, F, C, 2) -> 合并实虚部 -> (F, T, 6)
        noisy_input = noisy_spec.squeeze(0)  # (T, F, C, 2)
        noisy_input = noisy_input.permute(1, 0, 2, 3)  # (F, T, C, 2)
        noisy_input = noisy_input.reshape(257, 100, 6)  # (F, T, 6)
        
        return noisy_input, mask


if __name__ == "__main__":
    dataset = SpeechNoiseDataset("data/synthetic", num_frames=100)
    x, y = dataset[0]
    print(f"输入: {x.shape} (F,T,6)")
    print(f"Mask: {y.shape} (F,T,2)")
    print(f"Mask: [{y.min():.4f}, {y.max():.4f}]")
    print("[OK]")
