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
    """语音降噪数据集"""
    
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
        
        # STFT 变换
        clean_spec = self.stft_processor.forward(clean.unsqueeze(0), return_complex=True)  # (1, T, F, C, 2)
        noisy_spec = self.stft_processor.forward(noisy.unsqueeze(0), return_complex=True)
        
        # clean_spec: (1, T, F, C, 2)
        # 计算幅度谱
        clean_mag = torch.sqrt((clean_spec[..., 0] ** 2 + clean_spec[..., 1] ** 2))  # (1, T, F, C)
        noisy_mag = torch.sqrt((noisy_spec[..., 0] ** 2 + noisy_spec[..., 1] ** 2))
        
        # IBM: clean > noisy 则为 1
        mask = (clean_mag > noisy_mag).float()  # (1, T, F, C)
        
        # 只取参考通道 (第一个通道)
        mask = mask[:, :, :, 0]  # (1, T, F)
        
        # 对时间维度取平均
        mask = mask.mean(dim=1, keepdim=True)  # (1, 1, F)
        mask = mask.squeeze(1).transpose(0, 1)  # (F, 1)
        
        # 模型输入：取前 num_frames 帧
        input_spec = noisy_spec[:, :self.num_frames, :, :, :]  # (1, num_frames, F, C, 2)
        input_spec = input_spec.permute(0, 2, 1, 3, 4)  # (1, F, num_frames, C, 2)
        
        return input_spec.squeeze(0), mask  # ((F, num_frames, C, 2), (F, 1))


if __name__ == "__main__":
    # 测试数据集
    dataset = SpeechNoiseDataset("data/synthetic")
    print(f"Dataset size: {len(dataset)}")
    
    x, y = dataset[0]
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {y.shape}")
    print(f"Mask range: [{y.min():.4f}, {y.max():.4f}]")
    print("[OK] Dataset test passed!")
