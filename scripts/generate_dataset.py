"""
数据生成脚本

生成仿真数据集用于训练和验证：
- 语音：从免费语料库下载或使用 TTS
- 噪声：多种常见噪声类型
- 多通道：模拟麦克风阵列
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
import librosa


class SyntheticDataGenerator:
    """
    合成数据生成器
    
    生成多通道带噪语音数据
    """
    def __init__(self, sample_rate=16000, num_channels=3):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.fps = sample_rate // 160  # 160 hop → 100fps
        
    def generate_pure_speech(self, duration=1.0):
        """
        生成纯语音片段（用谐波模型模拟）
        
        简化方案：用正弦波 + 共振峰模拟元音
        """
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # 基频（模拟人声）
        f0 = 150 + np.random.uniform(-20, 20)  # 150Hz 左右
        
        # 共振峰（模拟元音）
        formants = {
            'a': [730, 1090, 2440],
            'e': [530, 1840, 2480],
            'i': [270, 2290, 3010],
            'o': [570, 840, 2410],
            'u': [300, 870, 2240]
        }
        
        f_type = np.random.choice(list(formants.keys()))
        f1, f2, f3 = formants[f_type]
        
        # 生成谐波
        speech = np.zeros_like(t)
        for harmonic in range(1, 15):
            amplitude = 1.0 / harmonic
            speech += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)
        
        # 添加共振峰
        for f_res in [f1, f2, f3]:
            bandwidth = f_res * 0.1
            envelope = np.exp(-bandwidth * t)
            speech += 0.3 * np.sin(2 * np.pi * f_res * t) * envelope
        
        # 幅度包络（模拟音节）
        envelope = np.exp(-5 * np.abs(t - duration/2))
        speech *= envelope
        
        # 归一化
        speech = speech / np.abs(speech).max() * 0.5
        
        return speech.astype(np.float32)
    
    def generate_noise(self, noise_type='babble', duration=1.0, snr_db=10):
        """
        生成噪声
        
        类型：
        - babble: 人声嘈杂
        - white: 白噪声
        - pink: 粉红噪声
        - street: 街道噪声（模拟）
        """
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        if noise_type == 'white':
            noise = np.random.randn(len(t))
        elif noise_type == 'pink':
            # 粉红噪声：1/f
            noise = np.random.randn(len(t))
            noise = np.cumsum(noise)
            noise = noise - noise.mean()
        elif noise_type == 'babble':
            # 多个人声叠加
            noise = np.zeros_like(t)
            for _ in range(5):
                f0 = 100 + np.random.uniform(50, 200)
                for harmonic in range(1, 10):
                    noise += np.random.uniform(0.1, 0.3) * np.sin(2 * np.pi * f0 * harmonic * t + np.random.uniform(0, 2*np.pi))
        elif noise_type == 'street':
            # 模拟街道：低频噪声 + 脉冲
            noise = 0.5 * np.random.randn(len(t))
            # 汽车 horn
            for _ in range(np.random.randint(1, 4)):
                pos = np.random.randint(0, len(t) - 1000)
                horn = np.sin(2 * np.pi * 400 * t[pos:pos+1000]) * np.exp(-t[pos:pos+1000] * 0.5)
                noise[pos:pos+1000] += horn
        
        # 归一化
        noise = noise / np.abs(noise).max() * 0.3
        
        return noise.astype(np.float32)
    
    def generate_multichannel(self, speech, noise, room_size=3.0):
        """
        生成多通道信号（模拟麦克风阵列）
        
        简单方案：不同通道的延迟 + 幅度衰减
        """
        channels = []
        
        for c in range(self.num_channels):
            # 通道延迟（模拟声波到达时间差）
            delay_samples = int(c * 2)  # 每个通道相差 2 个采样
            
            # 通道衰减
            atten = 1.0 - c * 0.1
            
            if delay_samples > 0:
                s_delayed = np.pad(speech[:-delay_samples], (delay_samples, 0), mode='constant')
            else:
                s_delayed = speech
            
            channel = s_delayed * atten + noise
            
            channels.append(channel)
        
        return np.stack(channels, axis=0)  # (C, T)
    
    def add_noise_to_speech(self, speech, noise, snr_db):
        """
        添加噪声到语音（指定 SNR）
        """
        speech_power = np.mean(speech ** 2)
        noise_power = np.mean(noise ** 2)
        
        # 计算目标噪声功率
        target_noise_power = speech_power / (10 ** (snr_db / 10))
        noise_scale = np.sqrt(target_noise_power / (noise_power + 1e-10))
        
        noisy = speech + noise * noise_scale
        return noisy.astype(np.float32)
    
    def generate_sample(self, duration=1.0, snr_range=(0, 15), noise_types=None):
        """
        生成单个样本
        
        Returns:
            clean: 纯语音 (C, T)
            noisy: 带噪语音 (C, T)
            snr: 实际 SNR
        """
        if noise_types is None:
            noise_types = ['white', 'pink', 'babble', 'street']
        
        # 生成语音
        clean_speech = self.generate_pure_speech(duration)
        
        # 生成噪声
        noise_type = np.random.choice(noise_types)
        noise = self.generate_noise(noise_type, duration)
        
        # 随机 SNR
        snr_db = np.random.uniform(*snr_range)
        
        # 混合
        noisy_speech = self.add_noise_to_speech(clean_speech, noise, snr_db)
        
        # 多通道
        clean_mc = self.generate_multichannel(clean_speech, np.zeros_like(clean_speech))
        noisy_mc = self.generate_multichannel(noisy_speech, noise)
        
        return {
            'clean': clean_mc,  # (C, T)
            'noisy': noisy_mc,  # (C, T)
            'snr_db': snr_db,
            'noise_type': noise_type,
            'duration': duration
        }


def generate_dataset(output_dir, num_samples=100, duration=1.0):
    """
    生成数据集并保存
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticDataGenerator(sample_rate=16000, num_channels=3)
    
    print(f"生成 {num_samples} 个样本...")
    
    metadata = []
    
    for i in range(num_samples):
        sample = generator.generate_sample(duration=duration)
        
        # 保存为 numpy
        filename = f"sample_{i:04d}.npz"
        filepath = output_dir / filename
        
        np.savez(filepath,
                 clean=sample['clean'],
                 noisy=sample['noisy'],
                 snr_db=sample['snr_db'],
                 noise_type=sample['noise_type'])
        
        metadata.append({
            'filename': filename,
            'snr_db': float(sample['snr_db']),
            'noise_type': sample['noise_type']
        })
        
        if (i + 1) % 20 == 0:
            print(f"  已生成 {i+1}/{num_samples} 个样本")
    
    # 保存元数据
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ 数据集已保存到 {output_dir}")
    print(f"   样本数：{num_samples}")
    print(f"   格式：{generator.num_channels}通道, {int(duration*generator.sample_rate)}采样点/样本")
    
    return output_dir


if __name__ == "__main__":
    import sys
    
    # 默认输出目录
    output_dir = "data/synthetic"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    # 生成 100 个样本
    generate_dataset(output_dir, num_samples=100, duration=1.0)
