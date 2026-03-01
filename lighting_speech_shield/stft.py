"""
STFT 处理模块

功能:
- 多通道语音信号的 STFT/ISTFT 变换
- 复数频谱处理
- 流式处理支持
"""

import torch
import numpy as np
import torchaudio


class STFTProcessor:
    """
    STFT 处理器
    
    参数:
        sample_rate: 采样率 (默认 16000)
        n_fft: FFT 大小 (默认 512)
        hop_length: Hop 长度 (默认 160, 对应 10ms)
        window: 窗函数 (默认 hann)
    """
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, window='hann'):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        
        # 创建 STFT 变换
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window_fn=torch.hann_window,
            power=None,  # 返回复数
            normalized=False,
            center=True,
            pad_mode='reflect'
        )
        
        # ISTFT
        self.istft = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window_fn=torch.hann_window,
            n_iter=32,
            normalized=False,
            center=True,
            pad_mode='reflect'
        )
    
    def forward(self, audio, return_complex=True):
        """
        音频 → STFT
        
        Args:
            audio: (B, C, T) - 多通道音频
            return_complex: 是否返回 (B, C, T, F, 2) (实/虚) 格式
        
        Returns:
            spec: (B, C, T, F, 2) 或 (B, C, F, T) 复数频谱
        """
        B, C, T = audio.shape
        
        # 对每个通道做 STFT
        specs = []
        for c in range(C):
            spec = self.stft(audio[:, c, :])  # (B, F, T)
            specs.append(spec)
        
        # 合并通道：(B, C, F, T)
        spec = torch.stack(specs, dim=1)
        
        if return_complex:
            # 转换为 (B, C, T, F, 2) (实部，虚部) 格式
            spec_real = spec.real.unsqueeze(-1)
            spec_imag = spec.imag.unsqueeze(-1)
            spec = torch.cat([spec_real, spec_imag], dim=-1)  # (B, C, T, F, 2)
            spec = spec.permute(0, 2, 3, 1, 4)  # (B, T, F, C, 2)
        
        return spec
    
    def inverse(self, spec, target_audio=None):
        """
        STFT → 音频 (使用 Griffin-Lim)
        
        Args:
            spec: 频谱 (B, C, F, T) 或 mask 处理后的频谱
            target_audio: 原始音频用于相位重建
        
        Returns:
            audio: (B, 1, T) 降噪后的音频
        """
        # 简化：假设输入是幅度谱，用原始相位
        if spec.dim() == 5:
            # (B, T, F, C, 2) → (B, C, F, T)
            spec = spec.permute(0, 3, 2, 1, 4)
            real = spec[..., 0]
            imag = spec[..., 1]
            spec_complex = torch.complex(real, imag)
            spec_mag = torch.abs(spec_complex)
        else:
            spec_mag = spec
        
        # 使用目标音频的相位（如果有）
        if target_audio is not None:
            target_spec = self.stft(target_audio[:, 0, :])
            phase = torch.angle(target_spec)
            spec_reconstructed = spec_mag * torch.exp(1j * phase)
        else:
            # 没有相位信息，用 Griffin-Lim
            return self.istft(spec_mag)
        
        # ISTFT
        audio = torch.istft(
            spec_reconstructed,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=spec_mag.device),
            center=True
        )
        
        return audio.unsqueeze(1)  # (B, 1, T)


def apply_mask(spec, mask, ref_channel=0):
    """
    应用 mask 到参考通道
    
    Args:
        spec: (B, T, F, C, 2) 复数频谱
        mask: (B, F, 1) mask
        ref_channel: 参考通道索引
    
    Returns:
        denoised_spec: 降噪后的频谱
    """
    # mask shape: (B, F, 1) → (B, 1, F, 1, 1)
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    
    # 应用到参考通道
    spec_denoised = spec.clone()
    spec_denoised[:, :, :, ref_channel:ref_channel+1, :] *= mask_expanded
    
    return spec_denoised


if __name__ == "__main__":
    # 测试 STFT 处理
    processor = STFTProcessor(sample_rate=16000, n_fft=512, hop_length=160)
    
    # 模拟 3 通道输入
    audio = torch.randn(1, 3, 16000)  # 1 秒
    
    # STFT
    spec = processor.forward(audio, return_complex=True)
    print(f"Input: {audio.shape}")
    print(f"STFT: {spec.shape}")
    
    # 模拟 mask
    B, T, F, C, _ = spec.shape
    fake_mask = torch.zeros(B, F, 1)  # 全 0 mask
    
    # 应用 mask
    denoised = apply_mask(spec, fake_mask)
    
    # ISTFT
    output = processor.inverse(denoised, target_audio=audio)
    print(f"Output: {output.shape}")
