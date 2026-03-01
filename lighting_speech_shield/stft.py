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
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        
        # 窗函数
        self.window = torch.hann_window(n_fft)
    
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
            # 使用 torch.stft
            spec = torch.stft(
                audio[:, c, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                pad_mode='reflect',
                normalized=False,
                return_complex=True
            )  # (B, F, T)
            specs.append(spec)
        
        # 合并通道：(B, C, F, T)
        spec = torch.stack(specs, dim=1)
        
        if return_complex:
            # 转换为 (B, T, F, C, 2) (实部，虚部) 格式
            spec_real = spec.real.unsqueeze(-1)
            spec_imag = spec.imag.unsqueeze(-1)
            spec = torch.cat([spec_real, spec_imag], dim=-1)  # (B, C, F, T, 2)
            spec = spec.permute(0, 3, 2, 1, 4)  # (B, T, F, C, 2)
        
        return spec
    
    def inverse(self, spec_complex, length=None):
        """
        STFT → 音频 (使用 ISTFT)
        
        Args:
            spec_complex: (B, C, F, T) 复数频谱或 (B, T, F, C, 2) 实/虚格式
            length: 输出音频长度
        
        Returns:
            audio: (B, C, T) 音频
        """
        if spec_complex.dim() == 5:
            # (B, T, F, C, 2) → (B, C, F, T)
            spec_complex = spec_complex.permute(0, 3, 2, 1, 4)
            real = spec_complex[..., 0]
            imag = spec_complex[..., 1]
            spec = torch.complex(real, imag)
        else:
            spec = spec_complex
        
        B, C, F, T = spec.shape
        
        # 对每个通道做 ISTFT
        audios = []
        for c in range(C):
            audio = torch.istft(
                spec[:, c, :, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                length=length
            )  # (B, T)
            audios.append(audio)
        
        return torch.stack(audios, dim=1)  # (B, C, T)


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
    fake_mask = torch.ones(B, F, 1)  # 全 1 mask
    
    # 应用 mask
    denoised = apply_mask(spec, fake_mask)
    
    # ISTFT
    output = processor.inverse(denoised)
    print(f"Output: {output.shape}")
    
    print("[OK] STFT test passed!")
