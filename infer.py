"""
推理脚本 - 支持离线批量和流式推理

用法:
    # 离线推理
    python infer.py --input test.wav --output denoised.wav --model checkpoints/best_model_v2.pth
    
    # 流式推理 (低延迟)
    python infer.py --input test.wav --output denoised.wav --streaming
"""

import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path

from lighting_speech_shield.model_v2 import ComplexMaskNet
from lighting_speech_shield.stft import STFTProcessor


class Denoiser:
    """降噪器 - 支持离线和流式推理"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stft = STFTProcessor(16000, 512, 160)
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model = ComplexMaskNet(base_channels=12, use_attention=True).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        print(f"Model loaded: {model_path}")
        print(f"Device: {self.device}")
    
    def denoise_offline(self, audio, num_frames=100):
        """
        离线批量降噪
        
        Args:
            audio: (C, T) tensor
            num_frames: 帧数
        
        Returns:
            denoised: (C, T) tensor
        """
        with torch.no_grad():
            # STFT
            audio = audio.unsqueeze(0).to(self.device)  # (1, C, T)
            spec = self.stft.forward(audio, return_complex=True)  # (1, T, F, C, 2)
            
            # 填充到目标帧数
            T = spec.shape[1]
            if T < num_frames:
                repeat = (num_frames // T) + 1
                spec = spec.repeat(1, repeat, 1, 1, 1)
            spec = spec[:, :num_frames, :, :, :]
            
            # 准备输入
            noisy_input = spec.squeeze(0).permute(1, 0, 2, 3).reshape(257, num_frames, 6)
            
            # 推理
            mask = self.model(noisy_input.unsqueeze(0).to(self.device))  # (1, F, T, 2)
            
            # 应用 mask 到参考通道
            ref_ch = 0
            noisy_ref = spec[:, :, :, ref_ch, :]  # (1, T, F, 2)
            
            # 复数乘法
            mask_real = mask[..., 0]  # (1, F, T)
            mask_imag = mask[..., 1]
            
            noisy_real = noisy_ref[..., 0].permute(0, 2, 1)  # (1, F, T)
            noisy_imag = noisy_ref[..., 1].permute(0, 2, 1)
            
            clean_real = mask_real * noisy_real
            clean_imag = mask_imag * noisy_imag
            
            # 重建频谱
            clean_spec = torch.stack([clean_real, clean_imag], dim=-1)  # (1, F, T, 2)
            clean_spec = clean_spec.permute(0, 2, 1, 3)  # (1, T, F, 2)
            
            # ISTFT
            spec_complex = torch.complex(clean_spec[..., 0], clean_spec[..., 1])
            
            # 截取原始长度
            if T > spec.shape[1]:
                spec_complex = spec_complex[:, :T, :]
            
            audio_denoised = self.stft.inverse(spec_complex, length=audio.shape[2])
            
            return audio_denoised.squeeze(0).cpu()
    
    def denoise_streaming(self, audio_chunk):
        """
        流式降噪 - 低延迟
        
        Args:
            audio_chunk: (C, T) 音频块
        
        Returns:
            denoised_chunk: (C, T) 降噪后的块
        """
        with torch.no_grad():
            audio_chunk = audio_chunk.unsqueeze(0).to(self.device)
            spec = self.stft.forward(audio_chunk, return_complex=True)
            
            # 取参考通道
            ref_spec = spec[:, :, :, 0, :]  # (1, T, F, 2)
            
            # 输入准备
            noisy_input = ref_spec.permute(2, 1, 0, 3).reshape(257, -1, 6)
            
            # 推理
            mask = self.model(noisy_input.unsqueeze(0))
            
            # 应用 mask
            noisy_real = ref_spec[..., 0].permute(0, 2, 1)
            noisy_imag = ref_spec[..., 1].permute(0, 2, 1)
            
            clean_real = mask[..., 0] * noisy_real
            clean_imag = mask[..., 1] * noisy_imag
            
            # ISTFT
            clean_spec = torch.stack([clean_real, clean_imag], dim=-1).permute(0, 2, 1, 3)
            spec_complex = torch.complex(clean_spec[..., 0], clean_spec[..., 1])
            
            audio_denoised = self.stft.inverse(spec_complex)
            
            return audio_denoised.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser(description='Inference for Lighting Speech Shield')
    parser.add_argument('--input', type=str, required=True, help='Input audio file')
    parser.add_argument('--output', type=str, required=True, help='Output audio file')
    parser.add_argument('--model', type=str, default='checkpoints/best_model_v2.pth', help='Model checkpoint')
    parser.add_argument('--streaming', action='store_true', help='Use streaming mode')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    
    args = parser.parse_args()
    
    # 加载音频
    print(f"Loading: {args.input}")
    waveform, sr = torchaudio.load(args.input)
    
    if sr != args.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, args.sample_rate)
    
    # 多通道处理
    if waveform.shape[0] > 1:
        # 平均为单通道
        waveform = waveform.mean(dim=0, keepdim=True)
    else:
        waveform = waveform  # (1, T)
    
    # 复制为 3 通道 (模拟麦克风阵列)
    waveform_3ch = waveform.repeat(3, 1)  # (3, T)
    
    # 初始化降噪器
    denoiser = Denoiser(args.model)
    
    if args.streaming:
        print("Using streaming mode...")
        # 流式处理
        chunk_size = 1600  # 100ms
        results = []
        
        for i in range(0, waveform_3ch.shape[1], chunk_size):
            chunk = waveform_3ch[:, i:i+chunk_size]
            if chunk.shape[1] < 160:
                continue
            denoised = denoiser.denoise_streaming(chunk)
            results.append(denoised)
        
        waveform_denoised = torch.cat(results, dim=1)
    else:
        print("Using offline mode...")
        waveform_denoised = denoiser.denoise_offline(waveform_3ch)
    
    # 保存
    torchaudio.save(args.output, waveform_denoised.cpu(), args.sample_rate)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
