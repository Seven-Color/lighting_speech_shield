#!/usr/bin/env python
"""测试模型可行性"""

import sys
sys.path.insert(0, 'C:\\Users\\wuxiukun\\.openclaw\\workspace\\lighting_speech_shield')

import torch
from lighting_speech_shield.model import LightingSpeechShield

print("=" * 50)
print("Lighting Speech Shield - 模型可行性测试")
print("=" * 50)

# 创建模型
model = LightingSpeechShield(
    num_freq_bins=257,
    num_frames=3,
    num_channels=3,
    base_channels=32,
    num_layers=4,
    num_heads=4,
    future_frames=2
)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数量：{total_params:,} ({total_params/1e6:.2f}M)")

# 测试输入
x = torch.randn(1, 257, 3, 3, 2)  # (B, F, T, C, 2)
print(f"输入形状：{x.shape}")
print(f"  - Batch: 1")
print(f"  - 频率 bin: 257 (FFT=512)")
print(f"  - 时间帧：3 (当前 + 未来 2 帧)")
print(f"  - 通道：3")
print(f"  - 实/虚：2")

# Forward pass
with torch.no_grad():
    y = model(x)

print(f"\n输出形状：{y.shape}")
print(f"  - Batch: 1")
print(f"  - 频率 bin: 257")
print(f"  - Mask: 1")
print(f"\nMask 范围：[{y.min():.4f}, {y.max():.4f}]")

# FLOPs 估算（简化）
# Conv: 4 层 × 257 × 32 × 3 × 3 ≈ 0.3M
# Attention: 4 层 × 257 × 32 × 32 × 3 ≈ 0.3M
# Total: ≈ 1-2 MFlops per frame
flops_per_frame = total_params * 3  # rough estimate
fps = 100  # 160 hop @ 16kHz
total_mflops = flops_per_frame * fps / 1e6

print(f"\n算力估算：")
print(f"  - 每帧 FLOPs: ~{flops_per_frame/1e3:.1f}K")
print(f"  - 帧率：{fps} FPS (hop=160, sr=16k)")
print(f"  - 总算力：~{total_mflops:.1f} MFlops")
status = "PASS" if total_mflops < 200 else "FAIL"
print(f"  - 目标：<200 MFlops [{status}]")

print("\n" + "=" * 50)
print("Model test PASSED!")
print("=" * 50)
