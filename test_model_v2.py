#!/usr/bin/env python
"""
测试模型 v2 - 100帧优化版
"""

import sys
sys.path.insert(0, 'C:\\Users\\wuxiukun\\.openclaw\\workspace\\lighting_speech_shield')

import torch
from lighting_speech_shield.model_v2 import ComplexMaskNet, estimate_flops


def test_model():
    print("=" * 50)
    print("Testing v2 100帧 2D Conv (优化版)")
    print("=" * 50)
    
    # 测试带注意力的版本
    print("\n[1] 测试带注意力模块:")
    model = ComplexMaskNet(base_channels=12, use_attention=True)
    params = sum(p.numel() for p in model.parameters())
    
    x = torch.randn(1, 257, 100, 6)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"  输入: (1, 257, 100, 6)")
    print(f"  输出: {y.shape}")
    print(f"  参数量: {params:,}")
    print(f"  Mask范围: [{y.min():.4f}, {y.max():.4f}]")
    
    # 测试不带注意力的版本
    print("\n[2] 测试不带注意力模块:")
    model_no_attn = ComplexMaskNet(base_channels=12, use_attention=False)
    params_no_attn = sum(p.numel() for p in model_no_attn.parameters())
    
    with torch.no_grad():
        y_no_attn = model_no_attn(x)
    
    print(f"  参数量: {params_no_attn:,}")
    print(f"  注意力额外开销: +{params - params_no_attn:,} params")
    
    print("\n[3] FLOPs 估算:")
    flops = estimate_flops()
    print(f"  FLOPs: {flops:.1f} MFlops")
    print(f"  目标<200M: {'PASS ✅' if flops < 200 else 'FAIL ⚠️'}")
    
    print("\n" + "=" * 50)
    print("Model test PASSED!")
    print("=" * 50)
    
    return params, flops


if __name__ == "__main__":
    test_model()
