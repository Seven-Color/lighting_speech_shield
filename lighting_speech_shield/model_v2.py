"""
Lighting Speech Shield v2 - Conv3D 复数 Mask 版本

输入: (B, F=257, T=3-10, CH=3, 2) 
输出: (B, F=257, T=3-10, 2) 复数 mask
使用 Conv3D 处理时频域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x / rms * self.weight


class ComplexMaskNet(nn.Module):
    """Conv3D 复数 Mask 网络"""
    def __init__(self, num_freq=257, num_channels=3, base_channels=16):
        super().__init__()
        
        # 输入: (B, F, T, CH, 2) = (B, 257, T, 3, 2)
        # Conv3D 需要: (B, C, D, H, W) = (B, 2, T, F, CH)
        
        # 复数卷积 (2通道作为输入通道)
        self.conv3d = nn.Sequential(
            nn.Conv3d(2, base_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels),
            nn.GELU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels),
            nn.GELU(),
        )
        
        # 中间层 (减少复杂度)
        self.mid = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels),
            nn.GELU(),
        )
        
        # 上采样 (简化)
        self.up = nn.Identity()
        
        # 输出头：复数 mask (减少复杂度)
        self.mask_head = nn.Sequential(
            nn.Conv3d(base_channels, 16, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(16, 2, kernel_size=(1, 1, 1)),  # 输出 2 通道（实部+虚部）
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: (B, F, T, CH, 2)
        返回: (B, F, T, 2)
        """
        B, F, T, CH, _ = x.shape
        
        # 维度重排: (B, F, T, CH, 2) -> (B, 2, T, F, CH)
        x = x.permute(0, 4, 2, 1, 3).contiguous()
        
        # Conv3D
        x = self.conv3d(x)
        x = self.mid(x)
        x = self.up(x)
        
        # Mask 输出
        mask = self.mask_head(x)  # (B, 2, T, F, CH)
        
        # 维度恢复: (B, 2, T, F, CH) -> (B, F, T, 2)
        # 只取第一个通道 (CH=0)
        mask = mask[:, :, :, :, 0]  # (B, 2, T, F)
        mask = mask.permute(0, 3, 2, 1)  # (B, F, T, 2)
        
        return mask


def estimate_flops(input_shape=(1, 257, 3, 3, 2)):
    """估算 FLOPs"""
    B, F, T, CH, _ = input_shape
    C = 16  # base_channels
    
    flops = 0
    
    # Conv3D: C_in * C_out * D * H * W * output_size
    # conv3d 1: (2->16) * 3*3*3 * B*T*F*CH
    flops += 2 * 16 * 3 * 3 * 3 * B * T * F * CH
    # conv3d 2: (16->16) * 3*3*3 * B*T*F*CH
    flops += 16 * 16 * 3 * 3 * 3 * B * T * F * CH
    # mid: (16->16) * 3*3*3 * B*T*F*CH
    flops += 16 * 16 * 3 * 3 * 3 * B * T * F * CH
    
    # mask head
    flops += 16 * 16 * 1 * 1 * 1 * B * T * F * CH
    flops += 16 * 2 * 1 * 1 * 1 * B * T * F * CH
    
    return flops / 1e6


def model_info(input_shape=(1, 257, 3, 3, 2)):
    model = ComplexMaskNet()
    params = sum(p.numel() for p in model.parameters())
    mflops = estimate_flops(input_shape)
    
    x = torch.randn(*input_shape)
    with torch.no_grad():
        y = model(x)
    
    print(f"\n{'='*50}")
    print(f"Lighting Speech Shield v2 - Conv3D Complex Mask")
    print(f"{'='*50}")
    print(f"输入：{input_shape}")
    print(f"输出：{y.shape}")
    print(f"参数量：{params:,} ({params/1e6:.2f}M)")
    print(f"FLOPs (3帧): {mflops:.1f} MFlops")
    print(f"目标：<200 MFlops [{'PASS' if mflops < 200 else 'FAIL'}]")
    print(f"{'='*50}\n")
    
    return params, mflops


if __name__ == "__main__":
    print("Testing Lighting Speech Shield v2 - Conv3D...")
    
    # 测试 3 帧
    params, mflops = model_info((1, 257, 3, 3, 2))
    
    # 测试 10 帧
    x = torch.randn(1, 257, 10, 3, 2)
    model = ComplexMaskNet()
    with torch.no_grad():
        mask = model(x)
    
    print(f"输入 10 帧: {x.shape}")
    print(f"输出 Mask: {mask.shape}")
    print(f"Mask 范围: [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"\n[OK] Test passed!")
