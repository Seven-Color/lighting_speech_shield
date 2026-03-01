"""
Lighting Speech Shield v2 - 100帧 2D卷积版本

输入: (B, F=257, T=100, CH=3, 2) -> 合并实虚部 -> (B, F=257, T=100, CH=6)
输出: (B, F=257, T=100, 2) - 复数mask
100帧算力目标: <200MFlops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexMaskNet(nn.Module):
    """
    2D卷积网络
    输入: (B, F, T, 6) 合并了实虚部
    输出: (B, F, T, 2) 复数mask
    """
    def __init__(self, base_channels=12):
        super().__init__()
        
        # 输入: (B, F, T, 6) -> Conv2D需要 (B, C, H, W) = (B, 6, F, T)
        
        # 编码器 - 压缩
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),  # F/2, T/2
            nn.BatchNorm2d(base_channels*2),
            nn.GELU(),
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),  # F/4, T/4
            nn.BatchNorm2d(base_channels*4),
            nn.GELU(),
        )
        
        # 中间层
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.GELU(),
        )
        
        # 解码器 - 上采样
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.GELU(),
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        
        # 输出头
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, F, T, 6)
        B, F_dim, T_dim, C = x.shape
        
        # 维度重排: (B, F, T, C) -> (B, C, F, T)
        x = x.permute(0, 3, 1, 2)
        
        # 编码
        e1 = self.enc1(x)   # (B, C, F, T)
        e2 = self.enc2(e1)   # (B, 2C, F/2, T/2)
        e3 = self.enc3(e2)   # (B, 4C, F/4, T/4)
        
        # 中间
        m = self.mid(e3)
        
        # 解码 - 残差连接
        d3 = self.dec3(m)
        # 调整尺寸匹配
        target_size = (e2.shape[2], e2.shape[3])
        if d3.shape[2:] != target_size:
            d3 = torch.nn.functional.interpolate(d3, size=target_size, mode='bilinear', align_corners=False)
        d3 = d3 + e2
        
        d2 = self.dec2(d3)
        target_size = (e1.shape[2], e1.shape[3])
        if d2.shape[2:] != target_size:
            d2 = torch.nn.functional.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)
        d2 = d2 + e1
        
        # 输出
        out = self.out(d2)
        
        # 调整到目标F和T
        target_size = (F_dim, T_dim)
        out = torch.nn.functional.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        
        # 维度恢复: (B, 2, F, T) -> (B, F, T, 2)
        out = out.permute(0, 2, 3, 1)
        
        return out


def estimate_flops():
    """估算100帧FLOPs"""
    B, F, T = 1, 257, 100
    C = 12  # base_channels
    
    # 简化估算
    # enc1: 6->24, 3x3
    flops1 = 6 * C * 3 * 3 * B * F * T
    # enc2: 24->48, 3x3, stride
    flops2 = C * C*2 * 3 * 3 * B * (F//2) * (T//2)
    # enc3: 48->96, 3x3
    flops3 = C*2 * C*4 * 3 * 3 * B * (F//4) * (T//4)
    # mid: 96->96
    flops_mid = C*4 * C*4 * 3 * 3 * B * (F//4) * (T//4)
    # dec3: 96->48
    flops_d3 = C*4 * C*2 * 4 * 4 * B * (F//4) * (T//4)
    # dec2: 48->24
    flops_d2 = C*2 * C * 4 * 4 * B * (F//2) * (T//2)
    # out
    flops_out = C * 32 * 1 * 1 * B * F * T + 32 * 2 * 1 * 1 * B * F * T
    
    total = (flops1 + flops2 + flops3 + flops_mid + flops_d3 + flops_d2 + flops_out) / 1e6
    return total


def test_model():
    print("="*50)
    print("Testing v2 100帧 2D Conv")
    print("="*50)
    
    model = ComplexMaskNet(base_channels=24)
    params = sum(p.numel() for p in model.parameters())
    flops = estimate_flops()
    
    # 测试输入
    x = torch.randn(1, 257, 100, 3, 2)  # (B, F, T, CH, 2)
    
    # 合并实虚部: (B, F, T, CH, 2) -> (B, F, T, CH*2)
    x = x.reshape(1, 257, 100, 6)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"输入: (1, 257, 100, 3, 2)")
    print(f"合并后: (1, 257, 100, 6)")
    print(f"输出: {y.shape}")
    print(f"参数量: {params:,}")
    print(f"FLOPs: {flops:.1f} MFlops")
    print(f"目标<200M: {'PASS' if flops < 200 else 'FAIL'}")
    print(f"Mask范围: [{y.min():.4f}, {y.max():.4f}]")
    print("="*50)
    
    return params, flops


if __name__ == "__main__":
    test_model()
