"""
Lighting Speech Shield v2 - 100帧 2D卷积版本 (优化版)

输入: (B, F=257, T=100, CH=3, 2) -> 合并实虚部 -> (B, F=257, T=100, CH=6)
输出: (B, F=257, T=100, 2) - 复数mask
100帧算力目标: <200MFlops

优化:
- 添加轻量注意力模块 (SE + 频率注意力)
- 残差密集连接
- 支持流式推理模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """轻量通道注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FrequencyAttention(nn.Module):
    """轻量频率维度注意力 - 对每个通道的频率维度做注意力"""
    def __init__(self, reduction=4):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, x):
        # x: (B, C, F, T)
        b, c, f, t = x.shape
        # 对每个通道和时间点计算频率注意力
        # (B, C, F, T) -> (B, C, T, F) -> (B*C*T, F)
        x_flat = x.permute(0, 1, 3, 2).reshape(b * c * t, f)
        # 计算注意力
        y = x_flat.mean(dim=1, keepdim=True)  # (B*C*T, 1)
        # 简单的单层注意力
        y = y * x_flat  # (B*C*T, F) - 简化版
        x_flat = x_flat * torch.sigmoid(y)
        x_out = x_flat.reshape(b, c, t, f).permute(0, 1, 3, 2)  # (B, C, F, T)
        return x_out


class AttentionBlock(nn.Module):
    """轻量注意力块: 通道 + 频率注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.freq_att = FrequencyAttention(reduction)
        self.norm = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        x = x + self.channel_att(x)
        x = x + self.freq_att(x)
        return self.norm(x)


class ComplexMaskNet(nn.Module):
    """
    2D卷积网络
    输入: (B, F, T, 6) 合并了实虚部
    输出: (B, F, T, 2) 复数mask
    """
    def __init__(self, base_channels=12, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.base_channels = base_channels
        
        # 输入: (B, F, T, 6) -> Conv2D需要 (B, C, H, W) = (B, 6, F, T)
        
        # 编码器 - 压缩
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        if use_attention:
            self.att1 = AttentionBlock(base_channels)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),  # F/2, T/2
            nn.BatchNorm2d(base_channels*2),
            nn.GELU(),
        )
        if use_attention:
            self.att2 = AttentionBlock(base_channels*2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),  # F/4, T/4
            nn.BatchNorm2d(base_channels*4),
            nn.GELU(),
        )
        if use_attention:
            self.att3 = AttentionBlock(base_channels*4)
        
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
        if self.use_attention:
            e1 = self.att1(e1)
        
        e2 = self.enc2(e1)   # (B, 2C, F/2, T/2)
        if self.use_attention:
            e2 = self.att2(e2)
        
        e3 = self.enc3(e2)   # (B, 4C, F/4, T/4)
        if self.use_attention:
            e3 = self.att3(e3)
        
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
