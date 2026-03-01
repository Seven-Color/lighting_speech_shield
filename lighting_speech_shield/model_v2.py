"""
Lighting Speech Shield v2 - 简化版

架构:
- 频域压缩：257 → 64
- UNet-like 编码器 - 解码器
- RMSNorm + Attention
- 目标：<200 MFlops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm - 轻量归一化"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class AttentionBlock(nn.Module):
    """多头自注意力 + FFN"""
    def __init__(self, dim, num_heads=4, context_size=5, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.context_size = context_size
        
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """x: (B, T, dim)"""
        B, T, _ = x.shape
        
        # Pre-norm + QKV
        x_norm = self.norm1(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        # Multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Local attention mask (causal + future)
        mask = torch.ones(T, T, device=x.device)
        for i in range(T):
            for j in range(T):
                if abs(i - j) > self.context_size:
                    mask[i, j] = 0
        mask = mask == 0
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        
        # Residual + FFN
        x = x + out
        x = x + self.ffn(self.norm2(x))
        
        return x


class EncoderBlock(nn.Module):
    """编码器块：Conv + Attention + Downsample"""
    def __init__(self, in_ch, out_ch, num_heads=4):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        
        self.norm = RMSNorm(out_ch)
        self.attn = AttentionBlock(out_ch, num_heads)
        
        # 下采样：频域减半
        self.downsample = nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1, bias=False)
    
    def forward(self, x):
        """x: (B, C, F)"""
        # Conv
        x = self.conv(x)
        
        # (B, C, F) -> (B, F, C) for attention
        x = x.transpose(1, 2)
        x = self.attn(x)
        x = x.transpose(1, 2)  # (B, C, F)
        
        # Skip connection
        skip = x
        
        # Downsample
        x = self.downsample(x)
        
        return x, skip


class DecoderBlock(nn.Module):
    """解码器块：Upsample + Concat + Conv + Attention"""
    def __init__(self, in_ch, skip_ch, out_ch, num_heads=4):
        super().__init__()
        
        # 上采样
        self.upsample = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        
        # 融合 skip
        self.fuse = nn.Conv1d(out_ch + skip_ch, out_ch, 1, bias=False)
        
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        
        self.norm = RMSNorm(out_ch)
        self.attn = AttentionBlock(out_ch, num_heads)
    
    def forward(self, x, skip):
        """
        x: (B, C, F)
        skip: (B, skip_C, F_skip)
        """
        # Upsample
        x = self.upsample(x)
        
        # Fuse with skip
        x = self.fuse(torch.cat([x, skip], dim=1))
        
        # Conv
        x = self.conv(x)
        
        # Attention
        x = x.transpose(1, 2)
        x = self.attn(x)
        x = x.transpose(1, 2)
        
        return x


class LightingSpeechShieldV2(nn.Module):
    """
    Lighting Speech Shield v2
    
    架构:
    输入 → 频域压缩 → UNet(Enc-Dec) → 频域解压缩 → Mask
    """
    def __init__(self,
                 num_freq_in=257,
                 num_freq_compressed=64,
                 input_dim=18,  # T*C*2 = 3*3*2
                 base_channels=64):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, base_channels),
            RMSNorm(base_channels),
            nn.GELU()
        )
        
        # 频域压缩
        self.freq_compress = nn.Linear(num_freq_in, num_freq_compressed)
        
        # UNet 编码器
        self.enc1 = EncoderBlock(base_channels, base_channels, num_heads=4)  # 64 -> 64
        self.enc2 = EncoderBlock(base_channels, base_channels * 2, num_heads=4)  # 64 -> 128
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4, num_heads=8)  # 128 -> 256
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            AttentionBlock(base_channels * 4, num_heads=8),
            RMSNorm(base_channels * 4),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
        # UNet 解码器 (注意 skip 通道数)
        self.dec3 = DecoderBlock(base_channels * 4, base_channels * 4, base_channels * 2, num_heads=8)  # skip=256
        self.dec2 = DecoderBlock(base_channels * 2, base_channels * 2, base_channels, num_heads=4)  # skip=128
        self.dec1 = DecoderBlock(base_channels, base_channels, base_channels, num_heads=4)  # skip=64
        
        # 频域解压缩
        self.freq_decompress = nn.Linear(num_freq_compressed, num_freq_in)
        
        # Mask 头
        self.mask_head = nn.Sequential(
            nn.Linear(base_channels, base_channels // 2),
            RMSNorm(base_channels // 2),
            nn.GELU(),
            nn.Linear(base_channels // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: (B, F, T, C, 2) -> (B, F, 18)
        """
        B, F, T, C, _ = x.shape
        
        # 展平输入
        x = x.view(B, F, -1)  # (B, F, 18)
        
        # 输入投影
        x = self.input_proj(x)  # (B, F, 64)
        
        # 频域压缩：(B, F=257, 64) -> (B, F_comp=64, 64)
        x = x.transpose(1, 2)  # (B, 64, 257)
        x = self.freq_compress(x)  # Linear(257->64): (B, 64, 64)
        
        # 转置为 (B, C, F) 格式用于 UNet
        x = x.transpose(1, 2)  # (B, 64, 64)
        
        # 编码器
        enc1, skip1 = self.enc1(x)
        enc2, skip2 = self.enc2(enc1)
        enc3, skip3 = self.enc3(enc2)
        
        # Bottleneck
        x = self.bottleneck(enc3.transpose(1, 2)).transpose(1, 2)
        
        # 解码器
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # 频域解压缩：(B, F_comp=64, C=64) -> (B, F_in=257, C=64)
        x = x.transpose(1, 2)  # (B, 64, 64)
        x = self.freq_decompress(x)  # Linear(64->257): (B, 64, 257)
        x = x.transpose(1, 2)  # (B, 257, 64)
        
        # Mask 输出
        mask = self.mask_head(x)  # (B, F, 1)
        
        return mask


def estimate_flops(model):
    """简化 FLOPs 估算"""
    # 参数量估算 FLOPs
    params = sum(p.numel() for p in model.parameters())
    # 假设每个参数每次 forward 使用一次
    flops = params * 2  # MACs -> FLOPs
    return flops / 1e6


def model_info(model):
    """打印模型信息"""
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    mflops = estimate_flops(model)
    
    x = torch.randn(1, 257, 3, 3, 2)
    with torch.no_grad():
        y = model(x)
    
    print(f"\n{'='*50}")
    print(f"Lighting Speech Shield v2")
    print(f"{'='*50}")
    print(f"输入：{x.shape} -> 输出：{y.shape}")
    print(f"参数量：{params:,} ({params/1e6:.2f}M)")
    print(f"估算 FLOPs: {mflops:.1f} MFlops")
    print(f"目标：<200 MFlops [{'PASS' if mflops < 200 else 'FAIL'}]")
    print(f"{'='*50}\n")
    
    return params, mflops


if __name__ == "__main__":
    model = LightingSpeechShieldV2()
    params, flops = model_info(model)
    print(f"[OK] Model test passed!")
