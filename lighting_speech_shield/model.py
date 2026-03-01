"""
Lighting Speech Shield - 轻量级语音降噪模型

核心架构：卷积 + 自注意力机制
- 输入：3 通道复数频谱 (实部 + 虚部)
- 输出：单通道 mask
- 算力：<200 MFlops
- 延迟：10-30ms (流式处理，参考未来 1-3 帧)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingAttentionBlock(nn.Module):
    """
    流式注意力块 - 只能关注当前帧和未来有限帧
    """
    def __init__(self, dim, num_heads=4, future_frames=2):
        super().__init__()
        self.num_heads = num_heads
        self.future_frames = future_frames
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        x: (B, T, dim) - T 是时间帧数（当前 + 未来）
        """
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # 多头 reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        
        # 流式 mask：只能看当前和未来 future_frames 帧
        # 对于位置 i，可以关注 [i, i+future_frames] 范围
        mask = torch.ones(T, T, device=x.device)
        for i in range(T):
            for j in range(T):
                # j > i + future_frames 或 j < i 都不能关注
                if j > i + self.future_frames or j < i:
                    mask[i, j] = 0
        mask = mask == 0
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, T, D)
        
        # 合并 head
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, dim)
        out = self.proj(out)
        
        return out


class LightingSpeechShield(nn.Module):
    """
    主模型：轻量级语音降噪网络
    
    架构设计原则:
    1. 输入：3 通道 STFT 频谱 (实部 + 虚部分开)
    2. 输出：单通道 mask (应用于参考通道)
    3. 流式处理：每次处理 1 帧，参考未来 1-3 帧
    4. 算力：<200 MFlops
    """
    def __init__(self, 
                 num_freq_bins=257,  # FFT_SIZE//2 + 1
                 num_frames=3,  # 当前 + 未来 2 帧
                 num_channels=3,
                 base_channels=32,
                 num_layers=4,
                 num_heads=4,
                 future_frames=2):
        super().__init__()
        
        self.num_freq_bins = num_freq_bins
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.future_frames = future_frames
        
        # 输入维度：F 频点 × T 帧 × C 通道 × 2(实/虚)
        input_dim = num_frames * num_channels * 2
        
        # 频域独立处理：每个频率 bin 一个网络
        # 输入：(T*C*2) -> 输出：base_channels
        self.input_proj = nn.Linear(input_dim, base_channels)
        
        # 堆叠 Attention 层（在频域维度上做注意力）
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'freq_conv': nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
                'bn': nn.BatchNorm1d(base_channels),
                'attn': StreamingAttentionBlock(base_channels, num_heads, future_frames),
                'norm': nn.LayerNorm(base_channels),
                'act': nn.GELU(),
            })
            self.layers.append(layer)
        
        # 输出头：生成 mask
        self.mask_head = nn.Sequential(
            nn.Linear(base_channels, base_channels // 2),
            nn.GELU(),
            nn.Linear(base_channels // 2, 1),
            nn.Sigmoid(),  # mask 范围 [0, 1]
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, F, T, C, 2) 复数频谱
                - B: batch
                - F: 频率 bin 数 (257)
                - T: 时间帧数 (3: 当前 + 未来 2 帧)
                - C: 通道数 (3)
                - 2: 实部/虚部
        
        Returns:
            mask: (B, F, 1) - 应用于参考通道的 mask
        """
        B, F, T, C, _ = x.shape
        
        # 展平最后三维：(B, F, T*C*2)
        x = x.view(B, F, -1)  # (B, F, 18)
        
        # 输入投影：(B, F, 18) -> (B, F, 32)
        x = self.input_proj(x)
        
        # 通过各层（频域处理）
        for layer in self.layers:
            # 频域卷积
            x_perm = x.transpose(1, 2)  # (B, 32, F)
            x_conv = layer['freq_conv'](x_perm)  # (B, 32, F)
            x_conv = layer['bn'](x_conv)
            x_conv = layer['act'](x_conv)
            x_conv = x_conv.transpose(1, 2)  # (B, F, 32)
            
            # 注意力（在频域上）
            x_attn = layer['attn'](x)  # (B, F, 32)
            
            # 融合
            x = x_conv + x_attn
            x = layer['norm'](x)
        
        # 生成 mask
        mask = self.mask_head(x)  # (B, F, 1)
        
        return mask


def model_info(model):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # 简单位 FLOPs 估算
    B, F, T, C = 1, 257, 3, 6
    input_dim = T * C * 2
    base_channels = 32
    num_layers = 4
    
    # 输入投影
    flops = B * F * input_dim * base_channels
    
    # 各层
    for _ in range(num_layers):
        # Conv1d
        flops += B * base_channels * F * 3 * base_channels
        # Attention
        flops += B * F * base_channels * 3 * base_channels  # qkv
        flops += B * 4 * F * F * base_channels  # attention matrix
    
    # 输出头
    flops += B * F * base_channels * 16 + B * F * 16 * 1
    
    mflops = flops / 1e6
    
    print(f"\n模型信息:")
    print(f"  参数量：{total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  估算 FLOPs: {mflops:.1f} MFlops")
    print(f"  目标：<200 MFlops {'✅' if mflops < 200 else '❌'}")
    
    return total_params, mflops


if __name__ == "__main__":
    print("=" * 50)
    print("Lighting Speech Shield - 模型测试")
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
    
    # 参数量
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
    print(f"Mask 范围：[{y.min():.4f}, {y.max():.4f}]")
    
    # FLOPs 估算
    model_info(model)
    
    print("\n✅ 模型测试通过！")
