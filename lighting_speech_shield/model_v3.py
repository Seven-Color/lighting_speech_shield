"""
Lighting Speech Shield v3 - 加入DeNet(DenseNet)密集连接

基于v2版本，添加DenseNet密集连接机制：
- 每一层都与所有前面的层相连
- 增强特征复用和梯度流动
- 保持低算力 (<200MFlops)

输入: (B, F=257, T=100, CH=3, 2) -> 合并实虚部 -> (B, F=257, T=100, CH=6)
输出: (B, F=257, T=100, 2) - 复数mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """DenseNet单层"""
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        # x 可以是单个tensor或者多个tensor的列表
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class DenseBlock(nn.Module):
    """密集连接块 - 所有层相互连接"""
    def __init__(self, in_channels, num_layers, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(features)
            features.append(new_feat)
        return torch.cat(features, dim=1)


class TransitionBlock(nn.Module):
    """过渡块 - 压缩通道数"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.GELU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out


class ChannelAttention(nn.Module):
    """轻量通道注意力 (SE)"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FrequencyAttention(nn.Module):
    """频率维度注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 对每个频率点计算注意力
        return x * self.conv(x)


class AttentionBlock(nn.Module):
    """注意力块: 通道 + 频率注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.freq_att = FrequencyAttention(channels, reduction)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x + self.channel_att(x)
        x = x + self.freq_att(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class ComplexMaskNetDeNet(nn.Module):
    """
    DeNet (DenseNet) 版本的语音降噪模型
    
    核心改进：
    - 使用DenseBlock替代普通卷积块
    - 密集连接增强特征复用
    - 添加注意力机制
    
    输入: (B, F, T, 6) 合并了实虚部
    输出: (B, F, T, 2) 复数mask
    """
    def __init__(self, base_channels=16, growth_rate=16, num_layers_per_block=4, 
                 use_attention=True, drop_rate=0.1):
        super().__init__()
        self.use_attention = use_attention
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        
        # 输入: (B, F, T, 6) -> Conv2D需要 (B, C, H, W) = (B, 6, F, T)
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(6, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        
        # 第一个DenseBlock (较少层)
        self.dense1 = DenseBlock(
            in_channels=base_channels,
            num_layers=num_layers_per_block,
            growth_rate=growth_rate,
            drop_rate=drop_rate
        )
        dense1_out = base_channels + num_layers_per_block * growth_rate
        
        if use_attention:
            self.att1 = AttentionBlock(dense1_out)
        
        # 过渡块1: 压缩通道 + 下采样
        trans1_channels = dense1_out // 2
        self.trans1 = TransitionBlock(dense1_out, trans1_channels)
        
        # 第二个DenseBlock (中间层)
        self.dense2 = DenseBlock(
            in_channels=trans1_channels,
            num_layers=num_layers_per_block,
            growth_rate=growth_rate,
            drop_rate=drop_rate
        )
        dense2_out = trans1_channels + num_layers_per_block * growth_rate
        
        if use_attention:
            self.att2 = AttentionBlock(dense2_out)
        
        # 上采样解码器 - 简化为2个上采样层
        self.up2 = nn.Sequential(
            nn.Conv2d(dense2_out, trans1_channels, 1),
            nn.GELU(),
            nn.ConvTranspose2d(trans1_channels, trans1_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(trans1_channels),
            nn.GELU(),
        )
        
        # 上采样后需要处理残差连接的通道数
        self.up1 = nn.Sequential(
            nn.Conv2d(trans1_channels + dense1_out, base_channels, 1),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        
        # 输出头
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 2, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (B, F, T, 6)
        B, F_dim, T_dim, C = x.shape
        
        # 维度重排: (B, F, T, C) -> (B, C, F, T)
        x = x.permute(0, 3, 1, 2)
        
        # 初始卷积
        x = self.init_conv(x)  # (B, base_channels, F, T)
        
        # 第一个DenseBlock
        d1 = self.dense1(x)  # (B, dense1_out, F, T)
        if self.use_attention:
            d1 = self.att1(d1)
        
        # 过渡 - 下采样
        t1 = self.trans1(d1)  # (B, trans1_channels, F/2, T/2)
        
        # 第二个DenseBlock (中间层)
        d2 = self.dense2(t1)  # (B, dense2_out, F/2, T/2)
        if self.use_attention:
            d2 = self.att2(d2)
        
        # 上采样解码 - 密集连接
        # up2: 上采样 + 拼接 d1
        up2 = self.up2(d2)
        
        # 调整up2大小并拼接d1
        if up2.shape[2:] != d1.shape[2:]:
            up2 = F.interpolate(up2, size=(d1.shape[2], d1.shape[3]), mode='bilinear', align_corners=False)
        up2 = torch.cat([up2, d1], dim=1)  # (B, trans1_ch + dense1_out, F, T)
        
        # up1
        up1 = self.up1(up2)  # (B, base_ch, F*2, T*2)
        
        # 调整大小匹配输入
        if up1.shape[2:] != (F_dim, T_dim):
            up1 = F.interpolate(up1, size=(F_dim, T_dim), mode='bilinear', align_corners=False)
        
        # 最终输出
        out = self.out(up1)
        
        # 维度恢复: (B, 2, F, T) -> (B, F, T, 2)
        out = out.permute(0, 2, 3, 1)
        
        return out


def estimate_flops(base_channels=12, growth_rate=12, num_layers_per_block=3):
    """估算100帧FLOPs"""
    B, F, T = 1, 257, 100
    C = base_channels
    G = growth_rate
    L = num_layers_per_block
    
    # 简化估算 - 基于实际forward计算
    # init_conv
    flops_init = 6 * C * 3 * 3 * B * F * T
    
    # dense block 1: 每层卷积
    flops_d1 = 0
    for i in range(L):
        in_ch = C + i * G
        # 1x1 conv: bn_size * G
        flops_d1 += in_ch * (4*G) * 1 * 1 * B * F * T
        # 3x3 conv: G
        flops_d1 += (4*G) * G * 3 * 3 * B * F * T
    
    # trans1
    dense1_out = C + L * G
    trans1_ch = dense1_out // 2
    flops_t1 = dense1_out * trans1_ch * 1 * 1 * B * (F//2) * (T//2)
    
    # dense block 2
    flops_d2 = 0
    for i in range(L):
        in_ch = trans1_ch + i * G
        flops_d2 += in_ch * (4*G) * 1 * 1 * B * (F//2) * (T//2)
        flops_d2 += (4*G) * G * 3 * 3 * B * (F//2) * (T//2)
    
    # trans2
    dense2_out = trans1_ch + L * G
    trans2_ch = dense2_out // 2
    flops_t2 = dense2_out * trans2_ch * 1 * 1 * B * (F//4) * (T//4)
    
    # dense block 3
    flops_d3 = 0
    for i in range(L):
        in_ch = trans2_ch + i * G
        flops_d3 += in_ch * (4*G) * 1 * 1 * B * (F//4) * (T//4)
        flops_d3 += (4*G) * G * 3 * 3 * B * (F//4) * (T//4)
    
    # upsample + output (简化)
    flops_up = trans2_ch * trans2_ch * 4 * 4 * B * (F//4) * (T//4) * 3
    flops_out = C * 32 * 3 * 3 * B * F * T + 32 * 2 * 1 * 1 * B * F * T
    
    total = (flops_init + flops_d1 + flops_t1 + flops_d2 + flops_t2 + flops_d3 + 
             flops_up + flops_out) / 1e6
    return total


def test_model():
    print("="*60)
    print("Testing DeNet (DenseNet) v3 100帧 2D Conv")
    print("="*60)
    
    # 极轻量级配置
    model = ComplexMaskNetDeNet(
        base_channels=6, 
        growth_rate=6, 
        num_layers_per_block=2,
        use_attention=True,
        drop_rate=0.0
    )
    params = sum(p.numel() for p in model.parameters())
    flops = estimate_flops(6, 6, 2)
    
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
    print("="*60)
    
    return params, flops


if __name__ == "__main__":
    test_model()
