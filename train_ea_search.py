"""
Lighting Speech Shield - 进化架构搜索版本 (Evolutionary Architecture Search)

约束：只使用 2D/1D 卷积、注意力、线性层、激活、Norm
禁止：GRU, LSTM 等循环算子

特点：
- 进化搜索最优模型架构配置
- 搜索空间：通道数、层数、注意力类型等
- 兼容已有的 Conv2D + Attention + DeNet 设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from lighting_speech_shield.model_v2 import ComplexMaskNet
from lighting_speech_shield.dataset import SpeechNoiseDataset


class ArchitectureGene:
    """架构基因 - 编码模型配置"""
    
    def __init__(self):
        # 可搜索的超参数
        self.base_channels = random.choice([8, 12, 16, 24])      # 基础通道数
        self.use_attention = random.choice([True, False])          # 是否使用注意力
        self.use_densenet = random.choice([True, False])           # 是否使用 DenseNet
        self.num_layers = random.choice([2, 3, 4, 5])              # 层数
        self.attention_type = random.choice(['se', 'freq', 'both']) # 注意力类型
        self.growth_rate = random.choice([8, 12, 16, 24])           # DenseNet 增长率
        self.drop_rate = random.choice([0.0, 0.05, 0.1])           # Dropout
        
        # 卷积核大小选项
        self.kernel_sizes = random.choice([[3], [3,5], [3,5,7]])
    
    def crossover(self, other):
        """基因交叉"""
        child = ArchitectureGene()
        
        # 随机选择父本基因
        if random.random() < 0.5:
            child.base_channels = self.base_channels
        else:
            child.base_channels = other.base_channels
            
        if random.random() < 0.5:
            child.use_attention = self.use_attention
        else:
            child.use_attention = other.use_attention
            
        if random.random() < 0.5:
            child.use_densenet = self.use_densenet
        else:
            child.use_densenet = other.use_densenet
            
        if random.random() < 0.5:
            child.num_layers = self.num_layers
        else:
            child.num_layers = other.num_layers
            
        if random.random() < 0.5:
            child.attention_type = self.attention_type
        else:
            child.attention_type = other.attention_type
            
        if random.random() < 0.5:
            child.growth_rate = self.growth_rate
        else:
            child.growth_rate = other.growth_rate
            
        if random.random() < 0.5:
            child.drop_rate = self.drop_rate
        else:
            child.drop_rate = other.drop_rate
            
        child.kernel_sizes = random.choice([self.kernel_sizes, other.kernel_sizes])
        
        return child
    
    def mutate(self, mutation_rate=0.3):
        """基因变异"""
        if random.random() < mutation_rate:
            self.base_channels = random.choice([8, 12, 16, 24])
        if random.random() < mutation_rate:
            self.use_attention = not self.use_attention
        if random.random() < mutation_rate:
            self.use_densenet = not self.use_densenet
        if random.random() < mutation_rate:
            self.num_layers = random.choice([2, 3, 4, 5])
        if random.random() < mutation_rate:
            self.attention_type = random.choice(['se', 'freq', 'both'])
        if random.random() < mutation_rate:
            self.growth_rate = random.choice([8, 12, 16, 24])
        if random.random() < mutation_rate:
            self.drop_rate = random.choice([0.0, 0.05, 0.1])
        if random.random() < mutation_rate:
            self.kernel_sizes = random.choice([[3], [3,5], [3,5,7]])
        
        return self
    
    def __str__(self):
        return (f"Gene(ch={self.base_channels}, att={self.use_attention}, "
                f"dense={self.use_densenet}, layers={self.num_layers}, "
                f"att_type={self.attention_type}, growth={self.growth_rate})")


class LightweightBlock(nn.Module):
    """轻量级构建块 - 只使用Conv/Attention/Norm/Activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 use_attention=False, attention_type='se'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
        self.use_attention = use_attention
        if use_attention:
            if attention_type == 'se':
                self.attn = SEAttention(out_channels)
            elif attention_type == 'freq':
                self.attn = FreqAttention(out_channels)
            elif attention_type == 'both':
                self.attn = nn.Sequential(
                    SEAttention(out_channels),
                    FreqAttention(out_channels)
                )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.use_attention:
            x = self.attn(x)
        return x


class SEAttention(nn.Module):
    """Squeeze-and-Excitation 注意力 - 只用 Conv/Linear"""
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


class FreqAttention(nn.Module):
    """频率维度注意力 - 只用 Conv/Linear"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.conv(x)


class EvolvableMaskNet(nn.Module):
    """
    可演化的降噪网络 - 基于基因配置构建
    只使用: Conv2d, Linear, Attention, Norm, Activation
    """
    
    def __init__(self, gene: ArchitectureGene, num_freq_bins=257, num_frames=100):
        super().__init__()
        
        self.gene = gene
        C = gene.base_channels
        
        # 输入投影
        self.input_conv = nn.Sequential(
            nn.Conv2d(6, C, 3, padding=1),
            nn.BatchNorm2d(C),
            nn.GELU()
        )
        
        # 主干网络 - 基于基因配置
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        channels = [C, C*2, C*4]
        
        for i in range(len(channels) - 1):
            # 编码器块
            block = LightweightBlock(
                channels[i], channels[i+1],
                kernel_size=gene.kernel_sizes[0] if i < len(gene.kernel_sizes) else 3,
                use_attention=gene.use_attention,
                attention_type=gene.attention_type
            )
            self.encoder.append(block)
        
        # 中间层
        if gene.use_densenet:
            self.mid = self._make_dense_block(channels[-1], gene.num_layers, gene.growth_rate)
            mid_channels = channels[-1] + gene.num_layers * gene.growth_rate
        else:
            self.mid = nn.Sequential(
                nn.Conv2d(channels[-1], channels[-1], 3, padding=1),
                nn.BatchNorm2d(channels[-1]),
                nn.GELU()
            )
            mid_channels = channels[-1]
        
        # 解码器
        for i in range(len(channels) - 2, -1, -1):
            block = nn.Sequential(
                nn.ConvTranspose2d(mid_channels if i == len(channels)-2 else channels[i+1], 
                                   channels[i], 4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i]),
                nn.GELU()
            )
            self.decoder.append(block)
            mid_channels = channels[i]
        
        # 输出头
        self.output = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _make_dense_block(self, in_channels, num_layers, growth_rate):
        """创建 DenseNet 块 - 只用 Conv/BatchNorm/Act"""
        layers = []
        ch = in_channels
        for i in range(num_layers):
            layers.append(self._make_dense_layer(ch, growth_rate))
            ch += growth_rate
        return nn.Sequential(*layers)
    
    def _make_dense_layer(self, in_channels, growth_rate):
        """单个 DenseNet 层"""
        bn_size = 4
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.GELU(),
            nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (B, F, T, 6)
        B, F, T, C = x.shape
        
        # 维度重排: (B, F, T, C) -> (B, C, F, T)
        x = x.permute(0, 3, 1, 2)
        
        # 输入
        x = self.input_conv(x)
        
        # 编码器
        encoder_outs = [x]
        for block in self.encoder:
            x = block(x)
            x = F.max_pool2d(x, 2)
            encoder_outs.append(x)
        
        # 中间
        x = self.mid(x)
        
        # 解码器
        for i, block in enumerate(self.decoder):
            x = block(x)
            # 残差连接
            if i < len(encoder_outs) - 1:
                skip = encoder_outs[-(i+2)]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip
        
        # 调整大小
        if x.shape[2:] != (F, T):
            x = F.interpolate(x, size=(F, T), mode='bilinear', align_corners=False)
        
        # 输出
        out = self.output(x)
        
        # 维度恢复: (B, 2, F, T) -> (B, F, T, 2)
        out = out.permute(0, 2, 3, 1)
        
        return out


class EvolutionConfig:
    """进化架构搜索配置"""
    population_size: int = 12
    elite_size: int = 2
    generations: int = 30
    tournament_size: int = 3
    mutation_rate: float = 0.4
    
    # 训练配置
    train_epochs: int = 3
    batch_size: int = 4
    num_frames: int = 100
    lr: float = 0.001


class EvolutionaryArchitectureSearch:
    """进化架构搜索"""
    
    def __init__(self, config: EvolutionConfig, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.population = []
        self.fitness_history = []
    
    def initialize_population(self):
        """初始化种群"""
        print(f"Initializing population with {self.config.population_size} architectures...")
        self.population = [ArchitectureGene() for _ in range(self.config.population_size)]
    
    def build_model(self, gene: ArchitectureGene):
        """根据基因构建模型"""
        return EvolvableMaskNet(gene, num_freq_bins=257, num_frames=self.config.num_frames)
    
    def evaluate(self, gene, train_loader, val_loader):
        """评估基因适应度"""
        model = self.build_model(gene).to(self.device)
        
        # 快速训练
        optimizer = optim.AdamW(model.parameters(), lr=self.config.lr)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(self.config.train_epochs):
            for input_spec, target_mask in train_loader:
                input_spec = input_spec.to(self.device)
                target_mask = target_mask.to(self.device)
                
                optimizer.zero_grad()
                pred = model(input_spec)
                loss = criterion(pred, target_mask)
                loss.backward()
                optimizer.step()
        
        # 验证
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for input_spec, target_mask in val_loader:
                if count >= 16:
                    break
                input_spec = input_spec.to(self.device)
                target_mask = target_mask.to(self.device)
                
                pred = model(input_spec)
                loss = criterion(pred, target_mask)
                total_loss += loss.item()
                count += 1
        
        return total_loss / max(count, 1)
    
    def tournament_selection(self, fitnesses):
        """锦标赛选择"""
        indices = random.sample(range(len(self.population)), self.config.tournament_size)
        best_idx = min(indices, key=lambda i: fitnesses[i])
        return copy.deepcopy(self.population[best_idx])
    
    def evolve(self, train_dataset, val_dataset=None):
        """执行进化搜索"""
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                  shuffle=True, num_workers=0)
        
        if val_dataset is None:
            val_size = min(32, len(train_dataset) // 5)
            val_dataset = Subset(train_dataset, list(range(val_size)))
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # 初始化
        self.initialize_population()
        
        best_gene = None
        best_fitness = float('inf')
        
        for gen in range(self.config.generations):
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{self.config.generations}")
            print(f"{'='*60}")
            
            # 评估
            fitnesses = []
            for i, gene in enumerate(self.population):
                fitness = self.evaluate(gene, train_loader, val_loader)
                fitnesses.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_gene = copy.deepcopy(gene)
                    print(f"  [NEW BEST] Arch {i}: {gene}")
                    print(f"             Fitness: {fitness:.6f}")
            
            avg_fitness = np.mean(fitnesses)
            self.fitness_history.append((best_fitness, avg_fitness))
            print(f"  Best: {best_fitness:.6f}, Avg: {avg_fitness:.6f}")
            
            # 精英保留
            sorted_idx = np.argsort(fitnesses)
            elite_genes = [self.population[i] for i in sorted_idx[:self.config.elite_size]]
            
            # 新种群
            new_population = list(elite_genes)
            
            while len(new_population) < self.config.population_size:
                # 选择
                parent1 = self.tournament_selection(fitnesses)
                parent2 = self.tournament_selection(fitnesses)
                
                # 交叉
                child = parent1.crossover(parent2)
                
                # 变异
                child.mutate(self.config.mutation_rate)
                
                new_population.append(child)
            
            self.population = new_population
            
            # 学习率衰减
            self.config.lr *= 0.95
        
        return best_gene, best_fitness


def run_architecture_search(data_dir='data/synthetic', generations=15):
    """运行架构搜索"""
    config = EvolutionConfig()
    config.generations = generations
    config.population_size = 8
    config.train_epochs = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 数据集
    dataset = SpeechNoiseDataset(data_dir, num_frames=config.num_frames)
    print(f"Dataset: {len(dataset)} samples")
    
    # 分割
    val_size = min(32, len(dataset) // 5)
    train_dataset = dataset
    val_dataset = Subset(dataset, list(range(val_size)))
    
    # 进化搜索
    searcher = EvolutionaryArchitectureSearch(config, device=str(device))
    best_gene, best_fitness = searcher.evolve(train_dataset, val_dataset)
    
    print(f"\n{'='*60}")
    print("Architecture Search Complete!")
    print(f"Best Gene: {best_gene}")
    print(f"Best Fitness: {best_fitness:.6f}")
    print(f"{'='*60}")
    
    # 用最佳基因训练最终模型
    print("\nTraining final model with best architecture...")
    final_model = EvolvableMaskNet(best_gene).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = optim.AdamW(final_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(20):
        final_model.train()
        total_loss = 0
        for input_spec, target_mask in train_loader:
            input_spec, target_mask = input_spec.to(device), target_mask.to(device)
            optimizer.zero_grad()
            pred = final_model(input_spec)
            loss = criterion(pred, target_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: loss = {total_loss/len(train_loader):.6f}")
    
    # 保存
    save_path = Path('checkpoints')
    save_path.mkdir(exist_ok=True)
    torch.save({
        'gene': best_gene.__dict__,
        'model_state': final_model.state_dict()
    }, save_path / 'eas_best_model.pth')
    
    return final_model, best_gene


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/synthetic')
    parser.add_argument('--generations', type=int, default=15)
    args = parser.parse_args()
    
    run_architecture_search(args.data_dir, args.generations)
