"""
Lighting Speech Shield - 进化算法训练版本 (完整版)

约束：只使用 2D/1D 卷积、注意力、线性层、激活、Norm
禁止：GRU, LSTM 等循环算子

特点：
- 自包含数据生成（无需外部数据）
- 进化算法训练模型权重
- 训练后自动评估效果
- 支持断点续训
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent))

from lighting_speech_shield.stft import STFTProcessor


# ============================================================================
# 数据生成模块（自包含）
# ============================================================================

class SyntheticDataGenerator:
    """合成数据生成器 - 生成多通道带噪语音"""
    
    def __init__(self, sample_rate=16000, num_channels=3):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        
    def generate_pure_speech(self, duration=1.0):
        """生成模拟语音（谐波模型）"""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        f0 = 150 + np.random.uniform(-30, 30)
        
        formants = {
            'a': [730, 1090, 2440], 'e': [530, 1840, 2480],
            'i': [270, 2290, 3010], 'o': [570, 840, 2410], 'u': [300, 870, 2240]
        }
        f_type = np.random.choice(list(formants.keys()))
        f1, f2, f3 = formants[f_type]
        
        speech = np.zeros_like(t)
        for h in range(1, 12):
            speech += (1.0/h) * np.sin(2*np.pi * f0 * h * t)
        for f_res in [f1, f2, f3]:
            bw = f_res * 0.1
            speech += 0.2 * np.sin(2*np.pi*f_res*t) * np.exp(-bw*t)
        
        envelope = np.exp(-3 * np.abs(t - duration/2))
        speech *= envelope
        speech = speech / (np.abs(speech).max() + 1e-8) * 0.5
        return speech.astype(np.float32)
    
    def generate_noise(self, noise_type='white', duration=1.0):
        """生成噪声"""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        if noise_type == 'white':
            noise = np.random.randn(len(t)).astype(np.float32)
        elif noise_type == 'pink':
            noise = np.cumsum(np.random.randn(len(t)))
            noise = (noise - noise.mean()).astype(np.float32)
        elif noise_type == 'babble':
            noise = np.zeros_like(t)
            for _ in range(6):
                f0 = 100 + np.random.uniform(50, 180)
                for h in range(1, 8):
                    noise += np.random.uniform(0.1, 0.2) * np.sin(
                        2*np.pi*f0*h*t + np.random.uniform(0, 2*np.pi))
            noise = noise.astype(np.float32)
        else:
            noise = np.random.randn(len(t)).astype(np.float32) * 0.3
        
        noise = noise / (np.abs(noise).max() + 1e-8) * 0.3
        return noise
    
    def generate_multichannel(self, speech, noise):
        """多通道生成（延迟+衰减）"""
        channels = []
        for c in range(self.num_channels):
            delay = int(c * 3)
            atten = 1.0 - c * 0.08
            s_delayed = np.pad(speech[:-delay], (delay, 0), mode='constant') if delay > 0 else speech
            ch = s_delayed * atten + noise * (1 + c * 0.1)
            channels.append(ch)
        return np.stack(channels, axis=0)
    
    def mix(self, speech, noise, snr_db):
        """按SNR混合"""
        sp = np.mean(speech**2)
        ns = np.mean(noise**2)
        scale = np.sqrt(sp / (ns * 10**(snr_db/10) + 1e-8))
        return speech + noise * scale
    
    def generate_sample(self, duration=1.0, snr_range=(0, 15)):
        """生成单个样本"""
        noise_types = ['white', 'pink', 'babble']
        clean = self.generate_pure_speech(duration)
        noise = self.generate_noise(random.choice(noise_types), duration)
        snr = np.random.uniform(*snr_range)
        
        noisy = self.mix(clean, noise, snr)
        clean_mc = self.generate_multichannel(clean, np.zeros_like(clean))
        noisy_mc = self.generate_multichannel(noisy, noise)
        
        return {
            'clean': clean_mc, 'noisy': noisy_mc, 
            'snr_db': float(snr), 'noise_type': noise_types[0]
        }


class SpeechNoiseDataset(Dataset):
    """语音降噪数据集"""
    
    def __init__(self, data_dir=None, num_samples=200, num_frames=100, 
                 sample_rate=16000, num_channels=3, n_fft=512, hop_length=160):
        self.num_frames = num_frames
        self.stft = STFTProcessor(sample_rate, n_fft, hop_length)
        
        # 如果有现成数据则加载，否则生成
        if data_dir and Path(data_dir).exists():
            self.load_existing(data_dir)
        else:
            print(f"Generating {num_samples} synthetic samples...")
            self.generate_data(num_samples, sample_rate, num_channels)
    
    def load_existing(self, data_dir):
        """加载现有数据"""
        with open(Path(data_dir) / 'metadata.json') as f:
            self.metadata = json.load(f)
        self.data_dir = Path(data_dir)
        self._cached_data = {}
        
    def generate_data(self, num_samples, sample_rate, num_channels):
        """生成合成数据"""
        gen = SyntheticDataGenerator(sample_rate, num_channels)
        self.metadata = []
        self._cached_data = {}
        
        for i in tqdm(range(num_samples), desc="Generating data"):
            sample = gen.generate_sample(duration=1.0)
            self._cached_data[i] = {
                'clean': sample['clean'],
                'noisy': sample['noisy']
            }
            self.metadata.append({'idx': i, 'snr_db': sample['snr_db']})
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if hasattr(self, '_cached_data') and idx in self._cached_data:
            clean = self._cached_data[idx]['clean']
            noisy = self._cached_data[idx]['noisy']
        else:
            item = self.metadata[idx]
            data = np.load(self.data_dir / item['filename'])
            clean = data['clean']
            noisy = data['noisy']
        
        clean = torch.from_numpy(clean).float()
        noisy = torch.from_numpy(noisy).float()
        
        # STFT
        clean_spec = self.stft.forward(clean.unsqueeze(0), return_complex=True)
        noisy_spec = self.stft.forward(noisy.unsqueeze(0), return_complex=True)
        
        # 扩展到目标帧数
        while clean_spec.shape[1] < self.num_frames:
            clean_spec = torch.cat([clean_spec, clean_spec], dim=1)
            noisy_spec = torch.cat([noisy_spec, noisy_spec], dim=1)
        
        clean_spec = clean_spec[:, :self.num_frames, :, :, :]
        noisy_spec = noisy_spec[:, :self.num_frames, :, :, :]
        
        # 计算mask (参考通道 C=0)
        noisy_c0 = noisy_spec[:, :, :, 0, :]
        clean_c0 = clean_spec[:, :, :, 0, :]
        
        noisy_mag = torch.sqrt(noisy_c0[..., 0]**2 + noisy_c0[..., 1]**2).clamp(min=1e-8)
        
        mask_real = clean_c0[..., 0] / noisy_mag
        mask_imag = clean_c0[..., 1] / noisy_mag
        
        mask_mag = torch.sqrt(mask_real**2 + mask_imag**2).clamp(max=2.0)
        mask_angle = torch.atan2(mask_imag, mask_real)
        mask_real = mask_mag * torch.cos(mask_angle)
        mask_imag = mask_mag * torch.sin(mask_angle)
        
        # 整理形状: (T, F, 2) -> (F, T, 2)
        mask = torch.stack([mask_real.squeeze(0), mask_imag.squeeze(0)], dim=-1)
        mask = mask.permute(1, 0, 2)
        
        # 输入: (T, F, C, 2) -> (F, T, C*2)
        noisy_input = noisy_spec.squeeze(0).permute(1, 0, 2, 3).reshape(257, self.num_frames, 6)
        
        return noisy_input, mask


# ============================================================================
# 模型定义（只使用 Conv/Attention/Linear/Norm/Act）
# ============================================================================

class SEAttention(nn.Module):
    """通道注意力 - 只用 Linear"""
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
        return x * self.fc(y).view(b, c, 1, 1)


class FreqAttention(nn.Module):
    """频率注意力 - 只用 Conv"""
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


class AttentionBlock(nn.Module):
    """注意力块: 通道 + 频率"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_att = SEAttention(channels, reduction)
        self.freq_att = FreqAttention(channels, reduction)
        self.norm = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        return self.norm(x + self.channel_att(x) + self.freq_att(x))


class ComplexMaskNet(nn.Module):
    """
    2D卷积降噪网络
    输入: (B, F, T, 6) 合并实虚部
    输出: (B, F, T, 2) 复数mask
    
    只使用: Conv2d, Linear, Attention, BatchNorm, Activation
    """
    def __init__(self, base_channels=12, use_attention=True):
        super().__init__()
        C = base_channels
        
        # 输入: (B, F, T, 6) -> (B, 6, F, T)
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, C, 3, padding=1),
            nn.BatchNorm2d(C), nn.GELU())
        if use_attention:
            self.att1 = AttentionBlock(C)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(C, C*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(C*2), nn.GELU())
        if use_attention:
            self.att2 = AttentionBlock(C*2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(C*2, C*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(C*4), nn.GELU())
        if use_attention:
            self.att3 = AttentionBlock(C*4)
        
        self.mid = nn.Sequential(
            nn.Conv2d(C*4, C*4, 3, padding=1),
            nn.BatchNorm2d(C*4), nn.GELU())
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(C*4, C*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(C*2), nn.GELU())
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(C*2, C, 4, stride=2, padding=1),
            nn.BatchNorm2d(C), nn.GELU())
        
        self.out = nn.Sequential(
            nn.Conv2d(C, 32, 1), nn.GELU(),
            nn.Conv2d(32, 2, 1), nn.Sigmoid())
        
        self.use_attention = use_attention
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, F, T, C = x.shape
        x = x.permute(0, 3, 1, 2)
        
        e1 = self.enc1(x)
        if self.use_attention: e1 = self.att1(e1)
        
        e2 = self.enc2(e1)
        if self.use_attention: e2 = self.att2(e2)
        
        e3 = self.enc3(e2)
        if self.use_attention: e3 = self.att3(e3)
        
        m = self.mid(e3)
        
        d3 = self.dec3(m)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = d3 + e2
        
        d2 = self.dec2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = d2 + e1
        
        out = self.out(d2)
        out = F.interpolate(out, size=(F, T), mode='bilinear', align_corners=False)
        
        return out.permute(0, 2, 3, 1)


# ============================================================================
# 进化算法模块
# ============================================================================

class EvolutionConfig:
    """进化算法配置"""
    population_size: int = 8
    elite_size: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    generations: int = 15
    epochs_per_gen: int = 2
    batch_size: int = 4
    num_frames: int = 100
    lr: float = 0.001
    eval_samples: int = 16


class EvolutionaryTrainer:
    """进化训练器"""
    
    def __init__(self, config: EvolutionConfig, device='cuda'):
        self.model_class = ComplexMaskNet
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.population = []
        self.fitness_history = []
        
    def initialize_population(self):
        """初始化种群"""
        print(f"Initializing population of {self.config.population_size}...")
        for i in range(self.config.population_size):
            model = self.model_class(base_channels=12, use_attention=True).to(self.device)
            
            # 不同初始化策略
            if i < self.config.population_size // 3:
                pass  # Kaiming (default)
            elif i < 2 * self.config.population_size // 3:
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight)
            else:
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.sparse_(m.weight, sparsity=0.4)
            
            self.population.append(model)
    
    def get_chromosome(self, model):
        """获取染色体（模型权重）"""
        return [p.data.cpu().numpy().copy() for p in model.parameters()]
    
    def set_chromosome(self, model, chromosome):
        """设置染色体"""
        for (name, param), chrom in zip(model.named_parameters(), chromosome):
            param.data = torch.from_numpy(chrom.reshape(param.shape)).to(param.device).float()
    
    def crossover(self, p1, p2):
        """均匀交叉"""
        child = []
        for g1, g2 in zip(p1, p2):
            mask = np.random.random(g1.shape) < 0.5
            child.append(np.where(mask, g1, g2))
        return child
    
    def mutate(self, chrom):
        """高斯变异"""
        mutated = []
        for layer in chrom:
            if np.random.random() < self.config.mutation_rate:
                noise = np.random.randn(*layer.shape).astype(np.float32)
                noise *= self.config.mutation_strength * (np.std(layer) + 1e-8)
                layer = layer + noise
            mutated.append(layer)
        return mutated
    
    def tournament_selection(self, fitnesses):
        """锦标赛选择"""
        indices = random.sample(range(len(self.population)), self.config.tournament_size)
        best = min(indices, key=lambda i: fitnesses[i])
        return self.population[best]
    
    def evaluate(self, model, dataloader):
        """评估适应度"""
        model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()
        
        count = 0
        with torch.no_grad():
            for x, y in dataloader:
                if count >= self.config.eval_samples:
                    break
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                loss = criterion(pred, y)
                total_loss += loss.item()
                count += x.size(0)
        
        return total_loss / max(count, 1)
    
    def train_epoch(self, model, dataloader):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config.lr)
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evolve(self, train_dataset, val_dataset=None):
        """执行进化"""
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                  shuffle=True, num_workers=0)
        
        if val_dataset is None:
            val_dataset = torch.utils.data.Subset(train_dataset, 
                                                   list(range(min(32, len(train_dataset)))))
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        self.initialize_population()
        
        best_model = None
        best_fitness = float('inf')
        
        for gen in range(self.config.generations):
            print(f"\n{'='*50}")
            print(f"Generation {gen + 1}/{self.config.generations}")
            print(f"{'='*50}")
            
            # 评估
            fitnesses = []
            for i, model in enumerate(self.population):
                fitness = self.evaluate(model, val_loader)
                fitnesses.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_model = copy.deepcopy(model)
                    print(f"  [NEW BEST] Model {i}: loss = {fitness:.6f}")
                else:
                    print(f"  Model {i}: loss = {fitness:.6f}")
            
            avg_fitness = np.mean(fitnesses)
            self.fitness_history.append((best_fitness, avg_fitness))
            print(f"  Best: {best_fitness:.6f}, Avg: {avg_fitness:.6f}")
            
            # 精英保留
            sorted_idx = np.argsort(fitnesses)
            elites = [self.population[i] for i in sorted_idx[:self.config.elite_size]]
            
            # 新种群
            new_pop = list(elites)
            
            while len(new_pop) < self.config.population_size:
                p1 = self.tournament_selection(fitnesses)
                p2 = self.tournament_selection(fitnesses)
                
                c1 = self.crossover(self.get_chromosome(p1), self.get_chromosome(p2))
                c2 = self.mutate(c1)
                
                child = self.model_class(base_channels=12, use_attention=True).to(self.device)
                self.set_chromosome(child, c2)
                
                # 对部分个体进行训练
                if len(new_pop) < self.config.population_size // 2:
                    self.train_epoch(child, train_loader)
                
                new_pop.append(child)
            
            self.population = new_pop
            self.config.lr *= 0.95
        
        return best_model, best_fitness


# ============================================================================
# 评估模块
# ============================================================================

def evaluate_model(model, test_dataset, device='cuda'):
    """评估模型效果"""
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    count = 0
    
    # 计算多个指标
    total_mse = 0.0
    total_mae = 0.0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            total_loss += criterion(pred, y).item()
            total_mse += F.mse_loss(pred, y).item() * x.size(0)
            total_mae += F.l1_loss(pred, y).item() * x.size(0)
            count += x.size(0)
    
    avg_loss = total_loss / count
    avg_mse = total_mse / count
    avg_mae = total_mae / count
    
    return {
        'mse': avg_mse,
        'mae': avg_mae,
        'loss': avg_loss
    }


def run_full_training(data_dir=None, generations=15):
    """完整训练流程：生成数据 -> 进化训练 -> 评估"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # 配置
    config = EvolutionConfig()
    config.generations = generations
    config.population_size = 8
    config.epochs_per_gen = 2
    config.batch_size = 4
    
    # 1. 准备数据
    print("\n" + "="*50)
    print("Step 1: Preparing data")
    print("="*50)
    
    dataset = SpeechNoiseDataset(
        data_dir=data_dir, 
        num_samples=200,  # 生成200个样本用于训练
        num_frames=config.num_frames
    )
    print(f"Dataset: {len(dataset)} samples")
    print(f"Input shape: (257, {config.num_frames}, 6)")
    
    # 分割训练/验证/测试
    n = len(dataset)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, train_size + val_size)))
    test_dataset = torch.utils.data.Subset(dataset, list(range(train_size + val_size, n)))
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 2. 进化训练
    print("\n" + "="*50)
    print("Step 2: Evolutionary Training")
    print("="*50)
    
    trainer = EvolutionaryTrainer(config, device=str(device))
    best_model, best_fitness = trainer.evolve(train_dataset, val_dataset)
    
    print(f"\nEvolution complete! Best fitness: {best_fitness:.6f}")
    
    # 3. 评估
    print("\n" + "="*50)
    print("Step 3: Evaluation")
    print("="*50)
    
    metrics = evaluate_model(best_model, test_dataset, device)
    
    print(f"\nTest Results:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  Loss: {metrics['loss']:.6f}")
    
    # 保存模型
    save_path = Path('checkpoints')
    save_path.mkdir(exist_ok=True)
    torch.save(best_model.state_dict(), save_path / 'ea_model.pth')
    print(f"\nModel saved to {save_path / 'ea_model.pth'}")
    
    # 保存训练历史
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump({
            'best_fitness': best_fitness,
            'test_metrics': metrics,
            'history': [{'best': h[0], 'avg': h[1]} for h in trainer.fitness_history]
        }, f, indent=2)
    
    print(f"History saved to {save_path / 'training_history.json'}")
    
    return best_model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evolutionary Training for Lighting Speech Shield')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--generations', type=int, default=15)
    args = parser.parse_args()
    
    run_full_training(args.data_dir, args.generations)
