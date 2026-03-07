# Lighting Speech Shield 🛡️

轻量级实时语音降噪模型，基于卷积和注意力机制。

## 特性

- ✅ 仅使用卷积 + 注意力机制
- ✅ 流式处理（延迟 10-30ms）
- ✅ 低算力（<200 MFlops）
- ✅ 3 通道输入，频域 mask 输出
- ✅ STFT: 512 FFT, 160 hop
- ✅ 混合精度训练 (AMP)
- ✅ 支持离线/流式推理

## 技术规格

| 参数 | 值 |
|------|-----|
| 采样率 | 16kHz |
| 输入通道 | 3 (麦克风阵列) |
| FFT 大小 | 512 |
| Hop 长度 | 160 (10ms) |
| 上下文帧 | 100 帧 |
| 算力 | ~200 MFlops |

## 安装

```bash
conda env create -f environment.yml
conda activate qi
```

## 快速开始

```bash
# 生成仿真数据集
python scripts/generate_dataset.py

# 训练模型 (优化版 - AMP + 梯度累积)
python train.py --epochs 10 --batch_size 2 --gradient_accumulation 4

# 推理测试 (离线)
python infer.py --input test.wav --output denoised.wav

# 推理测试 (流式 - 低延迟)
python infer.py --input test.wav --output denoised.wav --streaming

# 测试模型
python test_model_v2.py
```

## 训练参数

```bash
# 基本训练
python train.py --data_dir data/synthetic --epochs 10

# 大 batch 训练 (有效 batch = 2 * 4 = 8)
python train.py --batch_size 2 --gradient_accumulation 4

# 关闭 AMP
python train.py --no_amp
```

## 模型架构

```
输入 (3 通道) → STFT → 复数频谱 → Backbone(Conv+Attention) → Mask → 降噪输出
```

### 优化版本 v2
- 2D U-Net 结构
- 轻量通道注意力 (SE-like)
- 频率维度注意力
- 残差连接

## License

MIT
