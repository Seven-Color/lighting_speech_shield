# Lighting Speech Shield 🛡️

轻量级实时语音降噪模型，基于卷积和注意力机制。

## 特性

- ✅ 仅使用卷积 + 注意力机制
- ✅ 流式处理（延迟 10-30ms）
- ✅ 低算力（<200 MFlops）
- ✅ 3 通道输入，频域 mask 输出
- ✅ STFT: 512 FFT, 160 hop

## 技术规格

| 参数 | 值 |
|------|-----|
| 采样率 | 16kHz |
| 输入通道 | 3 (麦克风阵列) |
| FFT 大小 | 512 |
| Hop 长度 | 160 (10ms) |
| 上下文帧 | 未来 1-3 帧 |
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

# 训练模型
python train.py

# 推理测试
python infer.py --input test.wav --output denoised.wav
```

## 模型架构

```
输入 (3 通道) → STFT → 复数频谱 → Backbone(Conv+Attention) → Mask → 降噪输出
```

## License

MIT
