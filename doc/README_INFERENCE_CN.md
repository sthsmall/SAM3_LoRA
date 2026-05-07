# SAM3 + LoRA 推理指南

## 概述

本项目为SAM3 LoRA微调提供了两个推理脚本：

1. **`infer_sam.py`** - 基于官方SAM3模式的新脚本（推荐）
2. **`inference_lora.py`** - 原始简单推理脚本

## `infer_sam.py`（推荐）

基于官方SAM3批量推理模式，具有多项改进：

### 特性
- ✅ 支持多个文本提示
- ✅ 使用官方SAM3数据结构和变换
- ✅ 透明的手动后处理
- ✅ 正确的坐标处理（边界框限制在图像边界内）
- ✅ 多个提示的颜色编码可视化
- ✅ 自动检测最佳权重

### 使用方法

#### 单提示
```bash
python3 infer_sam.py \
    --config configs/full_lora_config.yaml \
    --image path/to/image.jpg \
    --prompt "crack" \
    --output output.png \
    --threshold 0.3
```

#### 多提示
```bash
python3 infer_sam.py \
    --config configs/full_lora_config.yaml \
    --image path/to/image.jpg \
    --prompt "crack" "defect" "damage" \
    --output output.png \
    --threshold 0.3
```

### 参数说明

- `--config`：训练配置YAML文件路径（必需）
- `--weights`：LoRA权重路径（可选，从配置自动检测）
- `--image`：输入图像路径（必需）
- `--prompt`：一个或多个文本提示（默认："object"）
- `--output`：输出可视化路径（默认："output.png"）
- `--threshold`：检测置信度阈值（默认：0.5）
- `--resolution`：输入分辨率（默认：1008）
- `--no-boxes`：不显示边界框
- `--no-masks`：不显示分割掩码

### 示例

**低阈值裂缝检测：**
```bash
python3 infer_sam.py \
    --config configs/full_lora_config.yaml \
    --image data/test/crack_image.jpg \
    --prompt "crack" \
    --threshold 0.3 \
    --output crack_detection.png
```

**多种缺陷类型：**
```bash
python3 infer_sam.py \
    --config configs/full_lora_config.yaml \
    --image data/test/defect_image.jpg \
    --prompt "crack" "spalling" "corrosion" \
    --threshold 0.4 \
    --output defect_analysis.png
```

**仅掩码可视化：**
```bash
python3 infer_sam.py \
    --config configs/full_lora_config.yaml \
    --image data/test/image.jpg \
    --prompt "crack" \
    --no-boxes \
    --output mask_only.png
```

## `inference_lora.py`（旧版）

具有基本功能的简单推理脚本。

### 使用方法
```bash
python3 inference_lora.py \
    --config configs/full_lora_config.yaml \
    --weights outputs/sam3_lora_full/best_lora_weights.pt \
    --image path/to/image.jpg \
    --prompt "crack" \
    --output output.png \
    --threshold 0.5
```

## 输出格式

两个脚本都会生成：

1. **可视化图像**：显示检测到的对象，包括：
   - 边界框（按提示着色）
   - 分割掩码（半透明叠加）
   - 置信度分数
   - 提示标签

2. **控制台摘要**：
   ```
   📊 摘要:
      提示 'crack': 1个检测
         最大置信度: 0.320
      提示 'damage': 3个检测
         最大置信度: 0.401
   ```

## 技巧

1. **阈值选择**：
   - 较低阈值（0.3）：更多检测，可能包含误报
   - 较高阈值（0.6）：更少检测，更高精度
   - 默认（0.5）：平衡方法

2. **提示工程**：
   - 要具体："concrete crack" vs "crack"
   - 尝试变化："defect"、"damage"、"deterioration"
   - 多个提示可以捕捉不同方面

3. **性能**：
   - 第一次推理因模型编译而较慢
   - 后续推理快得多
   - 分辨率影响质量和速度

## 故障排除

**问题**：未找到检测结果
- 尝试降低阈值（--threshold 0.3）
- 尝试不同的提示变化
- 检查LoRA权重是否正确加载

**问题**：边界框超出图像
- 这在`infer_sam.py`中已修复（边界框自动限制）
- 如果使用`inference_lora.py`，请更新到最新版本

**问题**：内存不足
- 降低分辨率（--resolution 512）
- 使用CPU而非GPU（在代码中修改设备）

## 模型权重

脚本从配置的输出目录自动检测最佳LoRA权重：
- 默认位置：`outputs/sam3_lora_full/best_lora_weights.pt`
- 如有需要，使用`--weights`参数覆盖

## 架构

`infer_sam.py`遵循官方SAM3推理流程：

1. **图像加载**：PIL图像 → SAM3数据点
2. **变换**：调整大小（1008x1008）→ 归一化
3. **批处理**：使用官方整理器整理
4. **推理**：通过SAM3 + LoRA进行前向传播
5. **后处理**：手动（透明和可控）
6. **可视化**：使用颜色编码叠加的Matplotlib