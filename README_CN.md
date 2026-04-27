# SAM3-LoRA：使用低秩适配进行高效微调

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**使用可训练参数减少99%的方式训练SAM3分割模型**

[快速开始](#快速开始) • [架构](#架构) • [训练](#训练) • [推理](#推理) • [示例](#真实世界示例-混凝土裂缝检测) • [配置](#配置)

</div>

---

## 概述

使用**LoRA（低秩适配）**微调SAM3（Segment Anything Model 3）——一种参数高效的方法，可将可训练参数从100%减少到约1%，同时保持性能。

### 最近更新

**2026-02-03**:
- **修复了训练/验证中的多类别分类分配错误**
- 之前，包含多个类别的图像会错误地将所有对象分配给众数（最频繁）类别
- 现在为每个类别创建单独的查询，将每个对象映射到其实际类别
- 受影响文件：`train_sam3_lora_native.py`、`train_sam3_lora_with_categories.py`、`validate_sam3_lora.py`

**2026-01-31**:
- **在`infer_sam.py`中将`--no-boxes`替换为`--boundingbox`选项**
- 新的`--boundingbox True/False`标志用于显式控制边界框（默认：False）
- 更新了README文档和推理示例

**2026-01-04**:
- **添加了使用DistributedDataParallel（DDP）的多GPU训练支持**
- 新的`--device`参数用于轻松选择GPU：`--device 0 1 2 3`
- 指定多个GPU时自动启动torchrun
- 跨GPU的批量大小线性缩放

### 为什么使用这个？

- ✅ **在消费级GPU上训练**：16GB显存而不是80GB
- ✅ **小型检查点**：10-50MB LoRA权重 vs 3GB完整模型
- ✅ **快速迭代**：更少内存 = 更快训练
- ✅ **易于使用**：YAML配置 + 简单CLI
- ✅ **生产就绪**：完整的训练 + 推理流水线
- ✅ **真实应用**：裂缝检测、缺陷检查等
- ✅ **多GPU支持**：使用`--device 0 1 2 3`跨多个GPU扩展训练

### 什么是LoRA？

LoRA不是微调所有模型权重，而是注入小的可训练矩阵：
```
W' = W_frozen + B×A  （其中秩 << 模型维度）
```

**结果**：只有约1%的参数需要训练！

### 架构

SAM3-LoRA将低秩适配应用于SAM3架构的关键组件：

<div align="center">
<img src="asset/Screenshot 2568-12-06 at 07.00.16.png" alt="SAM3 Architecture with LoRA" width="900">
<br>
<em>具有完整LoRA适配的SAM3模型架构</em>
</div>

<br>

**LoRA适配器应用于：**

| 组件 | 描述 | LoRA影响 |
|-----------|-------------|-------------|
| **视觉编码器（ViT）** | 从输入图像中提取视觉特征 | 高 - 主要特征学习 |
| **文本编码器** | 处理文本提示以进行引导分割 | 中 - 语义理解 |
| **几何编码器** | 处理几何提示（框、点） | 中 - 空间推理 |
| **DETR编码器** | 目标检测的Transformer编码器 | 高 - 场景理解 |
| **DETR解码器** | 目标查询的Transformer解码器 | 高 - 目标定位 |
| **掩码解码器** | 生成分割掩码 | 高 - 精细分割 |

**数据流：**
1. **输入**：图像 + 文本/几何提示
2. **编码**：多个编码器处理不同模态
3. **转换**：DETR编码器-解码器精炼表示
4. **输出**：高质量分割掩码

**LoRA优势：**
- ✅ 只有约1%参数可训练（冻结基础 + 小适配器）
- ✅ 适配器可为不同任务交换
- ✅ 保留原始模型权重
- ✅ 高效存储（10-50MB vs 3GB完整模型）

---

## 安装

### 先决条件

在安装之前，您需要：

1. **在Hugging Face上请求SAM3访问**
   - 访问 [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3)
   - 点击"Request Access"并接受许可条款
   - 等待批准（通常即时到几小时）

2. **获取您的Hugging Face令牌**
   - 访问 [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens)
   - 创建新令牌或使用现有令牌
   - 复制令牌（下一步需要）

### 安装

```bash
# 克隆存储库
git clone https://github.com/yourusername/sam3_lora.git
cd SAM3_LoRA

# 安装依赖
pip install -e .

# 登录Hugging Face
hf auth login
# 提示时粘贴您的令牌
```

**替代登录方法：**
```bash
# 或将令牌设置为环境变量
export HF_TOKEN="your_token_here"
```

**要求**：Python 3.8+、PyTorch 2.0+、CUDA（可选）、具有SAM3访问权限的Hugging Face账户

### 验证

验证您的设置是否完成：

```bash
# 测试Hugging Face登录
huggingface-cli whoami

# 测试SAM3访问（不应给出访问错误）
python3 -c "from transformers import AutoModel; print('✓ SAM3 accessible')"
```

如果看到错误，请查看[故障排除](#故障排除)部分。

---

## 快速开始

> **⚠️ 重要**：在继续之前，请确保您已完成[安装](#安装)步骤，包括Hugging Face登录。

**示例结果**：训练一个模型来检测混凝土裂缝，仅使用约1%的可训练参数！

<div align="center">
<img src="asset/output.png" alt="Example: Concrete Crack Detection" width="600">
<br>
<em>检测："混凝土裂缝"，置信度0.32 • 精确分割掩码</em>
</div>

<br>

### 1. 准备您的数据

以**COCO格式**组织您的数据集，每个分割有一个注释文件：

```
data/
├── train/                    # 必需
│   ├── img001.jpg
│   ├── img002.jpg
│   └── _annotations.coco.json
├── valid/                    # 可选但推荐
│   ├── img001.jpg
│   ├── img002.jpg
│   └── _annotations.coco.json
└── test/                     # 可选
    ├── img001.jpg
    └── _annotations.coco.json
```

> **注意**：验证数据（`data/valid/`）是**可选的**，但强烈推荐用于监控训练进度和防止过拟合。

**COCO注释格式**（`_annotations.coco.json`）：
```json
{
  "images": [
    {
      "id": 0,
      "file_name": "img001.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "defect"}
  ]
}
```

**支持的分割格式：**
- **多边形**：`"segmentation": [[x1, y1, x2, y2, ...]]`（多边形列表）
- **RLE**：`"segmentation": {"counts": "...", "size": [h, w]}`（游程编码）

### 2. 训练您的模型

```bash
# 使用默认配置训练
python3 train_sam3_lora_native.py

# 或指定自定义配置
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml
```

**预期输出：**
```
构建SAM3模型...
应用LoRA...
已应用LoRA到64个模块
可训练参数：11,796,480 (1.38%)

从/workspace/data2加载训练数据...
加载的COCO数据集：训练分割
  图像：778
  注释：1631
  类别：{0: 'CRACKS', 1: 'CRACKS', 2: 'JOINT', 3: 'LOCATION', 4: 'MARKING'}

从/workspace/data2加载验证数据...
加载的COCO数据集：验证分割
  图像：152
  注释：298
找到验证数据：152张图像
开始训练100个周期...
训练样本：778，验证样本：152

周期1：100%|████████| 98/98 [07:47<00:00, loss=140]
验证：100%|████████| 19/19 [00:32<00:00, val_loss=23.7]

周期1/100 - 训练损失：156.234567，验证损失：17.032280
✓ 保存新最佳模型（val_loss：17.032280）

周期2：100%|████████| 98/98 [07:24<00:00, loss=167]
验证：100%|████████| 19/19 [00:31<00:00, val_loss=20.1]

周期2/100 - 训练损失：142.891234，验证损失：15.641912
✓ 保存新最佳模型（val_loss：15.641912）
...
```

**验证策略（遵循SAM3）：**
- **训练期间**：仅计算验证**损失**（快速，无NMS或指标）
- **训练后**：运行`validate_sam3_lora.py`进行完整指标（mAP、cgF1）和NMS过滤

这种方法**显著加快训练速度**，同时仍通过验证损失监控过拟合。

### 3. 运行推理

```bash
# 基本推理（自动使用最佳模型）
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image test_image.jpg \
  --output predictions.png

# 使用文本提示以获得更好准确性
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image test_image.jpg \
  --prompt "yellow school bus" \
  --output predictions.png

# 多个提示以检测不同对象
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image test_image.jpg \
  --prompt "crack" "defect" "damage" \
  --output predictions.png
```

---

## 训练

### 基本训练
```bash
# 使用默认配置（单GPU）
python3 train_sam3_lora_native.py

# 或指定自定义配置
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml
```

### 多GPU训练

使用`--device`参数在多个GPU上训练。脚本自动处理分布式训练设置。

```bash
# 单GPU（默认 - GPU 0）
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml

# 单GPU（特定GPU）
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 1

# 多GPU（2个GPU）
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1

# 多GPU（4个GPU）
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1 2 3

# 多GPU（特定GPU，例如0、2、3）
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 2 3
```

**多GPU特性：**
- ✅ 指定多个GPU时自动启动`torchrun`
- ✅ 使用DistributedDataParallel（DDP）进行高效梯度同步
- ✅ 使用DistributedSampler进行适当的数据分片
- ✅ 跨所有GPU同步验证损失
- ✅ 仅在rank 0上保存模型（无文件冲突）

**有效批量大小：**
使用多GPU时，您的有效批量大小线性缩放：
```
有效批量大小 = 批量大小 × GPU数量
```

| 配置批量大小 | GPU数量 | 有效批量大小 |
|-------------------|------|---------------------|
| 4 | 1 | 4 |
| 4 | 2 | 8 |
| 4 | 4 | 16 |

**预期输出（多GPU）：**
```
在GPU上启动分布式训练：[0, 1]
进程数：2
启用多GPU训练，使用2个GPU
构建SAM3模型...
应用LoRA...
可训练参数：11,796,480 (1.38%)
模型使用DistributedDataParallel包装
有效批量大小：4 x 2 = 8
开始训练100个周期...
```

### 自定义配置

创建配置文件（例如`configs/my_config.yaml`）：

```yaml
lora:
  rank: 16                    # LoRA秩（越高 = 容量越大）
  alpha: 32                   # 缩放因子（通常为2×rank）
  dropout: 0.1                # 正则化dropout
  target_modules:             # 要适配的层
    - "q_proj"                # 查询投影
    - "k_proj"                # 键投影
    - "v_proj"                # 值投影
    - "fc1"                   # MLP层1
    - "fc2"                   # MLP层2

  # 将LoRA应用于哪些模型组件
  apply_to_vision_encoder: true
  apply_to_mask_decoder: true
  apply_to_detr_encoder: false
  apply_to_detr_decoder: false

training:
  data_dir: "/path/to/data"   # 包含train/valid/test文件夹的根目录
  batch_size: 8               # 根据GPU内存调整
  num_epochs: 100             # 训练周期数
  learning_rate: 5e-5         # 学习率（SAM3微调推荐5e-5）
  weight_decay: 0.01          # 权重衰减
  gradient_accumulation_steps: 8  # 有效批量 = 批量大小 × 累积步数

output:
  output_dir: "outputs/my_model"
```

**重要说明：**
- **类别感知提示**：训练自动使用类别名称作为文本提示（例如"crack"、"joint"），从COCO注释中提取
- 每个训练图像都使用其特定对象类别（小写）进行提示
- 这种方法通过使用任务特定词汇同时利用SAM3的预训练文本理解来提高性能

然后训练：
```bash
python3 train_sam3_lora_native.py --config configs/my_config.yaml
```

### 模型检查点

训练期间自动保存两个模型：
- **`best_lora_weights.pt`**：基于验证损失的最佳模型（仅在验证损失改善时保存）
- **`last_lora_weights.pt`**：最后一个周期的模型（每个周期后保存）

**有验证数据**：训练仅监控验证**损失**（快速）。当验证损失减少时保存最佳模型。

**无验证数据**：训练正常继续但将最后一个周期保存为两个文件。您将看到：
```
⚠️ 未找到验证数据 - 无验证训练
...
ℹ️ 无验证数据 - 考虑添加data/valid/以获得更好的模型选择
```

---

## 验证

### 概述

SAM3-LoRA使用**两阶段验证方法**遵循SAM3的原始设计：

1. **训练期间**：仅计算验证**损失**（快速，无昂贵指标）
2. **训练后**：运行完整评估，包括mAP、cgF1指标和NMS过滤

这种方法**显著加快训练速度**，同时仍通过验证损失监控过拟合。

### 快速验证

训练完成后，评估您的模型：

```bash
# 验证LoRA适配模型（自动使用最佳模型）
python3 validate_sam3_lora.py \
  --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_full/best_lora_weights.pt \
  --val_data_dir /workspace/data2/valid

# 在测试集上评估
python3 validate_sam3_lora.py \
  --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_full/best_lora_weights.pt \
  --val_data_dir /workspace/data2/test

# 基线：使用原始SAM3模型（无LoRA）进行比较
python3 validate_sam3_lora.py \
  --val_data_dir /workspace/data2/valid \
  --use-base-model
```

**预期输出：**
```
运行SAM3 LoRA验证
构建SAM3模型...
从outputs/sam3_lora_full/best_lora_weights.pt加载LoRA权重
加载的COCO数据集：验证分割
  图像：152
  注释：298

处理中：100%|████████| 152/152 [02:15<00:00]

验证结果：
================================================================================
  总预测：946（经过NMS从1353个初始检测中）
  总真实值：298

COCO评估指标：
--------------------------------------------------------------------------------
  mAP (IoU 0.50:0.95)：0.245
  mAP@50 (IoU 0.50)：0.287
  mAP@75 (IoU 0.75)：0.198

类别无关F1分数：
--------------------------------------------------------------------------------
  cgF1 (平均)：0.135
  cgF1@50：0.149
  cgF1@75：0.089
================================================================================
```

### 验证指标解释

| 指标 | 描述 | 良好值 | 优秀值 |
|--------|-------------|------------|-----------------|
| **mAP (0.50:0.95)** | 跨IoU阈值0.5到0.95的平均精度 | > 0.30 | > 0.50 |
| **mAP@50** | IoU阈值0.50的精度（较宽松） | > 0.40 | > 0.70 |
| **mAP@75** | IoU阈值0.75的精度（较严格） | > 0.25 | > 0.45 |
| **cgF1** | 概念级F1（SAM3的主要指标） | > 0.25 | > 0.50 |
| **cgF1@50** | IoU 0.50的cgF1 | > 0.30 | > 0.60 |
| **cgF1@75** | IoU 0.75的cgF1 | > 0.15 | > 0.35 |

**理解指标：**
- **mAP**：标准COCO指标 - 越高越好，惩罚过度/不足分割
- **cgF1**：SAM3的概念级指标 - 平衡概念而非单个实例的精确率和召回率

---

## 推理

使用您训练的LoRA模型在新图像上运行推理。`infer_sam.py`脚本基于官方SAM3模式，支持**多个文本提示**和**NMS过滤**以获得干净、非重叠的检测。

### 命令行

```bash
# 基本推理（自动使用最佳模型）
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --output predictions.png

# 使用文本提示（推荐以获得更好准确性）
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --prompt "yellow school bus" \
  --output predictions.png

# 多个提示以检测不同对象类型
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image street_scene.jpg \
  --prompt "car" "person" "bus" \
  --output segmentation.png

# 使用最后一个周期模型代替
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_full/last_lora_weights.pt \
  --image path/to/image.jpg \
  --prompt "person with red backpack" \
  --output predictions.png

# 使用自定义置信度阈值
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --prompt "building" \
  --threshold 0.3 \
  --output predictions.png

# 调整NMS以减少重叠框
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --prompt "seal" \
  --threshold 0.3 \
  --nms-iou 0.3 \
  --output clean_detections.png

# 带边界框
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --prompt "crack" \
  --boundingbox True \
  --output with_boxes.png
```

### NMS（非极大值抑制）

NMS移除重叠的边界框以产生干净的可视化。没有NMS，您可能会看到许多重叠框的网格状模式。

```bash
# 默认NMS IoU = 0.5（适用于大多数情况）
python3 infer_sam.py --config configs/full_lora_config.yaml --image test.jpg --prompt "object"

# 更激进的NMS（更少框，更少重叠）
python3 infer_sam.py --config configs/full_lora_config.yaml --image test.jpg --prompt "object" --nms-iou 0.3

# 较不激进的NMS（保留更多重叠检测）
python3 infer_sam.py --config configs/full_lora_config.yaml --image test.jpg --prompt "object" --nms-iou 0.7
```

**NMS IoU指南：**
| 值 | 效果 | 使用场景 |
|-------|--------|----------|
| 0.3 | 激进过滤 | 每个区域单个对象，干净输出 |
| 0.5 | 平衡（默认） | 大多数一般使用场景 |
| 0.7 | 保留更多框 | 密集对象，重叠实例 |

### 文本提示

文本提示帮助引导模型更准确地分割特定对象。**新功能**：您现在可以在单个命令中使用多个提示！

**单提示示例：**
- `"yellow school bus"` - 特定颜色和对象类型
- `"person wearing red hat"` - 具有独特特征的对象
- `"car"` - 简单、清晰的对象类型
- `"crack"` - 用于缺陷检测
- `"building with glass windows"` - 具有区分特征的对象

**多提示示例：**
```bash
# 检测不同缺陷类型
--prompt "crack" "spalling" "corrosion"

# 检测街道场景中的多个对象
--prompt "car" "person" "traffic sign"
```

**更好提示的技巧：**
- 具体但简洁
- 相关时包括独特颜色或特征
- 使用自然语言描述
- 对于多个提示，按重要性从高到低排序
- 使词汇与训练数据匹配

### 推理参数

| 参数 | 描述 | 示例 | 默认 |
|-----------|-------------|---------|---------|
| `--config` | 训练配置文件路径 | `configs/full_lora_config.yaml` | 必需 |
| `--weights` | LoRA权重路径（可选） | `outputs/sam3_lora_full/best_lora_weights.pt` | 自动检测 |
| `--image` | 输入图像路径 | `test_image.jpg` | 必需 |
| `--prompt` | 一个或多个文本提示 | `"crack"` 或 `"crack" "defect"` | `"object"` |
| `--output` | 输出可视化路径 | `predictions.png` | `output.png` |
| `--threshold` | 置信度阈值（0.0-1.0） | `0.3` | `0.5` |
| `--nms-iou` | NMS IoU阈值（越低 = 框越少） | `0.3` | `0.5` |
| `--resolution` | 输入分辨率 | `1008` | `1008` |
| `--boundingbox` | 显示边界框（True/False） | `True` | `False` |
| `--no-masks` | 不显示分割掩码 | - | False |

### Python API

```python
from infer_sam import SAM3LoRAInference

# 使用NMS初始化推理引擎
inferencer = SAM3LoRAInference(
    config_path="configs/full_lora_config.yaml",
    weights_path="outputs/sam3_lora_full/best_lora_weights.pt",
    detection_threshold=0.5,
    nms_iou_threshold=0.5  # 调整以获得更干净输出（越低 = 框越少）
)

# 使用单文本提示运行预测
predictions = inferencer.predict(
    image_path="image.jpg",
    text_prompts=["yellow school bus"]
)

# 使用多个文本提示运行预测
predictions = inferencer.predict(
    image_path="image.jpg",
    text_prompts=["crack", "defect", "damage"]
)

# 可视化结果
inferencer.visualize(
    predictions,
    output_path="output.png",
    show_boxes=True,
    show_masks=True
)

# 访问每个提示的预测（已应用NMS）
for idx, prompt in enumerate(["crack", "defect"]):
    result = predictions[idx]
    print(f"提示 '{result['prompt']}':")
    print(f"  检测数：{result['num_detections']}")
    if result['num_detections'] > 0:
        print(f"  框：{result['boxes'].shape}")      # [N, 4] xyxy格式
        print(f"  分数：{result['scores'].shape}")    # [N]
        print(f"  掩码：{result['masks'].shape}")      # [N, H, W]
```

---

## 配置

### LoRA参数

| 参数 | 描述 | 典型值 |
|-----------|-------------|----------------|
| `rank` | LoRA秩（瓶颈维度） | 4, 8, 16, 32 |
| `alpha` | 缩放因子 | 2×rank（例如，rank=8时为16） |
| `dropout` | Dropout概率 | 0.0 - 0.1 |
| `target_modules` | 要适配的层类型 | q_proj, k_proj, v_proj, fc1, fc2 |

### 组件标志

| 标志 | 描述 | 何时启用 |
|------|-------------|----------------|
| `apply_to_vision_encoder` | 视觉骨干 | 总是（主要特征提取器） |
| `apply_to_mask_decoder` | 掩码生成 | 推荐用于分割 |
| `apply_to_detr_encoder` | 目标检测编码器 | 用于复杂场景 |
| `apply_to_detr_decoder` | 目标检测解码器 | 用于复杂场景 |
| `apply_to_text_encoder` | 文本理解 | 用于基于文本的提示 |

### 预设配置

**最小（最快，最低内存）**
```yaml
lora:
  rank: 4
  alpha: 8
  target_modules: ["q_proj", "v_proj"]
  apply_to_vision_encoder: true
  # 其他所有：false
```

**平衡（推荐）**
```yaml
lora:
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "fc1", "fc2"]
  apply_to_vision_encoder: true
  apply_to_mask_decoder: true
  # 其他：false
```

**最大（最佳性能）**
```yaml
lora:
  rank: 32
  alpha: 64
  target_modules: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
  apply_to_vision_encoder: true
  apply_to_mask_decoder: true
  apply_to_detr_encoder: true
  apply_to_detr_decoder: true
```

---

## 真实世界示例：混凝土裂缝检测

SAM3-LoRA在检测结构缺陷（如混凝土裂缝）方面表现出色。这是一个真实示例：

<div align="center">
<img src="asset/output.png" alt="Concrete Crack Detection" width="800">
</div>

**检测结果：**
- **提示**："concrete crack"
- **置信度**：0.32（使用阈值0.3）
- **分割**：精确掩码跟随裂缝模式
- **应用**：基础设施检查、结构健康监测

**运行此示例：**
```bash
# 检测混凝土结构中的裂缝
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/concrete.jpg \
  --prompt "concrete crack" \
  --threshold 0.3 \
  --output crack_detection.png

# 检测多种缺陷类型
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/concrete.jpg \
  --prompt "crack" "spalling" "corrosion" \
  --threshold 0.3 \
  --output defect_analysis.png
```

**使用场景：**
- 🏗️ 土木工程检查
- 🌉 桥梁和基础设施监测
- 🏢 建筑维护
- 🛣️ 路面分析
- 🏭 工业设施评估

---

## 测试结果：道路损坏检测

我们评估了微调的SAM3-LoRA模型在坑洞检测上的表现，并将其与未微调的基础SAM3模型进行比较。

### 验证指标比较

<div align="center">
<img src="asset/Screenshot 2568-12-10 at 08.20.20.png" alt="Validation Metrics" width="800">
<br>
<em>验证性能：LoRA微调模型 vs 基础SAM3模型</em>
</div>

<br>

**关键发现：**
- **LoRA模型（微调）**：显示改进的精确度和更好的多坑洞检测
- **基础模型**：倾向于产生更多误报并遗漏一些实例
- **数据集**：路面坑洞检测（data3）

### 视觉比较

<div align="center">
<img src="asset/combined_comparison_all.jpg" alt="Visual Comparison" width="900">
<br>
<em>并排比较：真实值（绿色）| LoRA模型（红色）| 基础模型（蓝色）</em>
</div>

<br>

**视觉结果观察：**

| 图像 | 真实值 | LoRA模型 | 基础模型 | 分析 |
|-------|--------------|------------|------------|----------|
| **img_0034** | 1个坑洞 | 1个检测 ✓ | 5个检测 ✗ | LoRA完美匹配真实值，基础有4个误报 |
| **img_0001** | 1个坑洞 | 1个检测 ✓ | 1个检测 ✓ | 两个模型表现良好 |
| **img_0080** | 1个坑洞 | 2个检测 ~ | 2个检测 ~ | 两个都有1个误报 |
| **img_0070** | 1个坑洞 | 1个检测 ✓ | 1个检测 ✓ | 两个模型表现良好 |
| **img_0060** | 4个坑洞 | 4个检测 ✓ | 2个检测 ✗ | LoRA找到所有实例，基础遗漏2个 |

**总结：**
- **LoRA模型**：5个中有3个完美匹配，多实例图像上召回率更好
- **基础模型**：5个中有2个完美匹配，处理多实例和误报有困难
- **总体**：使用LoRA微调显著提高了领域特定任务的检测准确性

**训练细节：**
- **提示**："pothole"（从COCO类别名称自动检测）
- **架构**：完整LoRA适配（视觉、文本、DETR编码器/解码器）
- **数据集**：具有COCO格式注释的道路损坏图像
- **阈值**：两个模型均为0.5置信度

---

## 项目结构

```
sam3_lora/
├── configs/
│   └── full_lora_config.yaml      # 默认训练配置
├── data/                          # COCO格式数据集
│   ├── train/
│   │   ├── img001.jpg             # 训练图像
│   │   ├── img002.jpg
│   │   └── _annotations.coco.json # COCO注释
│   ├── valid/
│   │   ├── img001.jpg             # 验证图像
│   │   ├── img002.jpg
│   │   └── _annotations.coco.json # COCO注释
│   └── test/
│       ├── img001.jpg             # 测试图像（可选）
│       └── _annotations.coco.json # COCO注释
├── outputs/
│   └── sam3_lora_full/
│       ├── best_lora_weights.pt   # 最佳模型（最低验证损失）
│       └── last_lora_weights.pt   # 最后一个周期模型
├── sam3/                          # SAM3模型库
├── lora_layers.py                 # LoRA实现
├── train_sam3_lora_native.py      # 训练脚本（仅计算验证损失）
├── validate_sam3_lora.py          # 完整评估脚本（mAP、cgF1、NMS）
├── validate_single_image.py       # 单图像验证与可视化
├── infer_sam.py                   # 推理脚本（推荐）
├── inference_lora.py              # 遗留推理脚本
├── README_INFERENCE.md            # 详细推理指南
└── README.md                      # 此文件
```

---

## 故障排除

### 常见问题

**1. Hugging Face认证错误**
```
错误：访问facebook/sam3被拒绝
```
**解决方案：**
- 确保您已在https://huggingface.co/facebook/sam3请求访问
- 等待批准（检查您的电子邮件）
- 运行`huggingface-cli login`并粘贴您的令牌
- 或设置：`export HF_TOKEN="your_token"`

**2. 导入错误**
```bash
# 确保包已安装
pip install -e .
```

**3. CUDA内存不足**
```yaml
# 在配置中减少批量大小和秩
training:
  batch_size: 1

lora:
  rank: 4
```

**4. 损失非常低（< 0.001）**
- 模型可能过拟合
- 减少LoRA秩
- 添加更多dropout
- 检查基础模型是否正确冻结

**5. 损失不减少**
- 增加学习率
- 增加LoRA秩
- 训练更多周期
- 检查数据质量

**6. 可训练参数数量错误**
```
预期：~0.5-2%（对于秩4-16）
如果看到63%：基础模型未冻结（最新版本已修复此错误）
```

**7. 无验证数据**
```
⚠️ 未找到验证数据 - 无验证训练
```
**解决方案：**
- 创建`data/valid/`目录，结构与`data/train/`相同
- 拆分您的数据：约80%训练，约20%验证
- 没有验证数据训练也能工作，但您不会看到验证指标

**8. 注释格式错误**
```
FileNotFoundError：未找到COCO注释文件：/path/to/data/train/_annotations.coco.json
```
**解决方案：**
- 确保您的数据是COCO格式，每个分割文件夹中有`_annotations.coco.json`
- 每个分割（train/valid/test）需要自己的注释文件
- 图像应与注释文件在同一目录
- 支持的分割格式：多边形列表或RLE字典