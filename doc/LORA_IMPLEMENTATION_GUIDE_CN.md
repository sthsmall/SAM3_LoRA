# SAM3 LoRA 实现指南

本仓库为SAM3（Segment Anything Model 3）提供LoRA（低秩适配）微调，以高效适配到自定义分割任务。

## 特性

- **LoRA实现**：使用低秩适配的高效微调
- **灵活配置**：基于YAML的配置便于实验
- **可配置目标模块**：将LoRA应用于特定的transformer组件：
  - 注意力中的查询/键/值投影
  - 注意力中的输出投影
  - 前馈网络层
- **与SAM3训练流程兼容**：遵循与`sam3/train`相同的训练程序
- **内存高效**：只训练一小部分参数

## 目录结构

```
sam3_lora/
├── data/                          # COCO格式的训练数据
│   ├── train/                     # 训练图像和注释
│   ├── valid/                     # 验证图像和注释
│   └── test/                      # 测试图像和注释
├── src/
│   ├── lora/                      # LoRA实现
│   │   ├── lora_layer.py         # LoRA层定义
│   │   └── lora_utils.py         # LoRA注入工具
│   ├── data/                      # 数据加载工具
│   │   └── dataset.py            # 数据集和DataLoader
│   ├── train/                     # 训练逻辑
│   │   └── train_lora.py         # LoRA训练器
│   └── configs/                   # 配置文件
│       └── lora_config_example.yaml  # 示例配置
├── train.py                       # 主训练脚本
└── LORA_IMPLEMENTATION_GUIDE.md   # 本文件
```

## 安装

1. 安装SAM3依赖：
```bash
cd /workspace/sam3
pip install -e .
```

2. 安装额外依赖：
```bash
pip install tensorboard pyyaml
```

## 数据格式

训练数据应为COCO格式，结构如下：

```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
└── valid/
    ├── image1.jpg
    ├── image2.jpg
    └── _annotations.coco.json
```

注释文件应遵循COCO JSON格式：
```json
{
  "images": [
    {
      "image_id": 1,
      "file_name": "image1.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345,
      "segmentation": {...}
    }
  ]
}
```

## 配置

编辑`src/configs/lora_config_example.yaml`来自定义训练：

### 关键配置选项

#### LoRA参数
```yaml
lora:
  rank: 8                    # LoRA矩阵的秩（越高=容量越大）
  alpha: 16.0                # LoRA缩放因子（通常为2*rank）
  dropout: 0.1               # LoRA层的Dropout

  # 要应用LoRA的目标模块
  target_modules:
    - q_proj                 # 注意力中的查询投影
    - k_proj                 # 注意力中的键投影
    - v_proj                 # 注意力中的值投影
    - out_proj               # 注意力中的输出投影
    - linear1                # 第一个FFN层
    - linear2                # 第二个FFN层
```

**LoRA配置技巧：**
- **rank**：从4-8开始。更高值（16-32）提供更多容量但使用更多内存
- **alpha**：通常设置为2*rank或1*rank。控制LoRA更新的幅度
- **target_modules**：
  - 使用`["q_proj", "v_proj"]`进行最小训练（最快，内存最少）
  - 使用`["q_proj", "k_proj", "v_proj", "out_proj"]`仅限注意力
  - 使用所有模块以获得最大适配容量

#### 数据集配置
```yaml
dataset:
  train_img_folder: /workspace/sam3_lora/data/train
  train_ann_file: /workspace/sam3_lora/data/train/_annotations.coco.json
  val_img_folder: /workspace/sam3_lora/data/valid
  val_ann_file: /workspace/sam3_lora/data/valid/_annotations.coco.json
  resolution: 1008           # 输入分辨率
  max_ann_per_img: 200       # 每张图像的最大注释数
```

#### 训练配置
```yaml
training:
  max_epochs: 20
  batch_size: 2
  learning_rate: 1e-4
  gradient_accumulation_steps: 1
  use_amp: true              # 使用自动混合精度
  amp_dtype: bfloat16        # bfloat16或float16
```

## 使用方法

### 基本训练

```bash
python train.py --config src/configs/lora_config_example.yaml
```

### 恢复训练

```bash
python train.py --config src/configs/lora_config_example.yaml --resume experiments/checkpoints/best.pt
```

### 高级用法

#### 自定义LoRA配置

仅将LoRA应用于注意力层：

```yaml
lora:
  rank: 8
  alpha: 16.0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - out_proj
```

将LoRA应用于所有transformer组件：

```yaml
lora:
  rank: 16
  alpha: 32.0
  target_modules:
    - all
```

#### 使用分割掩码训练

```yaml
training:
  enable_segmentation: true  # 启用掩码预测
```

## 预期结果

使用LoRA微调：
- **可训练参数**：约1-5%的总模型参数
- **内存使用**：与全量微调相比显著减少
- **训练速度**：比全量微调快
- **性能**：在领域特定任务上与全量微调相当

## LoRA参数解释

### 秩（rank）

秩决定了LoRA低秩矩阵的维度。较高的秩意味着更多的容量来学习复杂的适应，但也意味着更多的可训练参数。

- **秩=4**：最小容量，最快训练，最少内存
- **秩=8**：平衡选择，适合大多数任务
- **秩=16**：更高容量，用于复杂任务
- **秩=32**：最大容量，最高内存需求

### Alpha

Alpha是缩放因子，用于控制LoRA更新的幅度。通常设置为rank的倍数（1x或2x）。

### Dropout

Dropout用于正则化，防止LoRA过拟合。通常设置为0.0-0.1。

### 目标模块

不同的目标模块影响不同类型的适应：

| 模块 | 效果 |
|------|------|
| q_proj, v_proj | 最小影响，仅查询和值适配 |
| k_proj, q_proj, v_proj | 注意力机制适配 |
| out_proj | 输出投影适配 |
| linear1, linear2 | FFN层适配，全容量 |

## 故障排除

### 内存不足

1. 减少batch_size
2. 降低LoRA rank
3. 使用较少的target_modules
4. 启用混合精度（use_amp: true）

### 损失不减少

1. 增加学习率
2. 增加LoRA rank
3. 训练更多周期
4. 检查数据质量

### 模型不收敛

1. 验证数据格式正确
2. 检查学习率设置
3. 尝试不同的rank值
4. 确保模型正确加载