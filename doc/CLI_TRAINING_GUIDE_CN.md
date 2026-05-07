# SAM3 LoRA - CLI训练指南

## 🚀 快速开始（现在可用的功能）

### 选项1：测试LoRA（推荐 - 立即可用）

使用简单transformer验证LoRA是否正常工作：

```bash
cd /workspace/sam3_lora
python3 test_lora_injection.py
```

**预期输出：**
```
============================================================
Testing LoRA Injection
============================================================
...
✓ Forward pass successful!
✓ Backward pass successful!
✓ All tests passed!
============================================================
```

---

## 📝 选项2：完整SAM3训练（需要设置）

### 步骤1：检查配置

编辑配置文件：
```bash
vim /workspace/sam3_lora/src/configs/lora_config_example.yaml
```

要验证的关键设置：
```yaml
paths:
  data_root: /workspace/sam3_lora/data
  experiment_log_dir: /workspace/sam3_lora/experiments
  bpe_path: /workspace/sam3/assets/bpe_simple_vocab_16e6.txt.gz
  sam3_checkpoint: null  # 或SAM3检查点路径

lora:
  rank: 8              # LoRA秩（4, 8, 16, 32）
  alpha: 16.0          # 缩放（通常为2*rank）
  dropout: 0.1
  target_modules:      # 哪些模块获得LoRA
    - q_proj
    - k_proj
    - v_proj
    - out_proj
    - linear1
    - linear2

training:
  max_epochs: 20
  batch_size: 2
  learning_rate: 1e-4
  use_amp: true
  amp_dtype: bfloat16
```

### 步骤2：基本训练命令

```bash
cd /workspace/sam3_lora

# 基本训练
python3 train.py --config src/configs/lora_config_example.yaml

# 选择GPU
CUDA_VISIBLE_DEVICES=0 python3 train.py --config src/configs/lora_config_example.yaml

# 指定设备
python3 train.py \
  --config src/configs/lora_config_example.yaml \
  --device cuda
```

### 步骤3：恢复训练

```bash
python3 train.py \
  --config src/configs/lora_config_example.yaml \
  --resume experiments/checkpoints/best.pt
```

---

## 🛠️ 当前问题及解决方案

### 问题：损失计算中的`NotImplementedError`

训练器的`_compute_loss()`方法尚未实现。

### 解决方案A：实现SAM3损失（推荐用于生产）

编辑`/workspace/sam3_lora/src/train/train_lora.py`：

```python
def _compute_loss(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
    """使用SAM3的损失函数计算损失。"""
    from sam3.train.loss.sam3_loss import Sam3LossWrapper

    # 使用SAM3的损失
    # 这是一个简化的示例 - 根据您的SAM3版本调整
    if hasattr(self, 'loss_fn'):
        return self.loss_fn(outputs, batch)
    else:
        # 后备：用于测试的简单虚拟损失
        if isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss']
        raise NotImplementedError("Loss function not configured")
```

### 解决方案B：简单演示训练循环

创建最小训练脚本用于测试：

```bash
cat > /workspace/sam3_lora/train_simple.py << 'EOF'
#!/usr/bin/env python3
"""
用于在不使用完整SAM3的情况下测试LoRA的简化训练脚本。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.lora.lora_utils import LoRAConfig, inject_lora_into_model, print_trainable_parameters
from src.data.dataset import LoRASAM3Dataset

def main():
    # 1. 创建简单模型（用于演示）
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True)
            self.head = nn.Linear(256, 1)

        def forward(self, x):
            x = self.encoder(x)
            return self.head(x.mean(dim=1))

    model = SimpleModel()

    # 2. 注入LoRA
    lora_config = LoRAConfig(rank=8, alpha=16.0, target_modules=["q_proj", "v_proj"])
    model = inject_lora_into_model(model, lora_config, verbose=True)
    print_trainable_parameters(model)

    # 3. 创建优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    # 4. 训练循环
    model.train()
    for epoch in range(5):
        # 虚拟数据
        x = torch.randn(4, 10, 256)
        y = torch.randn(4, 1)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # 5. 保存LoRA权重
    from src.lora.lora_utils import get_lora_state_dict
    lora_weights = get_lora_state_dict(model)
    torch.save(lora_weights, "demo_lora.pt")
    print("\n✓ Training complete! LoRA weights saved to demo_lora.pt")

if __name__ == "__main__":
    main()
EOF

python3 /workspace/sam3_lora/train_simple.py
```

---

## 🎯 替代方案：使用SAM3官方训练器配合LoRA

这是生产环境的推荐方法：

### 创建集成脚本

```bash
cat > /workspace/sam3_lora/train_with_sam3.py << 'EOF'
#!/usr/bin/env python3
"""
使用SAM3官方训练器配合LoRA训练SAM3。
"""

import torch
from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, apply_lora_to_model

def main():
    # 1. 构建SAM3模型
    model = build_sam3_image_model(
        device="cuda",
        load_from_HF=True
    )

    # 2. 配置LoRA
    lora_config = LoRAConfig(
        rank=8,
        alpha=16.0,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    )

    # 3. 注入LoRA
    model = apply_lora_to_model(model, lora_config)

    # 4. 继续使用SAM3官方训练器...
    # （此处添加您的训练逻辑）

if __name__ == "__main__":
    main()
EOF
```

---

## 🔧 常用命令参考

### 训练命令

```bash
# 基本训练
python3 train.py --config configs/base_config.yaml

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1 python3 train.py --config configs/base_config.yaml

# 使用不同的batch size
python3 train.py --config configs/base_config.yaml --batch_size 4

# 恢复中断的训练
python3 train.py --config configs/base_config.yaml --resume path/to/checkpoint.pt
```

### 推理命令

```bash
# 基本推理
python3 infer_sam.py --config configs/base_config.yaml --image test.jpg --prompt "object"

# 使用LoRA权重
python3 infer_sam.py --config configs/base_config.yaml --weights outputs/lora.pt --image test.jpg

# 多提示推理
python3 infer_sam.py --config configs/base_config.yaml --image test.jpg --prompt "car" "person"
```

### 数据准备

```bash
# 转换数据格式
python3 convert_roboflow_to_coco.py --input data/roboflow --output data/coco

# 验证数据集
python3 validate_dataset.py --data_dir data/train
```

---

## 📊 监控训练

### 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir experiments/logs

# 在浏览器中打开
# http://localhost:6006
```

### 检查LoRA权重

```bash
# 查看可训练参数
python3 -c "
from lora_layers import load_lora_weights, count_parameters
import torch
model = ... # 加载您的模型
stats = count_parameters(model)
print(f'Trainable: {stats[\"trainable_percentage\"]:.2f}%')
"
```

---

## ⚠️ 常见问题

### CUDA内存不足

```bash
# 使用更小的batch size
python3 train.py --config configs/minimal_config.yaml

# 使用更小的LoRA rank
# 在配置中设置 rank: 4
```

### 训练不稳定

```bash
# 降低学习率
# 在配置中设置 learning_rate: 5e-5

# 增加warmup步骤
# 在配置中设置 warmup_steps: 500
```

### 模型不收敛

```bash
# 增加训练周期
# 在配置中设置 max_epochs: 50

# 尝试更高的rank
# 在配置中设置 rank: 16
```

---

## 🎓 进一步学习

### 推荐资源

1. **LoRA论文**：Hu et al., 2021 - LoRA: Low-Rank Adaptation of Large Language Models
2. **SAM3文档**：Meta AI Research
3. **PyTorch文档**：官方PyTorch教程

### 高级主题

1. 自定义目标模块
2. 知识蒸馏
3. 模型量化
4. 分布式训练