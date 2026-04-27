# 训练损失未降低 - 诊断

## 问题：损失剧烈波动

从训练输出：
```
Step 1: 169
Step 2: 141
Step 3: 169
Step 4: 242 ← 增加！
Step 5: 212
Step 6: 182
Step 7: 186
Step 8: 161
Step 9: 189
Step 10: 181
Step 11: 181
Step 12: 172
Step 13: 218 ← 再次增加！
Step 14: 186
Step 15: 175
Step 16: 198
```

**这是不正常的** - 损失应该更平稳地下降！

---

## 根本原因

### 1. **关键问题：配置指向错误的数据目录**

**文件**：`configs/full_lora_config.yaml`（第35行）
```yaml
data_dir: "/workspace/data3"  # ← 错误！应该是 /workspace/data2
```

**影响**：如果data3不存在或数据不同，可能导致问题。

---

### 2. **小批量大小导致的高梯度方差**

**当前设置**：
- `batch_size: 1`（每步仅一个样本）
- `gradient_accumulation_steps: 16`
- **有效批量大小：16**

**问题**：batch_size=1时，每次梯度更新仅从**一个样本**计算，导致：
- **梯度估计方差高**
- **损失不稳定**，跳跃不定
- **收敛差** - 模型无法学习稳定模式

**为什么会这样**：
- 样本A可能有3个裂缝 → 损失专注于检测3个对象
- 样本B可能有1个裂缝 → 损失专注于检测1个对象
- 样本C可能有10个裂缝 → 损失专注于检测多个对象
- 模型每一步都收到冲突信号！

---

### 3. **增加的LoRA Rank = 更大的内存压力**

**配置变更**：
- `rank: 64`（原来是32）- **翻倍**
- `alpha: 128`（原来是64）

**影响**：
- 更多的可训练参数：**11.8M参数**（模型的1.38%）
- 更高的内存使用：**22.6GB / 32.6GB已用**（69%满）
- 训练进程不断被**OOM杀死**（退出代码137）

---

### 4. **学习率预热未正确工作**

**配置**：`warmup_steps: 200`

**问题**：使用梯度累积时，预热应该按**梯度更新**计数，而不是数据样本。模型可能过早获得完整学习率，导致不稳定。

---

### 5. **未验证梯度裁剪**

**配置**：`max_grad_norm: 1.0`

**问题**：梯度是否实际被裁剪？大梯度可能导致损失突变。

---

## 解决方案（按优先级排序）

### 立即修复1：更正数据目录
```yaml
# 在configs/full_lora_config.yaml第35行
data_dir: "/workspace/data2"  # ← 修复这个！
```

### 立即修复2：降低LoRA Rank以避免OOM
```yaml
# 在configs/full_lora_config.yaml
lora:
  rank: 32  # 恢复原值（原来是64）
  alpha: 64  # 恢复原值（原来是128）
```

### 立即修复3：增加批量大小以减少方差
```yaml
# 在configs/full_lora_config.yaml
training:
  batch_size: 2  # 从1增加
  gradient_accumulation_steps: 8  # 从16减少
  # 有效批量大小保持：2 × 8 = 16
```

**为什么有帮助**：
- 每梯度步使用**2个样本**而不是1个
- 方差减少50%
- 学习更稳定
- 相同的有效批量大小（16）

### 修复4：正确添加学习率预热

当前实现应该处理预热，但通过检查损失是否在约200步后稳定来验证。

### 修复5：监控梯度范数

添加日志以验证梯度没有爆炸：
```python
# 在total_loss.backward()之后
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm:.4f}")
```

---

## 替代方案：使用轻量配置

如果OOM继续，使用更轻的配置：
```bash
python3 train_sam3_lora_native.py --config configs/light_lora_config.yaml
```

**轻量配置**：
- `rank: 16`（更少内存）
- `batch_size: 2`（更稳定）
- 跳过视觉编码器LoRA（节省内存）
- 相同的有效训练能力

---

## 修复后的预期行为

**第一个周期，损失应该**：
1. 开始很高（150-200）
2. 平滑下降：150 → 140 → 130 → 120 → ...
3. 预热后稳定（约200步）
4. 继续逐渐下降

**不应该**：
- 跳跃不定：169 → 141 → 169 → 242 → 212
- 频繁增加
- 多步保持不变

---

## 总结

| 问题 | 影响 | 修复 |
|-------|--------|------|
| 错误的数据目录 | 未知/关键 | 改为`/workspace/data2` |
| batch_size=1 | 高方差 | 增加到2 |
| rank=64 | OOM杀死训练 | 减少到32 |
| gradient_accumulation=16 | 不太重要 | 减少到8 |

**优先级**：首先修复数据目录和batch_size！

---

## RTX 3090优化建议

针对RTX 3090（24GB显存）的推荐配置：

```yaml
# 稳定训练配置
lora:
  rank: 16                   # 平衡容量和显存
  alpha: 32                  # 2x rank
  dropout: 0.1
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "out_proj"
    # 跳过fc1/fc2以节省显存

training:
  batch_size: 2              # RTX3090可设为2
  gradient_accumulation_steps: 8  # 有效批量 = 16
  learning_rate: 1e-4        # 对于轻量模型使用较高学习率
  mixed_precision: "bf16"    # 使用bf16节省显存
  warmup_steps: 100
  max_grad_norm: 1.0         # 启用梯度裁剪
```

### 额外的RTX 3090技巧

1. **使用bf16而非fp16**：3090上bf16更稳定
2. **避免empty_cache()**：会导致显存碎片化
3. **启用梯度累积**：用更少显存实现更大批量
4. **监控显存使用**：`nvidia-smi -l 1`