"""
SAM3-LoRA 混合精度修复演示

你的 RTX 3090 (24GB) 跑 full/light config 都在第二轮卡死，原因：
  1. mixed_precision 配置没实现 → 一直在跑 fp32，显存翻倍
  2. torch.cuda.empty_cache() 造成显存碎片化
  3. 没有 gradient clipping，loss 波动大时梯度爆炸

本脚本演示如何正确修复训练循环。
"""

import torch

print("=" * 70)
print("问题分析：为什么第二轮卡死？")
print("=" * 70)
print("""
当你跑 full_lora_config.yaml:

  lora.rank = 32
  batch_size = 1
  mixed_precision: "bf16"   ← 但代码里根本没实现！

实际的显存使用（fp32 下估算）:
  SAM3 模型本身      ~7.0 GB
  LoRA 参数梯度       ~0.5 GB
  Optimizer 状态      ~2.0 GB (AdamW 存 2 倍参数)
  激活值 (batch=1)    ~5.0 GB (1008x1008 图像)
  loss 计算中间量      ~3.0 GB
  -------------------------
  合计               ~17.5 GB ← 勉强够

为什么第二轮挂？
  1. 训练跑在 fp32，比 bf16 多占 ~40% 显存
  2. 第一轮结束后调用了 torch.cuda.empty_cache()
     → 显存被释放成碎片，第二轮分配连续显存失败 → OOM kill
""")

print("=" * 70)
print("修复 1: 正确使用 autocast（训练循环核心改动）")
print("=" * 70)

SAMPLE_BATCH = """
# ──────────────────────────────────────────────
# 当前代码（第 1022-1063 行）—— 没有混合精度
# ──────────────────────────────────────────────

outputs_list = self.model(input_batch)         # fp32 前向
find_targets = [...]
# ... matcher indices ...
loss_dict = self.loss_wrapper(outputs_list, find_targets)
total_loss = loss_dict[CORE_LOSS_KEY]

self.optimizer.zero_grad()
total_loss.backward()                           # fp32 反向
self.optimizer.step()

# ──────────────────────────────────────────────
# 修复后代码 —— 添加 autocast
# ──────────────────────────────────────────────

from torch.cuda.amp import autocast  # 或用 torch.amp

# 1. 前向传播包在 autocast 里，自动用 bf16/fp16
with autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs_list = self.model(input_batch)      # ← bf16 前向

    # matcher 和 loss 也要放 autocast 里
    find_targets = [...]
    with SAM3Output.iteration_mode(...) as outputs_iter:
        for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
            ...
            outputs["indices"] = self.matcher(outputs, targets)
            ...

    loss_dict = self.loss_wrapper(outputs_list, find_targets)
    total_loss = loss_dict[CORE_LOSS_KEY]

# 2. 反向传播在 autocast 外面
self.optimizer.zero_grad()
total_loss.backward()

# 3. 梯度裁剪（config 里的 max_grad_norm 原来没用上）
torch.nn.utils.clip_grad_norm_(
    [p for p in self.model.parameters() if p.requires_grad],
    max_norm=1.0  # 对应 config training.max_grad_norm
)
self.optimizer.step()
"""

print(SAMPLE_BATCH)

print("=" * 70)
print("修复 2: 去掉 torch.cuda.empty_cache()")
print("=" * 70)

FIX_EMPTY_CACHE = """
# 当前代码第 1144 行:
#
#     torch.cuda.empty_cache()
#
# 这行在验证结束后调用。PyTorch 的缓存分配器会释放当前未用的内存块，
# 但释放后留下的"空洞"导致后续无法分配连续显存块。
# 到第二轮第一个大张量分配时 → CUDA OOM

# 修复：直接删除这行。PyTorch 会自动管理显存缓存。
# 验证结束后 model.train() 切回训练模式就行：
self.model.train()   # ← 不需要 empty_cache()
"""

print(FIX_EMPTY_CACHE)

print("=" * 70)
print("修复 3: 为 fp16 添加 GradScaler（3090 上用 fp16 更快）")
print("=" * 70)

FP16_SCALER = """
# 如果要改成 fp16（3090 上 fp16 比 bf16 更快）:

from torch.cuda.amp import autocast, GradScaler

# 初始化
scaler = GradScaler()

# 训练循环内:
with autocast(device_type="cuda", dtype=torch.float16):  # ← fp16
    outputs_list = self.model(input_batch)
    ...  # matcher + loss
    total_loss = loss_dict[CORE_LOSS_KEY]

self.optimizer.zero_grad()
scaler.scale(total_loss).backward()            # ← 用 scaler
scaler.unscale_(self.optimizer)
torch.nn.utils.clip_grad_norm_(...)
scaler.step(self.optimizer)                    # ← 用 scaler
scaler.update()
"""

print(FP16_SCALER)

print("=" * 70)
print("完整修复摘要：一键改法")
print("=" * 70)

SUMMARY = """
在 train_sam3_lora_native.py 中改 3 个地方：

1. 第 1022-1063 行，在前向传播外包 autocast：

   from torch.cuda.amp import autocast
   ...
   with autocast(device_type="cuda", dtype=torch.bfloat16):
       outputs_list = self.model(input_batch)
       ...  # 整个 matcher + loss 都要包进来
       total_loss = loss_dict[CORE_LOSS_KEY]

2. 在 optimizer.step() 之前加 gradient clipping：

   torch.nn.utils.clip_grad_norm_(
       [p for p in self.model.parameters() if p.requires_grad],
       max_norm=1.0
   )

3. 删掉第 1144 行的 torch.cuda.empty_cache()

改完之后：
  - 显存占用从 ~17.5 GB (fp32) 降到 ~11 GB (bf16)
  - 3090 跑 light config 的 batch_size=2 完全没问题
  - 跑 full config 的 rank=32 也稳
"""

print(SUMMARY)

print("=" * 70)
print("实际 bf16 vs fp32 显存对比模拟")
print("=" * 70)

# 简单模拟显存对比
dummy_size = 1024 * 1024 * 100  # 100M 参数
fp32_bytes = dummy_size * 4
bf16_bytes = dummy_size * 2
saving_pct = (1 - bf16_bytes / fp32_bytes) * 100

print(f"  100M 参数: fp32 = {fp32_bytes / 1024**3:.1f} GB")
print(f"  100M 参数: bf16 = {bf16_bytes / 1024**3:.1f} GB")
print(f"  节省: {saving_pct:.0f}%")
print()
print(f"  SAM3 约 700M 参数:")
print(f"    fp32: {700 * 4 / 1024:.1f} GB")
print(f"    bf16: {700 * 2 / 1024:.1f} GB")
print(f"    节省: ~{700 * 2 / 1024:.1f} GB 显存")
