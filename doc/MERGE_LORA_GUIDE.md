# LoRA 权重合并指南

## 概述

训练完成后，LoRA 权重保存在单独的文件中（\~几MB），而非完整模型（\~几GB）。本脚本将 LoRA 权重合并回基础模型，生成独立的完整 `.pt` 文件，方便部署和推理。

**支持两种模型类型**：

- **图像模型** (`--model_type image`): 适用于单图像分割任务
- **视频模型** (`--model_type video`): 适用于视频跟踪任务（**推荐**）

## 合并公式

对于每个被 LoRA 适配的 Linear 层：

```
W_merged = W_original + (B @ A) * (alpha / rank)
```

其中：

- `W_original`: 原始冻结的权重矩阵
- `A`, `B`: LoRA 的低秩矩阵
- `alpha`, `rank`: LoRA 配置参数

## 快速开始

### 方法 1: 使用自动化脚本（推荐）

一键完成合并和测试：

```bash
# 合并视频模型
bash merge_and_test.sh \
    --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt \
    --output sam3_merged_video.pt \
    --model_type video \
    --apply_temporal_disambiguation False
```

### 方法 2: 分步执行

#### Step 1: 合并 LoRA 权重

```bash
# 视频模型（推荐）
python merge_lora_to_base.py \
    --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt \
    --output sam3_merged_video.pt \
    --model_type video \
    --apply_temporal_disambiguation False

# 图像模型
python merge_lora_to_base.py \
    --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt \
    --output sam3_merged_image.pt \
    --model_type image
```

#### Step 2: 测试合并后的模型

```bash
# 测试视频预测器
python test_merged_video_predictor.py \
    --checkpoint sam3_merged_video.pt \
    --apply_temporal_disambiguation False
```

### 3. 指定基础模型路径

如果自动下载的 SAM3 基础模型不是您想要的版本：

```bash
python merge_lora_to_base.py \
    --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt \
    --output sam3_merged_video.pt \
    --model_type video \
    --checkpoint_path /path/to/your/sam3_base.pt \
    --bpe_path bpe_simple_vocab_16e6.txt.gz
```

### 4. 指定 LoRA 参数

如果 LoRA 训练时使用了非默认参数：

```bash
python merge_lora_to_base.py \
    --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt \
    --output sam3_merged_video.pt \
    --model_type video \
    --lora_rank 16 \
    --lora_alpha 32
```

### 5. 控制应用范围

根据训练配置调整 LoRA 应用范围：

```bash
python merge_lora_to_base.py \
    --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt \
    --output sam3_merged_video.pt \
    --model_type video \
    --apply_to_vision_encoder true \
    --apply_to_text_encoder true \
    --apply_to_detr_encoder true \
    --apply_to_detr_decoder true
```

## 参数说明

### 核心参数

| 参数                                | 类型   | 默认值   | 说明                     |
| --------------------------------- | ---- | ----- | ---------------------- |
| `--lora_checkpoint`               | str  | 必填    | LoRA 权重文件路径            |
| `--output`                        | str  | 必填    | 合并后模型的输出路径             |
| `--model_type`                    | str  | video | 模型类型：`image` 或 `video` |
| `--apply_temporal_disambiguation` | bool | False | 视频模型是否启用时间消歧           |

### 高级参数

| 参数                            | 类型   | 默认值                             | 说明                              |
| ----------------------------- | ---- | ------------------------------- | ------------------------------- |
| `--checkpoint_path`           | str  | None                            | SAM3 基础模型路径（默认从 HuggingFace 下载） |
| `--bpe_path`                  | str  | bpe\_simple\_vocab\_16e6.txt.gz | BPE 词汇表路径                       |
| `--lora_rank`                 | int  | 8                               | LoRA 秩（必须与训练配置一致）               |
| `--lora_alpha`                | int  | 16                              | LoRA 缩放因子（必须与训练配置一致）            |
| `--device`                    | str  | cuda/cpu                        | 加载模型的设备                         |
| `--apply_to_vision_encoder`   | bool | True                            | 是否在视觉编码器应用 LoRA                 |
| `--apply_to_text_encoder`     | bool | True                            | 是否在文本编码器应用 LoRA                 |
| `--apply_to_geometry_encoder` | bool | False                           | 是否在几何编码器应用 LoRA                 |
| `--apply_to_detr_encoder`     | bool | True                            | 是否在 DETR 编码器应用 LoRA             |
| `--apply_to_detr_decoder`     | bool | True                            | 是否在 DETR 解码器应用 LoRA             |
| `--apply_to_mask_decoder`     | bool | False                           | 是否在 Mask 解码器应用 LoRA             |
| `--verbose`                   | flag | False                           | 打印详细的合并过程                       |

## Temporal Disambiguation（时间消歧）

### 什么是时间消歧？

时间消歧是 SAM3 视频模型中的高级特性，用于处理复杂视频场景中的目标歧义问题。它通过智能选择历史帧来提高多目标跟踪的准确性。

### 何时使用？

| 场景        | 推荐设置    | 说明          |
| --------- | ------- | ----------- |
| 单目标跟踪     | `False` | 不需要复杂的时间推理  |
| 多目标、相似目标  | `True`  | 需要智能选择相关历史帧 |
| 目标经常遮挡/交错 | `True`  | 需要更好的歧义消除   |
| 边缘设备部署    | `False` | 减少计算开销      |
| 快速推理      | `False` | 降低延迟        |

### 性能对比

| 设置      | 准确性 | 计算量 | 内存占用 |
| ------- | --- | --- | ---- |
| `True`  | 更高  | 更大  | 更多   |
| `False` | 适中  | 更小  | 更少   |

**注意**: 使用 `False` 时，模型会缺少部分 tracker 组件的权重，但仍可正常工作。

## 输出文件

合并成功后会生成一个完整的模型文件：

```bash
ls -lh sam3_merged_video.pt
# -rw-r--r-- 1 user  1.2G sam3_merged_video.pt
```

文件大小与原始 SAM3 基础模型相同。

## 使用合并后的模型

### 1. 视频跟踪任务（推荐）

```python
from sam3.model_builder import build_sam3_video_predictor

# 加载视频预测器
predictor = build_sam3_video_predictor(
    checkpoint_path="sam3_merged_video.pt",
    bpe_path="bpe_simple_vocab_16e6.txt.gz",
    strict_state_dict_loading=False,
    apply_temporal_disambiguation=False,
)

# 开始推理会话
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path="path/to/video_or_images",
    )
)
session_id = response["session_id"]

# 添加文本提示
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="airplane",  # 你的文本提示
    )
)

# 传播到整个视频
for frame_data in predictor.handle_stream_request(
    request=dict(type="propagate_in_video", session_id=session_id)
):
    outputs = frame_data["outputs"]
    # 处理输出结果
    print(f"Frame {frame_data['frame_index']}: {len(outputs.get('out_obj_ids', []))} objects")

# 关闭会话
predictor.handle_request(
    request=dict(type="close_session", session_id=session_id)
)
```

### 2. 图像分割任务

```python
import torch
from sam3.model_builder import build_sam3_video_model
from PIL import Image

# 加载模型
model = build_sam3_video_model(
    checkpoint_path="sam3_merged_video.pt",
    bpe_path="bpe_simple_vocab_16e6.txt.gz",
    load_from_HF=False,
    apply_temporal_disambiguation=False,
)
model.eval()

# 加载图像
image = Image.open("test.jpg")

# 推理
with torch.no_grad():
    # 根据实际的推理接口调整
    output = model(image)

print(f"Output type: {type(output)}")
```

### 3. 直接加载 state\_dict

```python
import torch

# 直接加载
state_dict = torch.load("sam3_merged_video.pt")
print(f"Loaded {len(state_dict)} keys")

# 或者加载到模型
from sam3.model_builder import build_sam3_video_model

model = build_sam3_video_model(
    checkpoint_path="sam3_merged_video.pt",
    load_from_HF=False,
)
```

## 在 Demo 中使用

### 更新你的 demo 脚本

```python
# my_demo.py
CHECKPOINT_PATH = "sam3_merged_video.pt"  # 修改为合并后的模型路径

# 然后正常运行
python my_demo.py
```

## 注意事项

1. **参数一致性**: 确保 `--lora_rank` 和 `--lora_alpha` 与训练时完全一致
2. **模型类型匹配**: 合并时使用的 `--model_type` 必须与推理时使用的模型构建函数匹配
3. **BPE 路径**: 确保 `--bpe_path` 指向有效的 BPE 词汇表文件
4. **设备兼容性**: 合并过程可以在 CPU 上完成，但合并后模型可在任意设备使用
5. **存储空间**: 需要同时有基础模型和 LoRA 权重，约 1.3GB + 几 MB
6. **备份**: 建议保留原始 LoRA 权重文件，方便后续调整

## 文件对比

| 文件类型                   | 大小        | 包含内容       | 用途   |
| ---------------------- | --------- | ---------- | ---- |
| `sam3_base.pt`         | \~1.2 GB  | 原始 SAM3 权重 | 基础模型 |
| `lora_weights.pt`      | \~5-50 MB | 仅 LoRA 权重  | 轻量存储 |
| `sam3_merged_video.pt` | \~1.2 GB  | 完整视频模型     | 视频跟踪 |
| `sam3_merged_image.pt` | \~1.2 GB  | 完整图像模型     | 图像分割 |

## 故障排除

### 问题: Missing keys 警告

```
Warning: Missing keys: ['...']
```

**解决**:

- 检查 LoRA 训练配置与合并脚本参数是否一致
- 如果使用了 `--strict_state_dict_loading=False`，这些警告可以忽略

### 问题: Unexpected keys 警告

```
Warning: Unexpected keys: ['...']
```

**解决**: 可能是因为 checkpoint 格式不同，可以忽略或清理后重新加载。

### 问题: FileNotFoundError: bpe\_simple\_vocab\_16e6.txt.gz

```
FileNotFoundError: [Errno 2] No such file or directory: '.../bpe_simple_vocab_16e6.txt.gz'
```

**解决**: 确保 BPE 词汇表文件存在于指定路径，或使用绝对路径：

```bash
--bpe_path /absolute/path/to/bpe_simple_vocab_16e6.txt.gz
```

### 问题: 模型加载失败

```
RuntimeError: Error(s) in loading state_dict for Sam3VideoInferenceWithInstanceInteractivity
```

**解决**:

1. 确保合并时的 `--model_type` 与推理时使用的模型类型一致
2. 使用 `--strict_state_dict_loading=False` 允许缺失部分权重
3. 检查 checkpoint 文件是否完整

### 问题: 模型文件过大/过小

**解决**: 检查是否正确加载了完整模型，LoRA 权重本身不应该影响最终文件大小。

### 问题: 测试脚本运行失败

**解决**: 确保所有依赖已安装：

```bash
pip install decord pillow
```

## 测试验证

### 自动化测试

```bash
# 使用自动化脚本一键测试
bash merge_and_test.sh \
    --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt \
    --output sam3_merged_video.pt \
    --model_type video
```

### 手动测试

```bash
# Step 1: 合并
python merge_lora_to_base.py --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt --output sam3_merged_video.pt --model_type video

# Step 2: 测试
python test_merged_video_predictor.py --checkpoint sam3_merged_video.pt

# Step 3: 在 demo 中使用
# 编辑 my_demo.py，设置正确的 CHECKPOINT_PATH
python my_demo.py
```

## 完整工作流程示例

```bash
# 1. 合并 LoRA 权重（视频模型）
bash merge_and_test.sh \
    --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt \
    --output sam3_merged_video.pt \
    --model_type video \
    --apply_temporal_disambiguation False

# 2. 检查输出文件
ls -lh sam3_merged_video.pt

# 3. 在你的 demo 中使用
# 修改 my_demo.py:
#   CHECKPOINT_PATH = "/path/to/sam3_merged_video.pt"

# 4. 运行 demo
python my_demo.py
```

## 相关文件

| 文件                               | 说明          |
| -------------------------------- | ----------- |
| `merge_lora_to_base.py`          | LoRA 权重合并脚本 |
| `merge_and_test.sh`              | 自动化合并和测试脚本  |
| `test_merged_video_predictor.py` | 测试合并后的视频模型  |
| `test_merged_model.py`           | 测试合并后的图像模型  |
| `my_demo.py`                     | 视频跟踪示例演示    |

