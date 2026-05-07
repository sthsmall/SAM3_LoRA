# SAM3-LoRA 用于 MOT（多目标跟踪）项目说明

## 🎯 项目目标

本项目通过 **LoRA（Low-Rank Adaptation）微调 SAM3 分割模型**，提升其在特定场景下对目标掩码的识别能力，最终将微调后的模型应用于 **MOT（Multi-Object Tracking，多目标跟踪）** 任务，通过 SAM3 的 video 模式实现高质量的视频目标追踪。

## 📌 背景

SAM3（Segment Anything Model 3）是 Meta 推出的第三代分割基础模型，具备：

1. **图像分割**：给定 text prompt 或几何 prompt（box/point），生成精细的分割掩码
2. **视频追踪**：内置 tracker，可在视频中持续追踪目标，自动处理遮挡、消失、重现

然而，通用 SAM3 模型在特定领域（如工业检测、自动驾驶、无人机视角等）的识别精度有限。通过 LoRA 微调，可以用极少参数（约 1%）让模型适配到目标任务，同时保留 video tracker 的完整能力。

## 🔄 工作流程

```
COCO 标注数据
    │
    ▼
train_sam3_lora_native.py        ← LoRA 微调（基于 SAM3 原生 API）
    │
    ▼
outputs/best_lora_weights.pt      ← 仅 ~10-50MB 的 LoRA 权重
    │
    ▼
merge_lora_to_base.py             ← 将 LoRA 合并入 SAM3 video 模型
    │
    ▼
sam3_merged_video.pt              ← 约 3.4GB 的独立追踪模型
    │
    ▼
Sam3VideoPredictor / demo         ← 视频 MOT 推理
```

### 为什么合并（merge）而不直接加载 LoRA？

- LoRA 权重在推理时需要额外的运行时计算（`Wx + x·A·B`），增加延迟
- 合并后模型为纯 `nn.Linear` / `nn.MultiheadAttention`，无额外开销
- 合并后的 `.pt` 文件可以直接喂给 `Sam3VideoPredictor`，与原版 SAM3 推理流程完全一致

## 🧠 微调原理

### LoRA 注入位置

训练配置 `configs/full_lora_config.yaml` 中 LoRA 覆盖了 SAM3 的所有关键组件：

| 组件 | 作用 | MOT 相关度 |
|------|------|-----------|
| **Vision Encoder (ViT)** | 提取图像视觉特征 | ⭐⭐⭐ 高 — 理解目标外观 |
| **Text Encoder** | 编码 text prompt（如 "person", "car"） | ⭐⭐⭐ 高 — 目标语义 |
| **Geometry Encoder** | 处理 box/point 几何输入 | ⭐⭐ 中 |
| **DETR Encoder** | 图像-文本跨模态融合 | ⭐⭐⭐ 高 — 特征关联 |
| **DETR Decoder** | 生成目标 queries | ⭐⭐⭐ 高 — 目标定位 |
| **Mask Decoder** | 生成最终分割掩码 | ⭐⭐⭐ 高 — 掩码精度 |

### 训练策略

- **数据格式**：COCO JSON（`_annotations.coco.json`），支持 polygon 和 RLE 分割
- **Text prompt 使用 supercategory**：合并相近类别，例如 `"concrete_crack_01"` 和 `"asphalt_crack_02"` 共用 `"crack"` prompt
- **Loss**：SAM3 原生的多组件 loss（CE + bbox + GIoU + mask + dice）
- **验证**：训练时只算 validation loss（快速），训练后用 `validate_sam3_lora.py` 跑完整 mAP/cgF1 指标
- **多 GPU**：支持 DDP 分布式训练

## 📁 核心文件

| 文件 | 用途 |
|------|------|
| `train_sam3_lora_native.py` | ⭐ 微调训练脚本（生产推荐） |
| `merge_lora_to_base.py` | 合并 LoRA 权重到 base SAM3 |
| `validate_sam3_lora.py` | 完整评估（mAP、cgF1） |
| `infer_sam.py` | 单图推理验证 |
| `merge_and_test.sh` | 一键合并 + 测试 |
| `configs/full_lora_config.yaml` | 训练配置（LoRA 参数、数据路径等） |

## 🎬 MOT 推理管线

合并后的模型通过 `Sam3VideoPredictor` 运行视频追踪：

```python
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(
    checkpoint_path="sam3_merged_video.pt",
    bpe_path="bpe_simple_vocab_16e6.txt.gz",
)

# 逐帧处理，返回追踪结果（masklet + ID 绑定）
for frame_idx, frame in enumerate(video_frames):
    predictor.add_new_frame(frame)
    masks, obj_ids = predictor.get_tracking_results()
```

MOT 中的关键能力：
- **目标检测**：通过 text prompt（如 `"person"`）检测目标，生成掩码
- **ID 绑定**：SAM3 的 tracker 自动关联跨帧的同一目标
- **热启动消歧（hotstart）**：过滤假阳性，只保留稳定的追踪目标
- **重识别**：目标消失后重新出现时自动恢复追踪

## 📊 预期效果

- **可训练参数**：约 1%-2%（~12M / 848M）
- **LoRA 权重大小**：10-50MB（vs 完整模型 3.4GB）
- **4 卡训练**：有效 batch size 线性扩展
- **MOT 增益**：微调后目标掩码 mAP 和 cgF1 显著提升，假阳性减少

## 🔧 后续优化方向

1. **多类别 MOT**：不同类别目标独立追踪（如同时追人和车）
2. **数据增强**：运动模糊、遮挡模拟，提升 tracker 鲁棒性
3. **时序一致性训练**：在微调中加入帧间一致性 loss
4. **推理加速**：`torch.compile` 模式加速 video 推理

---

*项目来源：AI Research Group, KMUTT*
*更新日期：2026-05*
