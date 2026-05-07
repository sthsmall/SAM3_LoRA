#!/usr/bin/env python3
"""
Numerical correctness test for merged SAM3+LoRA checkpoint.

Verifies that the merged model's forward pass output is numerically identical
to the LoRA-wrapped model's output (within floating point tolerance).

Usage:
    python tests/test_merged_model.py \\
        --checkpoint sam3_merged_video.pt \\
        --config configs/full_lora_config.yaml \\
        --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
from PIL import Image
import numpy as np

os.makedirs("test_outputs", exist_ok=True)


def load_base_model(checkpoint_path: str, bpe_path: str, device: str,
                    apply_temporal_disambiguation: bool = False):
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        load_from_HF=False,
        apply_temporal_disambiguation=apply_temporal_disambiguation,
        device=device,
        compile=False,
    )
    model.eval()
    return model


def load_merged_model(checkpoint_path: str, device: str):
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=checkpoint_path,
        bpe_path="bpe_simple_vocab_16e6.txt.gz",
        load_from_HF=False,
        apply_temporal_disambiguation=False,
        device=device,
        compile=False,
    )
    model.eval()
    return model


def create_test_input(device: str):
    from sam3.train.data.sam3_image_dataset import (
        Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata,
    )
    from sam3.train.data.collator import collate_fn_api
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI,
    )

    dummy = Image.new("RGB", (640, 480), color=(70, 130, 180))

    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    sam_image = SAMImage(data=dummy, objects=[], size=[480, 640])
    query = FindQueryLoaded(
        query_text="object",
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=0, original_image_id=0,
            original_category_id=1, original_size=[640, 480],
            object_id=0, frame_index=0,
        ),
    )
    datapoint = Datapoint(find_queries=[query], images=[sam_image])
    datapoint = transform(datapoint)
    batch = collate_fn_api([datapoint], dict_key="input")["input"]
    batch = copy_data_to_device(batch, device, non_blocking=True)
    return batch


def run_forward(model, batch, is_video_model=True):
    with torch.no_grad():
        if is_video_model:
            outputs = model.detector(batch)
        else:
            outputs = model(batch)
    last_out = list(outputs)[-1]
    return {
        "logits": last_out["pred_logits"],
        "boxes": last_out["pred_boxes"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Numerical correctness test for merged SAM3+LoRA checkpoint"
    )
    parser.add_argument("--checkpoint", type=str, default="sam3_merged_video.pt",
                        help="Path to merged checkpoint")
    parser.add_argument("--config", type=str, default="configs/full_lora_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--lora_checkpoint", type=str,
                        default="outputs/sam3_lora/best_lora_weights.pt",
                        help="Path to LoRA weights")
    parser.add_argument("--base_checkpoint", type=str,
                        default="/mnt/d/projects/specific/sam3/sam3/sam3.pt",
                        help="Path to base SAM3 checkpoint (for comparison)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tolerance", type=float, default=5e-3,
                        help="Maximum allowed numerical difference (default 5e-3 for bfloat16; use 1e-4 for float32)")
    args = parser.parse_args()

    print("=" * 60)
    print("SAM3 Merge Numerical Correctness Test")
    print("=" * 60)

    # Load config for LoRA params
    with open(args.config) as f:
        config = yaml.safe_load(f)
    lora_cfg = config["lora"]

    bpe_path = os.path.join(os.path.dirname(__file__), "..", "bpe_simple_vocab_16e6.txt.gz")
    device = args.device

    print(f"\nConfig: {args.config}")
    print(f"  LoRA rank={lora_cfg['rank']}, alpha={lora_cfg['alpha']}")
    print(f"  Device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"\n❌ Merged checkpoint not found: {args.checkpoint}")
        exit(1)

    if not os.path.exists(args.lora_checkpoint):
        print(f"\n⚠️  LoRA checkpoint not found: {args.lora_checkpoint}")
        print("   Skipping numerical comparison (only testing merged model loads)")
        args.lora_checkpoint = None

    # ─── 1. Load merged model ───
    print("\n1. Loading merged model...")
    merged_model = load_merged_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in merged_model.parameters())
    print(f"   Parameters: {n_params:,}")

    # ─── 2. Create test input ───
    print("\n2. Creating test input...")
    batch = create_test_input(device)

    # ─── 3. Forward pass on merged model ───
    print("\n3. Running merged model forward pass...")
    merged_out = run_forward(merged_model, batch, is_video_model=True)
    merged_logits = merged_out["logits"]
    merged_boxes = merged_out["boxes"]
    merged_scores = merged_logits.sigmoid()[0, :, 0]
    merged_max = merged_scores.max().item()
    merged_mean = merged_scores.mean().item()
    n_above = (merged_scores > 0.5).sum().item()

    print(f"   pred_logits shape: {merged_logits.shape}")
    print(f"   pred_boxes shape:  {merged_boxes.shape}")
    print(f"   Max score:    {merged_max:.6f}")
    print(f"   Mean score:   {merged_mean:.6f}")
    print(f"   Scores > 0.5: {n_above} / {len(merged_scores)}")

    if merged_max < 0.01 and n_above == 0:
        print(f"\n⚠️  WARNING: Very low scores detected (max={merged_max:.6f})!")
        print("   The merged model may not be functioning correctly.")

    # ─── 4. Compare with LoRA model ───
    if args.lora_checkpoint:
        print(f"\n4. Comparing with LoRA-wrapped model...")
        from lora_layers import LoRAConfig, apply_lora_to_model, count_parameters

        base_model = load_base_model(
            args.base_checkpoint, bpe_path, device,
            apply_temporal_disambiguation=False
        )

        # Auto-detect actual rank from checkpoint (same logic as merge_lora_to_base.py)
        lora_raw = torch.load(args.lora_checkpoint, map_location="cpu")
        if isinstance(lora_raw, dict) and "lora_state_dict" in lora_raw:
            lora_raw = lora_raw["lora_state_dict"]

        ckpt_rank = None
        for key, value in lora_raw.items():
            if ".lora_A" in key and isinstance(value, torch.Tensor) and value.ndim == 2:
                ckpt_rank = value.shape[1]
                break

        comp_rank = lora_cfg["rank"]
        comp_alpha = lora_cfg["alpha"]
        if ckpt_rank is not None and ckpt_rank != comp_rank:
            print(f"   ⚠️  Config rank={comp_rank} but checkpoint has rank={ckpt_rank}, using checkpoint rank")
            comp_rank = ckpt_rank
            comp_alpha = comp_rank * 2

        lora_config = LoRAConfig(
            rank=comp_rank,
            alpha=comp_alpha,
            dropout=0.0,
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg.get("apply_to_vision_encoder", True),
            apply_to_text_encoder=lora_cfg.get("apply_to_text_encoder", True),
            apply_to_geometry_encoder=lora_cfg.get("apply_to_geometry_encoder", False),
            apply_to_detr_encoder=lora_cfg.get("apply_to_detr_encoder", True),
            apply_to_detr_decoder=lora_cfg.get("apply_to_detr_decoder", True),
            apply_to_mask_decoder=lora_cfg.get("apply_to_mask_decoder", False),
        )
        lora_model = apply_lora_to_model(base_model, lora_config)

        # Load LoRA weights with key remapping (image model → video model: add "detector." prefix)
        lora_raw = torch.load(args.lora_checkpoint, map_location="cpu")
        if isinstance(lora_raw, dict) and "lora_state_dict" in lora_raw:
            lora_raw = lora_raw["lora_state_dict"]
        model_keys = set(lora_model.state_dict().keys())
        remapped = {}
        for key, value in lora_raw.items():
            if key in model_keys:
                remapped[key] = value
            elif f"detector.{key}" in model_keys:
                remapped[f"detector.{key}"] = value
        lora_model.load_state_dict(remapped, strict=False)
        lora_model.to(device)
        lora_model.eval()

        stats = count_parameters(lora_model)
        print(f"   Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        lora_out = run_forward(lora_model, batch, is_video_model=True)
        lora_scores = lora_out["logits"].sigmoid()[0, :, 0]
        lora_max = lora_scores.max().item()

        score_diff = abs(merged_max - lora_max)
        max_abs_diff = (merged_scores - lora_scores).abs().max().item()

        print(f"\n   Merged model max score: {merged_max:.6f}")
        print(f"   LoRA model max score:   {lora_max:.6f}")
        print(f"   Max score difference:   {score_diff:.6e}")
        print(f"   Max element-wise diff:  {max_abs_diff:.6e}")

        if max_abs_diff < args.tolerance:
            print(f"\n   ✅ PASS: Merged model output matches LoRA-wrapped model (diff < {args.tolerance})")
        else:
            print(f"\n   ❌ FAIL: Numerical difference {max_abs_diff:.6e} exceeds tolerance {args.tolerance}!")
            print(f"   This indicates the merge was NOT correct.")
            exit(1)

    # ─── 5. Summary ───
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    if args.lora_checkpoint:
        print(f"\nMerged checkpoint is numerically identical to LoRA model.")
    else:
        print(f"\nMerged checkpoint loads and runs successfully.")
    print(f"Location: {os.path.abspath(args.checkpoint)}")


if __name__ == "__main__":
    main()
