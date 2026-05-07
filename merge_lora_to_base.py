#!/usr/bin/env python3
"""
Merge LoRA weights into base SAM3 model to create a standalone .pt file.

All LoRA hyperparameters (rank, alpha, target_modules, apply_to_*) are read
from the training config YAML file. No need to manually duplicate these values.

Usage:
    python merge_lora_to_base.py \\
        --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt \\
        --config configs/full_lora_config.yaml \\
        --output sam3_merged.pt

    python merge_lora_to_base.py \\
        --lora_checkpoint outputs/sam3_lora/best_lora_weights.pt \\
        --config configs/full_lora_config.yaml \\
        --output sam3_merged_video.pt \\
        --model_type video
"""

import argparse
import os
import torch
import torch.nn as nn
import yaml
from pathlib import Path

from sam3.model_builder import build_sam3_image_model,build_sam3_video_model
from lora_layers import LoRAConfig, apply_lora_to_model, LoRALinear, LoRALayer, MultiheadAttentionLoRA


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into the base model and restore original architecture.

    Steps:
    1. Replace LoRALinear wrappers with plain nn.Linear (merged weights)
    2. Replace MultiheadAttentionLoRA with nn.MultiheadAttention (fused QKV)
    """

    # --- Step 1: Merge LoRALinear -> nn.Linear ---
    lora_modules = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_modules.append((name, module))

    print(f"Found {len(lora_modules)} LoRA layers to merge")

    for name, module in lora_modules:
        # W_merged = W_original + (A @ B)^T * scaling
        # lora_A: (in_features, rank), lora_B: (rank, out_features)
        # A @ B = (in_features, out_features)
        # nn.Linear.weight is (out_features, in_features) -> transpose
        device = module.original_layer.weight.device
        lora_delta = (module.lora.lora_A.data.to(device) @ module.lora.lora_B.data.to(device)) * module.lora.scaling
        merged_weight = module.original_layer.weight.data + lora_delta.t()

        merged_linear = nn.Linear(
            module.original_layer.in_features,
            module.original_layer.out_features,
            bias=module.original_layer.bias is not None,
        )
        merged_linear.weight.data = merged_weight
        if module.original_layer.bias is not None:
            merged_linear.bias.data = module.original_layer.bias.data.clone()

        # Replace in parent module
        *parent_path, attr_name = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        setattr(parent, attr_name, merged_linear)

        print(f"  Merged: {name}")

    # --- Step 2: Convert MultiheadAttentionLoRA -> nn.MultiheadAttention ---
    mha_modules = []
    for name, module in model.named_modules():
        if isinstance(module, MultiheadAttentionLoRA):
            mha_modules.append((name, module))

    if mha_modules:
        print(f"\nConverting {len(mha_modules)} MultiheadAttentionLoRA -> nn.MultiheadAttention")

    for name, module in mha_modules:
        embed_dim = module.embed_dim
        num_heads = module.num_heads
        dropout = module.dropout
        batch_first = module.batch_first
        has_bias = module.q_proj.bias is not None

        # q_proj, k_proj, v_proj are now nn.Linear (merged in step 1)
        # Fuse them into in_proj_weight / in_proj_bias
        in_proj_weight = torch.cat(
            [module.q_proj.weight.data, module.k_proj.weight.data, module.v_proj.weight.data], dim=0
        )
        if has_bias:
            in_proj_bias = torch.cat(
                [module.q_proj.bias.data, module.k_proj.bias.data, module.v_proj.bias.data], dim=0
            )

        new_mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=has_bias,
            batch_first=batch_first,
        )
        # Load fused QKV
        new_mha.in_proj_weight.data = in_proj_weight
        if has_bias:
            new_mha.in_proj_bias.data = in_proj_bias
        # Copy output projection
        new_mha.out_proj.weight.data = module.out_proj.weight.data
        if module.out_proj.bias is not None:
            new_mha.out_proj.bias.data = module.out_proj.bias.data

        # Replace in parent module
        *parent_path, attr_name = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        setattr(parent, attr_name, new_mha)

        print(f"  Converted: {name}")

    return model


def load_lora_config_from_yaml(config_path: str):
    """Load LoRA configuration from a training YAML config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lora_cfg = config.get("lora", {})
    if not lora_cfg:
        raise ValueError(f"No 'lora' section found in {config_path}")

    print(f"📄 Loaded LoRA config from: {config_path}")

    lora_params = {
        "rank": lora_cfg["rank"],
        "alpha": lora_cfg["alpha"],
        "dropout": lora_cfg.get("dropout", 0.0),
        "target_modules": lora_cfg["target_modules"],
        "apply_to_vision_encoder": lora_cfg.get("apply_to_vision_encoder", True),
        "apply_to_text_encoder": lora_cfg.get("apply_to_text_encoder", True),
        "apply_to_geometry_encoder": lora_cfg.get("apply_to_geometry_encoder", False),
        "apply_to_detr_encoder": lora_cfg.get("apply_to_detr_encoder", True),
        "apply_to_detr_decoder": lora_cfg.get("apply_to_detr_decoder", True),
        "apply_to_mask_decoder": lora_cfg.get("apply_to_mask_decoder", False),
    }

    for k, v in lora_params.items():
        print(f"  {k}: {v}")

    return lora_params


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into base SAM3 model. "
                    "All LoRA settings are read from --config YAML."
    )
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint (.pt file)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for merged model (.pt file)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML (provides all LoRA settings)")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/mnt/d/projects/specific/sam3/sam3/sam3.pt",
                        help="Path to base SAM3 checkpoint (must match training)")
    parser.add_argument("--bpe_path", type=str, default="bpe_simple_vocab_16e6.txt.gz",
                        help="Path to BPE vocabulary file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to load model on")
    parser.add_argument("--model_type", type=str, default="video", choices=["image", "video"],
                        help="Model type: 'image' for Sam3Image, 'video' for Sam3Video")
    parser.add_argument("--apply_temporal_disambiguation", type=lambda x: x.lower() == "true", default=True,
                        help="Whether to apply temporal disambiguation (video model only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed merge progress")

    args = parser.parse_args()

    print("=" * 60)
    print("SAM3 LoRA Merge Script")
    print("=" * 60)
    print(f"LoRA checkpoint: {args.lora_checkpoint}")
    print(f"Output path: {args.output}")
    print(f"Device: {args.device}")
    print()

    lora_params = load_lora_config_from_yaml(args.config)

    # Check if LoRA checkpoint exists
    if not os.path.exists(args.lora_checkpoint):
        raise FileNotFoundError(f"LoRA checkpoint not found: {args.lora_checkpoint}")

    # Build base SAM3 model
    print(f"\nBuilding SAM3 {'video' if args.model_type == 'video' else 'image'} base model...")

    if args.model_type == "video":
        model = build_sam3_video_model(
            checkpoint_path=args.checkpoint_path,
            bpe_path=args.bpe_path,
            load_from_HF=False,
            apply_temporal_disambiguation=args.apply_temporal_disambiguation,
            device=args.device,
            compile=False,
        )
    else:
        model = build_sam3_image_model(
            device=args.device,
            compile=False,
            checkpoint_path=args.checkpoint_path,
            bpe_path=args.bpe_path,
            eval_mode=True
        )
    model.eval()

    # Load LoRA checkpoint first to detect actual rank
    print("Loading LoRA checkpoint...")
    lora_checkpoint = torch.load(args.lora_checkpoint, map_location="cpu")

    if isinstance(lora_checkpoint, dict) and "lora_state_dict" in lora_checkpoint:
        lora_state_dict = lora_checkpoint["lora_state_dict"]
    elif isinstance(lora_checkpoint, dict) and "model_state_dict" in lora_checkpoint:
        lora_state_dict = lora_checkpoint["model_state_dict"]
    else:
        lora_state_dict = lora_checkpoint

    # Auto-detect rank from checkpoint shape (lora_A: (in_features, rank))
    ckpt_rank = None
    for key, value in lora_state_dict.items():
        if ".lora_A" in key and isinstance(value, torch.Tensor) and value.ndim == 2:
            ckpt_rank = value.shape[1]
            break

    if ckpt_rank is not None and ckpt_rank != lora_params["rank"]:
        print(f"\n⚠️  RANK MISMATCH DETECTED!")
        print(f"   Config says: rank={lora_params['rank']}")
        print(f"   Checkpoint has: rank={ckpt_rank}")
        print(f"   → Using checkpoint rank={ckpt_rank} (overriding config)")
        lora_params["rank"] = ckpt_rank
        lora_params["alpha"] = ckpt_rank * 2
        print(f"   → Estimated alpha={lora_params['alpha']} (default alpha=2×rank)")
    elif ckpt_rank is not None:
        print(f"✅ Checkpoint rank matches config: rank={ckpt_rank}")

    # Create LoRA config and apply to model
    print(f"Applying LoRA configuration (rank={lora_params['rank']}, alpha={lora_params['alpha']})...")
    lora_config = LoRAConfig(
        rank=lora_params["rank"],
        alpha=lora_params["alpha"],
        dropout=0.0,
        target_modules=lora_params["target_modules"],
        apply_to_vision_encoder=lora_params["apply_to_vision_encoder"],
        apply_to_text_encoder=lora_params["apply_to_text_encoder"],
        apply_to_geometry_encoder=lora_params["apply_to_geometry_encoder"],
        apply_to_detr_encoder=lora_params["apply_to_detr_encoder"],
        apply_to_detr_decoder=lora_params["apply_to_detr_decoder"],
        apply_to_mask_decoder=lora_params["apply_to_mask_decoder"],
    )
    model = apply_lora_to_model(model, lora_config)

    # Remap keys: training uses image model (no detector. prefix),
    # but video model nests everything under detector.
    if args.model_type == "video":
        model_keys = set(model.state_dict().keys())
        remapped = {}
        unmapped = 0
        for key, value in lora_state_dict.items():
            if key in model_keys:
                remapped[key] = value
            elif f"detector.{key}" in model_keys:
                remapped[f"detector.{key}"] = value
            else:
                remapped[key] = value
                unmapped += 1
        if unmapped > 0:
            print(f"Warning: {unmapped} keys could not be remapped (no matching key in model)")
        lora_state_dict = remapped

    # Load state dict
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")

    # Merge LoRA weights
    print("\nMerging LoRA weights into base model...")
    model = merge_lora_weights(model)

    # Save as state dict
    print(f"\nSaving merged model to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    state_dict = model.state_dict()
    torch.save(state_dict, args.output)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done! Merged model saved ({file_size:.1f} MB)")
    print("=" * 60)
    print("\nUsage:")
    print(f"  model.load_state_dict(torch.load('{args.output}'))")


if __name__ == "__main__":
    main()
