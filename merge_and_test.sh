#!/bin/bash
# 
# Merge LoRA to SAM3 Video Model and Test
# This script merges LoRA weights into SAM3 video model and validates the checkpoint
#

set -e

echo "=========================================="
echo "SAM3 LoRA Merge & Test Pipeline"
echo "=========================================="
echo

# Default values
LORA_CHECKPOINT="${LORA_CHECKPOINT:-outputs/sam3_lora_test/best_lora_weights.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-sam3_merged_video.pt}"
MODEL_TYPE="${MODEL_TYPE:-video}"
APPLY_TEMPORAL_DISAMBIGUATION="${APPLY_TEMPORAL_DISAMBIGUATION:-True}"
CONFIG_FILE="${CONFIG_FILE:-configs/full_lora_config.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/d/projects/specific/sam3/sam3/sam3.pt}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lora_checkpoint)
            LORA_CHECKPOINT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --apply_temporal_disambiguation)
            APPLY_TEMPORAL_DISAMBIGUATION="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  LoRA checkpoint: $LORA_CHECKPOINT"
echo "  Output path: $OUTPUT_PATH"
echo "  Model type: $MODEL_TYPE"
echo "  Temporal disambiguation: $APPLY_TEMPORAL_DISAMBIGUATION"
echo "  Training config: $CONFIG_FILE"
echo "  Base checkpoint: $CHECKPOINT_PATH"
echo

# Check if LoRA checkpoint exists
if [ ! -f "$LORA_CHECKPOINT" ]; then
    echo "❌ Error: LoRA checkpoint not found: $LORA_CHECKPOINT"
    echo "Please provide a valid LoRA checkpoint using --lora_checkpoint argument"
    exit 1
fi

# Step 1: Merge LoRA weights
echo "=========================================="
echo "Step 1: Merging LoRA weights"
echo "=========================================="
python merge_lora_to_base.py \
    --lora_checkpoint "$LORA_CHECKPOINT" \
    --output "$OUTPUT_PATH" \
    --model_type "$MODEL_TYPE" \
    --apply_temporal_disambiguation "$APPLY_TEMPORAL_DISAMBIGUATION" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config "$CONFIG_FILE"

if [ ! -f "$OUTPUT_PATH" ]; then
    echo "❌ Error: Failed to create merged checkpoint"
    exit 1
fi

echo "✅ Merged checkpoint created: $OUTPUT_PATH"
echo

# Step 2: Test merged model with video predictor
echo "=========================================="
echo "Step 2: Testing merged checkpoint"
echo "=========================================="
python tests/test_merged_video_predictor.py \
    --checkpoint "$OUTPUT_PATH" \
    --apply_temporal_disambiguation "$APPLY_TEMPORAL_DISAMBIGUATION"

echo

# Step 3: Numerical correctness test (merged model output == LoRA model output)
echo "=========================================="
echo "Step 3: Numerical correctness test"
echo "=========================================="
python tests/test_merged_model.py \
    --checkpoint "$OUTPUT_PATH" \
    --config "$CONFIG_FILE" \
    --lora_checkpoint "$LORA_CHECKPOINT" \
    --base_checkpoint "$CHECKPOINT_PATH"

echo
echo "=========================================="
echo "✅ All steps completed successfully!"
echo "=========================================="
echo
echo "Your merged checkpoint is ready at: $OUTPUT_PATH"
echo
echo "You can use it in your demo like this:"
echo "  CHECKPOINT_PATH = \"$OUTPUT_PATH\""