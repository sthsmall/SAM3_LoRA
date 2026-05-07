#!/usr/bin/env python3
"""
Test the merged SAM3+LoRA video model using build_sam3_video_predictor.
This verifies that the merged checkpoint is compatible with the video inference pipeline.

Usage:
    python test_merged_video_predictor.py --checkpoint sam3_merged_video.pt
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

os.makedirs("test_outputs", exist_ok=True)


def test_video_predictor(checkpoint_path: str, apply_temporal_disambiguation: bool = False):
    """Test loading merged checkpoint with Sam3VideoPredictor."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Testing Video Predictor")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Apply temporal disambiguation: {apply_temporal_disambiguation}")
    print()
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        print("1. Loading model with build_sam3_video_predictor...")
        from sam3.model_builder import build_sam3_video_predictor
        
        bpe_path = os.path.join(os.path.dirname(__file__), "..", "bpe_simple_vocab_16e6.txt.gz")
        
        predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            gpus_to_use=[0] if torch.cuda.is_available() else None,
        )
        
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(predictor.model).__name__}")
        print(f"   Total parameters: {sum(p.numel() for p in predictor.model.parameters()):,}")
        
        # Test basic inference on a dummy image/video
        print("\n2. Testing basic inference...")
        
        import numpy as np
        from PIL import Image
        
        # Create dummy test image
        dummy_img_path = "test_outputs/dummy_test.jpg"
        dummy_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        dummy_img.save(dummy_img_path)
        print(f"   Created dummy image: {dummy_img_path}")
        
        # Start a session
        print("\n3. Starting inference session...")
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path="test_outputs",
                offload_video_to_cpu=True,
            )
        )
        session_id = response["session_id"]
        print(f"   Session created: {session_id}")
        
        # Add text prompt
        print("\n4. Adding text prompt...")
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text="object",
            )
        )
        
        outputs = response.get("outputs", {})
        num_objects = len(outputs.get("out_obj_ids", []))
        print(f"   Detected {num_objects} objects")
        
        # Close session
        print("\n5. Cleaning up...")
        predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print(f"\nMerged checkpoint is compatible with Sam3VideoPredictor!")
        print(f"Location: {os.path.abspath(checkpoint_path)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test merged SAM3+LoRA video model")
    parser.add_argument("--checkpoint", type=str, default="sam3_merged_video.pt",
                        help="Path to merged checkpoint")
    parser.add_argument("--apply_temporal_disambiguation", type=lambda x: x.lower() == "true",
                        default=False, help="Enable temporal disambiguation")
    
    args = parser.parse_args()
    
    success = test_video_predictor(
        checkpoint_path=args.checkpoint,
        apply_temporal_disambiguation=args.apply_temporal_disambiguation
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()