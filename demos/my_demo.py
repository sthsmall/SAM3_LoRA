import os
import glob
import json
import torch
import numpy as np
import cv2
import pycocotools.mask as mask_utils
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    render_masklet_frame,
    save_masklet_video,
)


DATASET_ROOT = "/root/autodl-tmp/laboratory_data/"
CHECKPOINT_PATH = "/root/autodl-tmp/sam3_merged.pt"

TEXT_PROMPT = "plane"
OUTPUT_DIR = "./demo/tracking_results"


def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
    return polygons


def mask_to_rle(mask):
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def save_mot_format(outputs_per_frame, image_files, output_path):
    mot_lines = []
    for frame_idx in sorted(outputs_per_frame.keys()):
        frame_outputs = outputs_per_frame[frame_idx]
        obj_ids = frame_outputs.get("out_obj_ids", [])
        boxes_xywh = frame_outputs.get("out_boxes_xywh", [])
        probs = frame_outputs.get("out_probs", [])

        if len(obj_ids) == 0:
            continue

        img = load_frame(image_files[frame_idx])
        height, width = img.shape[:2]

        for obj_id, box, prob in zip(obj_ids, boxes_xywh, probs):
            x, y, w, h = box
            if isinstance(x, np.ndarray):
                x, y, w, h = float(x), float(y), float(w), float(h)
            else:
                x, y, w, h = x, y, w, h
            
            x1 = int(x * width)
            y1 = int(y * height)
            w = int(w * width)
            h = int(h * height)
            
            # conf = float(prob) if isinstance(prob, (np.ndarray, np.floating)) else prob
            conf = 1

            frame_num = frame_idx + 1
            track_id = int(obj_id)
            mot_lines.append(f"{frame_num},{track_id},{x1},{y1},{w},{h},{conf},1,1")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(mot_lines))
    print(f"  MOT格式已保存: {output_path}")


def save_coco_format(outputs_per_frame, image_files, output_path):
    images = []
    annotations = []
    ann_id = 1

    for frame_idx in sorted(outputs_per_frame.keys()):
        frame_outputs = outputs_per_frame[frame_idx]
        obj_ids = frame_outputs.get("out_obj_ids", [])
        boxes_xywh = frame_outputs.get("out_boxes_xywh", [])
        binary_masks = frame_outputs.get("out_binary_masks", [])

        img = load_frame(image_files[frame_idx])
        height, width = img.shape[:2]

        img_id = frame_idx
        images.append({
            "id": img_id,
            "file_name": os.path.basename(image_files[frame_idx]),
            "height": height,
            "width": width
        })

        if len(obj_ids) == 0:
            continue

        for obj_id, box, mask in zip(obj_ids, boxes_xywh, binary_masks):
            x, y, w, h = box
            if isinstance(x, np.ndarray):
                x, y, w, h = float(x), float(y), float(w), float(h)
            else:
                x, y, w, h = x, y, w, h
            
            x_abs = int(x * width)
            y_abs = int(y * height)
            w_abs = int(w * width)
            h_abs = int(h * height)
            
            mask_h, mask_w = mask.shape[-2:]
            if mask.ndim == 3:
                mask = mask[0]
            
            mask_resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            polygons = mask_to_polygon(mask_resized)
            if not polygons:
                continue
            area = int(np.sum(mask_resized))
            
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_abs, y_abs, w_abs, h_abs],
                "area": area,
                "segmentation": polygons,
                "iscrowd": 0
            })
            ann_id += 1

    categories = [{"id": 1, "name": "object"}]

    coco_result = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_result, f, indent=2)
    print(f"  COCO格式已保存: {output_path}")


def get_sequences(dataset_root, split="train"):
    seq_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(seq_dir):
        print(f"目录不存在: {seq_dir}")
        return []
    sequences = sorted(
        [d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))]
    )
    return [(name, os.path.join(seq_dir, name)) for name in sequences]


def process_sequence(predictor, seq_name, seq_path, output_dir, text_prompt):
    import configparser
    
    img_dir = os.path.join(seq_path, "img1")
    if not os.path.isdir(img_dir):
        print(f"  跳过 {seq_name}: 没有 img1 目录")
        return

    seqinfo_file = os.path.join(seq_path, "seqinfo.ini")
    fps = 30
    if os.path.isfile(seqinfo_file):
        config = configparser.ConfigParser()
        config.read(seqinfo_file)
        fps = config.getint("Sequence", "frameRate", fallback=30)

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not image_files:
        print(f"  跳过 {seq_name}: 没有图像文件")
        return

    print(f"  序列 {seq_name}: {len(image_files)} 帧, fps={fps}")

    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=img_dir,
            offload_video_to_cpu=True,# ---
        )
    )
    session_id = response["session_id"]
    print(f"  会话已创建: {session_id} (offload_video_to_cpu=True)") #----

    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text_prompt,
        )
    )
    add_prompt_output = response["outputs"]
    num_detected = len(add_prompt_output.get("out_obj_ids", []))
    print(f"  首帧检测到 {num_detected} 个目标")

    # if num_detected == 0:
    #     print(f"  未检测到目标，跳过传播")
    #     predictor.handle_request(
    #         request=dict(type="reset_session", session_id=session_id)
    #     )
    #     predictor.handle_request(
    #         request=dict(type="close_session", session_id=session_id)
    #     )
    #     torch.cuda.empty_cache()
    #     return

    outputs_per_frame = {}
    CHUNK_SIZE = 100
    for frame_data in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            # max_frame_num_to_track=CHUNK_SIZE,
        )
    ):
        fi = frame_data["frame_index"]
        outputs_per_frame[fi] = frame_data["outputs"]
        outputs = frame_data["outputs"]
        # outputs_cpu = {}
        # for key, value in outputs.items():
        #     if isinstance(value, torch.Tensor):
        #         outputs_cpu[key] = value.cpu().numpy()
        #     elif isinstance(value, list):
        #         outputs_cpu[key] = [
        #             v.cpu().numpy() if isinstance(v, torch.Tensor) else v
        #             for v in value
        #         ]
        #     else:
        #         outputs_cpu[key] = value
        # outputs_per_frame[fi] = outputs_cpu
        del outputs, frame_data

    print(f"  传播完成: {len(outputs_per_frame)} 帧")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{seq_name}_tracking.mp4")

    save_masklet_video(image_files, outputs_per_frame, output_path, alpha=0.5, fps=fps)
    print(f"  视频已保存: {output_path}")

    mot_path = os.path.join(output_dir, f"{seq_name}_mot.txt")
    save_mot_format(outputs_per_frame, image_files, mot_path)

    coco_path = os.path.join(output_dir, f"{seq_name}_coco.json")
    save_coco_format(outputs_per_frame, image_files, coco_path)

    predictor.handle_request(
        request=dict(type="reset_session", session_id=session_id)
    )
    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print(f"  会话已关闭，显存已释放")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("加载模型...")
    num_gpus = torch.cuda.device_count()
    kwargs = dict(checkpoint_path=CHECKPOINT_PATH)
    if num_gpus > 0:
        kwargs["gpus_to_use"] = list(range(num_gpus))
    predictor = build_sam3_video_predictor(**kwargs)
    print("模型加载完成")

    for subset in ["AIR-aircraft"]:
        dataset_path = os.path.join(DATASET_ROOT, subset)
        if not os.path.isdir(dataset_path):
            print(f"数据集目录不存在: {dataset_path}")
            continue

        print(f"\n{'='*60}")
        print(f"处理数据集: {subset}")
        print(f"{'='*60}")

        for split in ["train"]:
            sequences = get_sequences(dataset_path, split)
            if not sequences:
                continue
            flag = True
            print(f"\n--- {split} 集: {len(sequences)} 个序列 ---")
            for i, (seq_name, seq_path) in enumerate[tuple[str, str]](sequences):
                # if "43" not in seq_name and flag:
                #     continue
                # if "43" in seq_name:
                #     flag = False
                #     continue

                print(f"\n[{i+1}/{len(sequences)}] {seq_name}")
                out_dir = os.path.join(OUTPUT_DIR, subset, split)
                process_sequence(predictor, seq_name, seq_path, out_dir, TEXT_PROMPT)


if __name__ == "__main__":
    main()
