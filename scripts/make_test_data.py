"""Prepare test data from FHD video frames with bounding box labels.

Case 1 & 2: Center-crop 640x480 around object center from FHD frames, then
            apply SR to produce 1280x960.
Case 3:     Use original FHD (1920x1080) frames as-is for baseline.

Bounding box labels are transformed to match the cropped region.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


CROP_W, CROP_H = 640, 480
FHD_W, FHD_H = 1920, 1080


def load_labels(label_path):
    """Load bounding box labels.

    Supports YOLO format (class cx cy w h, normalized) and JSON format.
    Returns list of dicts: [{'class_id': int, 'bbox': [x1, y1, x2, y2]}]
    where bbox is in absolute pixel coordinates.
    """
    labels = []
    ext = Path(label_path).suffix.lower()

    if ext == '.txt':
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                abs_cx = cx * FHD_W
                abs_cy = cy * FHD_H
                abs_w = w * FHD_W
                abs_h = h * FHD_H
                x1 = abs_cx - abs_w / 2
                y1 = abs_cy - abs_h / 2
                x2 = abs_cx + abs_w / 2
                y2 = abs_cy + abs_h / 2
                labels.append({'class_id': cls_id, 'bbox': [x1, y1, x2, y2]})
    elif ext == '.json':
        with open(label_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                labels.append({
                    'class_id': item.get('class_id', 0),
                    'bbox': item['bbox']
                })
        elif 'annotations' in data:
            for item in data['annotations']:
                bbox = item.get('bbox', [])
                if len(bbox) == 4:
                    if 'x' in str(type(bbox[0])):
                        x1, y1, w, h = bbox
                        labels.append({
                            'class_id': item.get('category_id', 0),
                            'bbox': [x1, y1, x1 + w, y1 + h]
                        })
                    else:
                        labels.append({
                            'class_id': item.get('category_id', 0),
                            'bbox': bbox
                        })
    return labels


def get_object_center(labels):
    """Compute the center of all bounding boxes."""
    if not labels:
        return FHD_W // 2, FHD_H // 2

    all_cx, all_cy = [], []
    for lb in labels:
        x1, y1, x2, y2 = lb['bbox']
        all_cx.append((x1 + x2) / 2)
        all_cy.append((y1 + y2) / 2)

    return int(np.mean(all_cx)), int(np.mean(all_cy))


def compute_crop_region(cx, cy, crop_w, crop_h, img_w, img_h):
    """Compute crop region clamped to image boundaries."""
    x1 = max(0, min(cx - crop_w // 2, img_w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, img_h - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def transform_labels(labels, crop_x1, crop_y1, crop_w, crop_h):
    """Transform bounding box labels to cropped coordinate space.

    Clips boxes to crop region and discards boxes that fall entirely outside.
    Output format: YOLO normalized (class cx cy w h).
    """
    transformed = []
    for lb in labels:
        x1, y1, x2, y2 = lb['bbox']
        nx1 = max(0, x1 - crop_x1)
        ny1 = max(0, y1 - crop_y1)
        nx2 = min(crop_w, x2 - crop_x1)
        ny2 = min(crop_h, y2 - crop_y1)

        if nx2 <= nx1 or ny2 <= ny1:
            continue

        ncx = ((nx1 + nx2) / 2) / crop_w
        ncy = ((ny1 + ny2) / 2) / crop_h
        nw = (nx2 - nx1) / crop_w
        nh = (ny2 - ny1) / crop_h

        transformed.append({
            'class_id': lb['class_id'],
            'bbox_norm': [ncx, ncy, nw, nh]
        })
    return transformed


def save_labels_yolo(labels, save_path):
    """Save labels in YOLO format."""
    with open(save_path, 'w') as f:
        for lb in labels:
            cx, cy, w, h = lb['bbox_norm']
            f.write(f"{lb['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def process_case_crop(input_dir, label_dir, output_img_dir, output_label_dir):
    """Case 1 & 2: Center-crop 640x480 around object center."""
    input_path = Path(input_dir)
    label_path = Path(label_dir)
    out_img = Path(output_img_dir)
    out_lbl = Path(output_label_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    img_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    ])

    for img_file in img_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'  [WARN] Failed to read {img_file}')
            continue

        img_h, img_w = img.shape[:2]

        lbl_file = label_path / (img_file.stem + '.txt')
        if not lbl_file.exists():
            lbl_file = label_path / (img_file.stem + '.json')

        labels = load_labels(str(lbl_file)) if lbl_file.exists() else []
        cx, cy = get_object_center(labels)

        crop_x1, crop_y1, crop_x2, crop_y2 = compute_crop_region(
            cx, cy, CROP_W, CROP_H, img_w, img_h
        )

        cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
        cv2.imwrite(str(out_img / img_file.name), cropped)

        new_labels = transform_labels(labels, crop_x1, crop_y1, CROP_W, CROP_H)
        save_labels_yolo(new_labels, str(out_lbl / (img_file.stem + '.txt')))

    print(f'  Processed {len(img_files)} images -> {output_img_dir}')


def process_case_baseline(input_dir, label_dir, output_img_dir, output_label_dir):
    """Case 3: Copy FHD originals as-is (no crop, no resize)."""
    input_path = Path(input_dir)
    label_path = Path(label_dir)
    out_img = Path(output_img_dir)
    out_lbl = Path(output_label_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    img_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    ])

    for img_file in img_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        cv2.imwrite(str(out_img / img_file.name), img)

        for ext in ('.txt', '.json'):
            lbl_file = label_path / (img_file.stem + ext)
            if lbl_file.exists():
                import shutil
                shutil.copy2(str(lbl_file), str(out_lbl / lbl_file.name))
                break

    print(f'  Copied {len(img_files)} images -> {output_img_dir}')


def main():
    parser = argparse.ArgumentParser(description='Prepare test data for SR evaluation')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing FHD source images')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Directory containing bounding box labels (YOLO txt or JSON)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory for test data')
    parser.add_argument('--case', type=str, required=True, choices=['crop', 'baseline'],
                        help='crop: Case 1/2 (640x480 center-crop), baseline: Case 3 (FHD as-is)')
    args = parser.parse_args()

    output_img_dir = os.path.join(args.output_dir, 'images')
    output_label_dir = os.path.join(args.output_dir, 'labels')

    if args.case == 'crop':
        print(f'[Case 1/2] Center-crop {CROP_W}x{CROP_H} around object center')
        process_case_crop(args.input_dir, args.label_dir, output_img_dir, output_label_dir)
    else:
        print(f'[Case 3] Baseline - FHD original {FHD_W}x{FHD_H}')
        process_case_baseline(args.input_dir, args.label_dir, output_img_dir, output_label_dir)

    print('Done.')


if __name__ == '__main__':
    main()
