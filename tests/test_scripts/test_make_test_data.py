"""Test scripts/make_test_data.py — test-data preparation for SR evaluation.

README states:
- Case 1 & 2 (SR applied): center-crop 640x480 around object center from FHD frame.
  Crop is then fed to SPAN x2 SR → 1280x960 for object detection.
- Case 3 (baseline): FHD original (1920x1080) passed as-is to object detection.
- Bounding-box labels are transformed to the cropped coordinate space (YOLO format).
- Labels that fall entirely outside the crop are discarded.
"""

import json
import os
import sys
import cv2
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from make_test_data import (  # noqa: E402
    load_labels,
    get_object_center,
    compute_crop_region,
    transform_labels,
    process_case_crop,
    process_case_baseline,
    CROP_W, CROP_H,
    FHD_W, FHD_H,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_crop_dimensions_match_readme():
    """README specifies 640x480 center-crop; constants must reflect this."""
    assert CROP_W == 640
    assert CROP_H == 480


def test_fhd_dimensions_match_readme():
    """README specifies FHD source at 1920x1080."""
    assert FHD_W == 1920
    assert FHD_H == 1080


# ---------------------------------------------------------------------------
# load_labels — YOLO .txt format
# ---------------------------------------------------------------------------

def test_load_labels_yolo(tmp_path):
    """YOLO normalised format (cls cx cy w h) is parsed into absolute pixel coords."""
    label_file = tmp_path / 'frame.txt'
    # One object at the center of FHD frame, 10% width/height
    label_file.write_text('0 0.5 0.5 0.1 0.1\n')

    labels = load_labels(str(label_file))
    assert len(labels) == 1
    lb = labels[0]
    assert lb['class_id'] == 0

    # absolute values
    expected_cx = 0.5 * FHD_W   # 960
    expected_cy = 0.5 * FHD_H   # 540
    expected_w  = 0.1 * FHD_W   # 192
    expected_h  = 0.1 * FHD_H   # 108
    x1, y1, x2, y2 = lb['bbox']
    assert abs((x1 + x2) / 2 - expected_cx) < 1e-3
    assert abs((y1 + y2) / 2 - expected_cy) < 1e-3
    assert abs((x2 - x1) - expected_w) < 1e-3
    assert abs((y2 - y1) - expected_h) < 1e-3


def test_load_labels_yolo_multiple(tmp_path):
    """All YOLO lines in the file must be loaded."""
    label_file = tmp_path / 'multi.txt'
    label_file.write_text('0 0.2 0.3 0.05 0.05\n1 0.8 0.7 0.1 0.1\n')

    labels = load_labels(str(label_file))
    assert len(labels) == 2
    assert labels[0]['class_id'] == 0
    assert labels[1]['class_id'] == 1


def test_load_labels_yolo_empty_file(tmp_path):
    """Empty YOLO file returns an empty list."""
    label_file = tmp_path / 'empty.txt'
    label_file.write_text('')
    assert load_labels(str(label_file)) == []


def test_load_labels_json_list(tmp_path):
    """JSON list format is parsed correctly."""
    label_file = tmp_path / 'frame.json'
    data = [{'class_id': 2, 'bbox': [100, 200, 300, 400]}]
    label_file.write_text(json.dumps(data))

    labels = load_labels(str(label_file))
    assert len(labels) == 1
    assert labels[0]['class_id'] == 2
    assert labels[0]['bbox'] == [100, 200, 300, 400]


# ---------------------------------------------------------------------------
# get_object_center
# ---------------------------------------------------------------------------

def test_get_object_center_single_box():
    """Center is the midpoint of the single bounding box."""
    labels = [{'class_id': 0, 'bbox': [100, 200, 300, 400]}]
    cx, cy = get_object_center(labels)
    assert cx == 200   # (100+300)/2
    assert cy == 300   # (200+400)/2


def test_get_object_center_multiple_boxes():
    """Center is the mean of each box's midpoint."""
    labels = [
        {'class_id': 0, 'bbox': [0, 0, 100, 100]},   # center (50, 50)
        {'class_id': 0, 'bbox': [100, 100, 300, 300]},  # center (200, 200)
    ]
    cx, cy = get_object_center(labels)
    assert cx == int((50 + 200) / 2)   # 125
    assert cy == int((50 + 200) / 2)   # 125


def test_get_object_center_no_labels():
    """When no labels exist the function falls back to image center (FHD)."""
    cx, cy = get_object_center([])
    assert cx == FHD_W // 2
    assert cy == FHD_H // 2


# ---------------------------------------------------------------------------
# compute_crop_region
# ---------------------------------------------------------------------------

def test_compute_crop_region_center():
    """Crop centered on object at image center stays inside boundaries."""
    cx, cy = FHD_W // 2, FHD_H // 2
    x1, y1, x2, y2 = compute_crop_region(cx, cy, CROP_W, CROP_H, FHD_W, FHD_H)
    assert x2 - x1 == CROP_W
    assert y2 - y1 == CROP_H
    assert x1 >= 0 and x2 <= FHD_W
    assert y1 >= 0 and y2 <= FHD_H


def test_compute_crop_region_top_left_clamp():
    """Object near top-left corner: crop must be clamped to (0,0)."""
    x1, y1, x2, y2 = compute_crop_region(0, 0, CROP_W, CROP_H, FHD_W, FHD_H)
    assert x1 == 0
    assert y1 == 0
    assert x2 == CROP_W
    assert y2 == CROP_H


def test_compute_crop_region_bottom_right_clamp():
    """Object at bottom-right corner: crop must be clamped to image edge."""
    x1, y1, x2, y2 = compute_crop_region(FHD_W, FHD_H, CROP_W, CROP_H, FHD_W, FHD_H)
    assert x2 == FHD_W
    assert y2 == FHD_H
    assert x2 - x1 == CROP_W
    assert y2 - y1 == CROP_H


def test_compute_crop_region_output_size():
    """Crop region must always have width=CROP_W and height=CROP_H."""
    for cx, cy in [(10, 10), (960, 540), (1900, 1050)]:
        x1, y1, x2, y2 = compute_crop_region(cx, cy, CROP_W, CROP_H, FHD_W, FHD_H)
        assert x2 - x1 == CROP_W, f"Width wrong for ({cx},{cy})"
        assert y2 - y1 == CROP_H, f"Height wrong for ({cx},{cy})"


# ---------------------------------------------------------------------------
# transform_labels
# ---------------------------------------------------------------------------

def test_transform_labels_box_fully_inside():
    """Box entirely inside the crop must be preserved and correctly normalised."""
    # Crop: x1=200, y1=200, 640x480
    labels = [{'class_id': 0, 'bbox': [250, 250, 450, 400]}]
    result = transform_labels(labels, crop_x1=200, crop_y1=200,
                              crop_w=CROP_W, crop_h=CROP_H)
    assert len(result) == 1
    ncx, ncy, nw, nh = result[0]['bbox_norm']
    # Verify center is within [0,1]
    assert 0 < ncx < 1
    assert 0 < ncy < 1
    assert 0 < nw <= 1
    assert 0 < nh <= 1


def test_transform_labels_box_fully_outside():
    """Box entirely outside the crop must be discarded."""
    labels = [{'class_id': 0, 'bbox': [0, 0, 10, 10]}]   # well outside crop at (1000,1000)
    result = transform_labels(labels, crop_x1=1000, crop_y1=1000,
                              crop_w=CROP_W, crop_h=CROP_H)
    assert result == []


def test_transform_labels_normalisation():
    """Output bbox_norm values must be normalised relative to crop dimensions."""
    # Box exactly fills the crop
    labels = [{'class_id': 1, 'bbox': [100, 100, 740, 580]}]  # 640x480
    result = transform_labels(labels, crop_x1=100, crop_y1=100,
                              crop_w=CROP_W, crop_h=CROP_H)
    assert len(result) == 1
    ncx, ncy, nw, nh = result[0]['bbox_norm']
    assert abs(ncx - 0.5) < 1e-6
    assert abs(ncy - 0.5) < 1e-6
    assert abs(nw - 1.0) < 1e-6
    assert abs(nh - 1.0) < 1e-6


def test_transform_labels_class_id_preserved():
    """class_id must be passed through unchanged."""
    labels = [{'class_id': 7, 'bbox': [300, 300, 500, 450]}]
    result = transform_labels(labels, crop_x1=200, crop_y1=200,
                              crop_w=CROP_W, crop_h=CROP_H)
    assert result[0]['class_id'] == 7


# ---------------------------------------------------------------------------
# process_case_crop (end-to-end)
# ---------------------------------------------------------------------------

def _make_fhd_image(path):
    """Write a synthetic FHD (1920x1080) image."""
    img = np.random.randint(0, 256, (FHD_H, FHD_W, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_process_case_crop_output_size(tmp_path):
    """Center-cropped image must be exactly 640x480 (CROP_W x CROP_H)."""
    input_dir = tmp_path / 'input'
    label_dir = tmp_path / 'labels'
    out_img   = tmp_path / 'out' / 'images'
    out_lbl   = tmp_path / 'out' / 'labels'
    input_dir.mkdir(); label_dir.mkdir()

    _make_fhd_image(input_dir / 'frame001.png')
    # label: object at FHD center
    (label_dir / 'frame001.txt').write_text('0 0.5 0.5 0.1 0.1\n')

    process_case_crop(str(input_dir), str(label_dir),
                      str(out_img), str(out_lbl))

    out = cv2.imread(str(out_img / 'frame001.png'))
    assert out is not None
    h, w = out.shape[:2]
    assert w == CROP_W, f"Expected width={CROP_W}, got {w}"
    assert h == CROP_H, f"Expected height={CROP_H}, got {h}"


def test_process_case_crop_label_file_created(tmp_path):
    """A YOLO .txt label must be written for every processed image."""
    input_dir = tmp_path / 'input'
    label_dir = tmp_path / 'labels'
    out_img   = tmp_path / 'out' / 'images'
    out_lbl   = tmp_path / 'out' / 'labels'
    input_dir.mkdir(); label_dir.mkdir()

    _make_fhd_image(input_dir / 'frame001.png')
    (label_dir / 'frame001.txt').write_text('0 0.5 0.5 0.1 0.1\n')

    process_case_crop(str(input_dir), str(label_dir),
                      str(out_img), str(out_lbl))

    assert (out_lbl / 'frame001.txt').exists()


def test_process_case_crop_no_label_file(tmp_path):
    """Frames without labels are still processed (no crash, empty label file)."""
    input_dir = tmp_path / 'input'
    label_dir = tmp_path / 'labels'
    out_img   = tmp_path / 'out' / 'images'
    out_lbl   = tmp_path / 'out' / 'labels'
    input_dir.mkdir(); label_dir.mkdir()

    _make_fhd_image(input_dir / 'unlabelled.png')

    process_case_crop(str(input_dir), str(label_dir),
                      str(out_img), str(out_lbl))

    out = cv2.imread(str(out_img / 'unlabelled.png'))
    assert out is not None
    assert out.shape[:2] == (CROP_H, CROP_W)


# ---------------------------------------------------------------------------
# process_case_baseline (end-to-end)
# ---------------------------------------------------------------------------

def test_process_case_baseline_preserves_resolution(tmp_path):
    """Baseline case must copy FHD images without any resize or crop."""
    input_dir = tmp_path / 'input'
    label_dir = tmp_path / 'labels'
    out_img   = tmp_path / 'out' / 'images'
    out_lbl   = tmp_path / 'out' / 'labels'
    input_dir.mkdir(); label_dir.mkdir()

    _make_fhd_image(input_dir / 'frame001.png')
    (label_dir / 'frame001.txt').write_text('0 0.5 0.5 0.1 0.1\n')

    process_case_baseline(str(input_dir), str(label_dir),
                          str(out_img), str(out_lbl))

    out = cv2.imread(str(out_img / 'frame001.png'))
    assert out is not None
    h, w = out.shape[:2]
    assert w == FHD_W, f"Expected FHD width={FHD_W}, got {w}"
    assert h == FHD_H, f"Expected FHD height={FHD_H}, got {h}"


def test_process_case_baseline_copies_label(tmp_path):
    """Baseline case must copy the original label file unchanged."""
    input_dir = tmp_path / 'input'
    label_dir = tmp_path / 'labels'
    out_img   = tmp_path / 'out' / 'images'
    out_lbl   = tmp_path / 'out' / 'labels'
    input_dir.mkdir(); label_dir.mkdir()

    original_label = '0 0.5 0.5 0.2 0.2\n'
    _make_fhd_image(input_dir / 'frame001.png')
    (label_dir / 'frame001.txt').write_text(original_label)

    process_case_baseline(str(input_dir), str(label_dir),
                          str(out_img), str(out_lbl))

    copied = (out_lbl / 'frame001.txt').read_text()
    assert copied == original_label


# ---------------------------------------------------------------------------
# Pipeline consistency: SR factor check
# ---------------------------------------------------------------------------

def test_sr_doubles_crop_to_readme_resolution():
    """After SPAN x2 SR, 640x480 crop must become 1280x960 as stated in README."""
    sr_scale = 2
    sr_w = CROP_W * sr_scale
    sr_h = CROP_H * sr_scale
    assert sr_w == 1280, f"SR width should be 1280, got {sr_w}"
    assert sr_h == 960,  f"SR height should be 960, got {sr_h}"
