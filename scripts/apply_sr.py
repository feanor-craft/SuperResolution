"""Apply fine-tuned SPAN x2 SR model to a directory of images.

Typical use: apply SR to 640x480 center-crops from make_test_data.py
to produce 1280x960 outputs for object detection.

Usage:
    python scripts/apply_sr.py \\
        --input_dir  test_data/case1_crop/images \\
        --output_dir test_data/case1_sr/images \\
        --model_path experiments/SPAN_x2_finetune_custom_DIV2K/models/net_g_100000.pth

The label files are NOT modified (640x480 labels remain valid after x2 upscale
because bounding-box coordinates scale proportionally).
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from basicsr.archs.span_arch import SPAN

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

SPAN_CFG = dict(
    num_in_ch=3,
    num_out_ch=3,
    feature_channels=48,
    upscale=2,
    bias=True,
    img_range=255.,
    rgb_mean=[0.4488, 0.4371, 0.4040],
)


def load_model(model_path: str, device: torch.device) -> SPAN:
    model = SPAN(**SPAN_CFG).to(device)
    ckpt = torch.load(model_path, map_location=device)

    # support both 'params_ema' (EMA weight, preferred) and 'params'
    if 'params_ema' in ckpt:
        state_dict = ckpt['params_ema']
    elif 'params' in ckpt:
        state_dict = ckpt['params']
    else:
        state_dict = ckpt  # bare state dict

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f'Loaded model from {model_path}')
    return model


def sr_image(model: SPAN, img_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Run SR on a single BGR uint8 image. Returns BGR uint8 at 2x resolution."""
    img_f = img_bgr.astype(np.float32) / 255.
    # HWC BGR -> 1CHW RGB
    t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
    t = t[:, [2, 1, 0], :, :].to(device)

    with torch.no_grad():
        out = model(t)

    # 1CHW RGB -> HWC BGR uint8
    out_np = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out_np = out_np[:, :, ::-1]  # RGB -> BGR
    return (out_np * 255).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Batch SR inference using fine-tuned SPAN x2')
    parser.add_argument('--input_dir', required=True,
                        help='Directory with input images (e.g. 640x480 center-crops)')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save SR output images (e.g. 1280x960)')
    parser.add_argument('--model_path', required=True,
                        help='Path to fine-tuned .pth checkpoint')
    parser.add_argument('--device', default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--ext', default='png',
                        help='Output image extension (default: png)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = load_model(args.model_path, device)

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in input_path.iterdir()
                        if f.suffix.lower() in IMG_EXTENSIONS])

    if not img_files:
        print(f'[ERROR] No images found in {args.input_dir}')
        return

    total = len(img_files)
    print(f'Processing {total} images...')
    t0 = time.time()

    for i, img_file in enumerate(img_files, 1):
        img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'  [WARN] Cannot read {img_file.name}, skipping')
            continue

        h_in, w_in = img.shape[:2]
        sr = sr_image(model, img, device)
        h_out, w_out = sr.shape[:2]

        save_name = img_file.stem + f'.{args.ext}'
        cv2.imwrite(str(output_path / save_name), sr)

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            print(f'  [{i}/{total}] {img_file.name}: {w_in}x{h_in} -> {w_out}x{h_out} '
                  f'({elapsed:.1f}s elapsed)')

    print(f'\nDone. {total} images saved to {output_path}')
    print(f'Input resolution : check first image')
    if img_files:
        sample = cv2.imread(str(img_files[0]))
        if sample is not None:
            h, w = sample.shape[:2]
            sr_sample = cv2.imread(str(output_path / (img_files[0].stem + f'.{args.ext}')))
            if sr_sample is not None:
                sh, sw = sr_sample.shape[:2]
                print(f'  {img_files[0].name}: {w}x{h} -> {sw}x{sh}')


if __name__ == '__main__':
    main()
