"""Synthesize LR images from HR images using bicubic x2 downsampling."""

import argparse
import cv2
import os
from pathlib import Path


DOMAINS = {
    'domain_maritime': 'datasets/domain_maritime/HR',
    'domain_ground': 'datasets/domain_ground/HR',
    'domain_air2ground': 'datasets/domain_air2ground/HR',
    'domain_ground2air': 'datasets/domain_ground2air/HR',
    'DIV2K': 'datasets/DIV2K/DIV2K_train_HR',
}

SCALE = 2
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def make_lr(hr_dir, lr_dir, scale):
    hr_path = Path(hr_dir)
    lr_path = Path(lr_dir)
    lr_path.mkdir(parents=True, exist_ok=True)

    hr_images = sorted([
        f for f in hr_path.iterdir()
        if f.suffix.lower() in IMG_EXTENSIONS
    ])

    if not hr_images:
        print(f'  [SKIP] No images found in {hr_dir}')
        return

    for img_file in hr_images:
        img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'  [WARN] Failed to read {img_file}')
            continue

        h, w = img.shape[:2]
        lr_h, lr_w = h // scale, w // scale
        img_lr = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        save_path = lr_path / img_file.name
        cv2.imwrite(str(save_path), img_lr)

    print(f'  Generated {len(hr_images)} LR images -> {lr_dir}')


def main():
    parser = argparse.ArgumentParser(description='Generate LR images via bicubic downsampling')
    parser.add_argument('--scale', type=int, default=SCALE, help='Downsampling scale factor')
    parser.add_argument('--domains', nargs='+', default=None,
                        help='Specific domains to process (default: all)')
    args = parser.parse_args()

    targets = DOMAINS
    if args.domains:
        targets = {k: v for k, v in DOMAINS.items() if k in args.domains}

    for domain, hr_dir in targets.items():
        if domain == 'DIV2K':
            lr_dir = f'datasets/DIV2K/DIV2K_train_LR_bicubic/X{args.scale}'
        else:
            lr_dir = f'datasets/{domain}/LR/X{args.scale}'

        print(f'Processing [{domain}]')
        print(f'  HR: {hr_dir}')
        print(f'  LR: {lr_dir}')

        if not Path(hr_dir).exists():
            print(f'  [SKIP] HR directory not found: {hr_dir}')
            continue

        make_lr(hr_dir, lr_dir, args.scale)

    print('\nDone.')


if __name__ == '__main__':
    main()
