"""Generate meta_info_mixed_x2.txt for all domain + DIV2K HR images."""

import argparse
import os
from pathlib import Path

import cv2

DOMAINS = {
    'domain_maritime': 'datasets/domain_maritime/HR',
    'domain_ground': 'datasets/domain_ground/HR',
    'domain_air2ground': 'datasets/domain_air2ground/HR',
    'domain_ground2air': 'datasets/domain_ground2air/HR',
    'DIV2K': 'datasets/DIV2K/DIV2K_train_HR',
}

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
OUTPUT_FILE = 'datasets/meta_info/meta_info_mixed_x2.txt'


def main():
    parser = argparse.ArgumentParser(description='Generate meta_info file for mixed domain dataset')
    parser.add_argument('--output', default=OUTPUT_FILE, help='Output meta_info file path')
    parser.add_argument('--domains', nargs='+', default=None, help='Domains to include (default: all)')
    args = parser.parse_args()

    targets = DOMAINS
    if args.domains:
        targets = {k: v for k, v in DOMAINS.items() if k in args.domains}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, 'w') as f:
        for domain, hr_dir in targets.items():
            hr_path = Path(hr_dir)
            if not hr_path.exists():
                print(f'[SKIP] {domain}: directory not found ({hr_dir})')
                continue

            images = sorted([p for p in hr_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f'  [WARN] Cannot read {img_path}')
                    continue
                h, w, c = img.shape
                f.write(f'{img_path.name} ({h},{w},{c})\n')
                total += 1

            print(f'[{domain}] {len(images)} images added')

    print(f'\nTotal: {total} images -> {output_path}')


if __name__ == '__main__':
    main()
