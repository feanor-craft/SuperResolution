# SuperResolution - SPAN x2 Fine-tuning Pipeline

SPAN(Swift Parameter-free Attention Network) 기반 x2 Super-Resolution 학습/테스트 파이프라인.

FHD 영상에서 객체 중심 센터크롭(640x480)을 SPAN x2 SR로 1280x960으로 업스케일한 뒤
객체검출 모델의 입력으로 활용하는 것을 목표로 합니다.

## 환경 설정

### 요구 사항
- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8 또는 12.1

### 설치

```bash
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

## 데이터셋 구조

```
datasets/
├── domain_maritime/
│   ├── HR/          # 해양 도메인 (800장)
│   └── LR/X2/       # bicubic x2 다운샘플링 (자동 생성)
├── domain_ground/
│   ├── HR/          # 지상 도메인 (800장)
│   └── LR/X2/
├── domain_air2ground/
│   ├── HR/          # 공대지 도메인 (800장)
│   └── LR/X2/
├── domain_ground2air/
│   ├── HR/          # 지대공 도메인 (800장)
│   └── LR/X2/
└── DIV2K/
    ├── DIV2K_train_HR/           # DIV2K (800장)
    └── DIV2K_train_LR_bicubic/X2/
```

## 파이프라인

### 1. LR 이미지 생성

HR 이미지를 bicubic x2 다운샘플링하여 LR 이미지를 합성합니다.

```bash
python scripts/make_lr.py
```

특정 도메인만 처리:
```bash
python scripts/make_lr.py --domains domain_maritime DIV2K
```

### 2. Fine-tuning

사전학습된 SPAN x2 모델을 4개 커스텀 도메인 + DIV2K 혼합 데이터로 fine-tuning합니다.

```bash
# Pretrained weight를 배치
# experiments/pretrained_models/SPAN_x2_pretrained.pth

# 학습 실행
python basicsr/train.py -opt options/train/SPAN/train_SPAN_x2_finetune.yml
```

학습 파라미터:
| 항목 | 값 |
|------|-----|
| Learning Rate | 2e-5 |
| Total Iterations | 100,000 |
| Scheduler | MultiStepLR (milestone: 50000, gamma: 0.5) |
| Loss | L1Loss |
| Batch Size | 16 |
| EMA Decay | 0.999 |

> **EMA (Exponential Moving Average):** 매 iteration마다 `EMA_weight = 0.999 × EMA_weight + 0.001 × current_weight` 수식으로 별도의 EMA 가중치(`net_g_ema`)를 유지합니다.
> 학습 중 파라미터 진동을 평탄화해 더 안정적인 추론 품질을 제공합니다.
> 체크포인트에는 일반 가중치(`params`)와 EMA 가중치(`params_ema`) 두 가지가 함께 저장되며, 추론 시에는 `params_ema`를 우선 사용합니다.

### 3. 테스트

Fine-tuned 모델로 각 도메인별 SR 품질을 평가합니다.

```bash
python basicsr/test.py -opt options/test/SPAN/test_SPAN_x2.yml
```

### 4. 테스트 데이터 준비

FHD 영상에서 객체검출 평가용 테스트 데이터를 생성합니다.

**Case 1 & 2 (SR 적용):** 객체 중심 640x480 센터크롭

```bash
python scripts/make_test_data.py \
    --input_dir path/to/fhd_frames \
    --label_dir path/to/labels \
    --output_dir test_data/case1_crop \
    --case crop
```

크롭된 640x480 이미지에 SPAN x2 SR을 적용하여 1280x960으로 업스케일 후 객체검출 모델에 입력합니다.

**Case 3 (SR 미적용 베이스라인):** FHD 원본 사용

```bash
python scripts/make_test_data.py \
    --input_dir path/to/fhd_frames \
    --label_dir path/to/labels \
    --output_dir test_data/case3_baseline \
    --case baseline
```

FHD 원본 1920x1080을 그대로 객체검출 모델에 입력합니다.

### 5. SR 배치 적용

Fine-tuned 모델로 640x480 크롭 이미지에 SR을 일괄 적용하여 1280x960 결과물을 생성합니다.

```bash
python scripts/apply_sr.py \
    --input_dir  test_data/case1_crop/images \
    --output_dir test_data/case1_sr/images \
    --model_path experiments/SPAN_x2_finetune_custom_DIV2K/models/net_g_100000.pth
```

- 체크포인트에서 `params_ema`(EMA 가중치)를 우선 로드하며, 없으면 `params`로 폴백합니다.
- `--device cpu` 옵션으로 CPU 추론도 가능합니다.

## 테스트 케이스 요약

| Case | 입력 | SR 적용 | 검출 입력 해상도 |
|------|------|---------|-----------------|
| Case 1 | 센터크롭 640x480 | SPAN x2 (커스텀 도메인 fine-tuned) | 1280x960 |
| Case 2 | 센터크롭 640x480 | SPAN x2 (DIV2K fine-tuned) | 1280x960 |
| Case 3 | FHD 원본 | 미적용 | 1920x1080 (리사이즈) |

## 평가 지표

### SR 품질
- **PSNR** (crop_border: 2)
- **SSIM** (crop_border: 2)

### 객체검출 성능
- mAP@0.5
- mAP@0.5:0.95
- AP_small

## 모델 아키텍처

```yaml
network_g:
  type: SPAN
  num_in_ch: 3
  num_out_ch: 3
  feature_channels: 48
  upscale: 2
  bias: true
  img_range: 255.
  rgb_mean: [0.4488, 0.4371, 0.4040]
```

## 프로젝트 구조

```
SuperResolution/
├── basicsr/                  # BasicSR 프레임워크
│   ├── archs/
│   │   └── span_arch.py      # SPAN 네트워크 아키텍처
│   ├── data/                 # 데이터셋 로더
│   ├── losses/               # Loss 함수
│   ├── metrics/              # PSNR, SSIM 등
│   └── models/               # 학습/테스트 모델
├── options/
│   ├── train/SPAN/           # 학습 설정
│   └── test/SPAN/            # 테스트 설정
├── scripts/
│   ├── make_lr.py            # LR 합성 스크립트
│   ├── make_test_data.py     # 테스트 데이터 준비
│   └── apply_sr.py           # 배치 SR 추론 스크립트
├── experiments/
│   └── pretrained_models/    # Pretrained weights
└── datasets/                 # 데이터셋 (gitignore)
```

## 베이스라인 모델

- [SPAN 원본 레포지토리](https://github.com/hongyuanyu/SPAN)
- [BasicSR 프레임워크](https://github.com/xinntao/BasicSR)

## License

This project is released under the [Apache 2.0 license](LICENSE.txt).
