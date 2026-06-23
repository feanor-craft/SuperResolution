# SPAN x2 Super-Resolution — 학습 및 테스트 매뉴얼

> **목표**: FHD(1920×1080) 영상에서 객체 중심 640×480 센터크롭을 SPAN x2 SR로 1280×960으로
> 업스케일하여 객체검출 모델의 입력 품질을 향상시킵니다.
>
> 사전학습된 SPAN x2 모델을 도메인 특화 데이터로 파인튜닝하고,
> 파인튜닝된 모델로 실제 영상 프레임에 SR을 적용합니다.

```
전체 파이프라인 요약:

[FHD 영상 프레임]
       │
       ▼ make_test_data.py --case crop
[640×480 센터크롭 + 라벨 변환]
       │
       ▼ apply_sr.py  (파인튜닝된 SPAN x2 사용)
[1280×960 SR 결과]
       │
       ▼
[객체검출 모델 입력]
```

---

## 목차

1. [환경 설정](#1-환경-설정)
2. [데이터셋 준비 (파인튜닝용)](#2-데이터셋-준비-파인튜닝용)
3. [Pretrained 모델 배치](#3-pretrained-모델-배치)
4. [LR 이미지 생성](#4-lr-이미지-생성)
5. [Fine-tuning 학습](#5-fine-tuning-학습)
6. [모델 품질 평가 (PSNR/SSIM)](#6-모델-품질-평가-psnrssim)
7. [실제 영상에 SR 적용 (640×480 → 1280×960)](#7-실제-영상에-sr-적용-640480--1280960)
8. [학습 재개 (Resume)](#8-학습-재개-resume)
9. [설정 파일 주요 파라미터](#9-설정-파일-주요-파라미터)
10. [자주 발생하는 오류 및 해결법](#10-자주-발생하는-오류-및-해결법)

---

## 1. 환경 설정

### 요구 사항

| 항목 | 버전 |
|------|------|
| Python | 3.10 이상 (3.11 권장) |
| PyTorch | 2.1 이상 |
| CUDA | 12.1 이상 |

### 설치

```bash
# uv 사용 시
uv venv .venv --python 3.11
source .venv/Scripts/activate          # Windows
# source .venv/bin/activate            # Linux/Mac

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install "numpy<2"
uv pip install -r requirements.txt
BASICSR_EXT=False python setup.py develop
```

설치 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import basicsr; print('BasicSR OK')"
```

---

## 2. 데이터셋 준비 (파인튜닝용)

파인튜닝에는 SR로 복원하고자 하는 도메인과 유사한 **고해상도(HR) 이미지**가 필요합니다.

> **핵심 개념**: SR 모델은 (LR, HR) 쌍으로 학습합니다.
> LR은 HR을 bicubic x2 다운샘플링하여 자동 생성합니다.
> 실제 640×480 프레임을 학습에 직접 사용하는 것이 아니라,
> 도메인 대표 HR 이미지를 수집하여 학습합니다.

```
datasets/
├── domain_maritime/
│   └── HR/          ← 해양 도메인 HR 이미지 (권장: 800장 이상)
├── domain_ground/
│   └── HR/
├── domain_air2ground/
│   └── HR/
├── domain_ground2air/
│   └── HR/
└── DIV2K/
    └── DIV2K_train_HR/   ← 범용 SR 데이터 (선택, 800장)
```

- HR 이미지: 최소 256×256 이상, 실제 도메인 촬영 이미지 사용 권장
- LR 이미지(`LR/X2/`)는 다음 단계에서 자동 생성됩니다

---

## 3. Pretrained 모델 배치

파인튜닝 시작점인 SPAN x2 공개 사전학습 가중치를 배치합니다.

```
experiments/
└── pretrained_models/
    └── SPAN_x2_pretrained.pth    ← SPAN 공식 레포에서 다운로드
```

SPAN GitHub: https://github.com/hongyuanyu/SPAN (Releases 또는 Google Drive)

```bash
mkdir -p experiments/pretrained_models
mv SPAN_x2_pretrained.pth experiments/pretrained_models/
```

---

## 4. LR 이미지 생성

HR 이미지를 bicubic x2 다운샘플링하여 학습용 LR 이미지를 합성합니다.

```bash
# 전체 도메인
python scripts/make_lr.py

# 특정 도메인만
python scripts/make_lr.py --domains domain_maritime DIV2K
```

---

## 5. Fine-tuning 학습

사전학습 모델을 도메인 특화 데이터로 파인튜닝합니다.

### 설정 파일: `options/train/SPAN/train_SPAN_x2_finetune.yml`

```yaml
path:
  pretrain_network_g: experiments/pretrained_models/SPAN_x2_pretrained.pth  # 사전학습 모델

datasets:
  train:
    dataroot_gt:             # HR 폴더 목록 (도메인 순서대로)
      - datasets/domain_maritime/HR
      - datasets/domain_ground/HR
      - datasets/domain_air2ground/HR
      - datasets/domain_ground2air/HR
      - datasets/DIV2K/DIV2K_train_HR
    dataroot_lq:             # LR 폴더 목록 (위와 순서 일치)
      - datasets/domain_maritime/LR/X2
      - datasets/domain_ground/LR/X2
      - datasets/domain_air2ground/LR/X2
      - datasets/domain_ground2air/LR/X2
      - datasets/DIV2K/DIV2K_train_LR_bicubic/X2

train:
  total_iter: 100000
  optim_g:
    lr: !!float 2e-5    # 파인튜닝용 낮은 학습률
```

### 학습 실행

```bash
python basicsr/train.py -opt options/train/SPAN/train_SPAN_x2_finetune.yml
```

### 학습 출력

```
experiments/SPAN_x2_finetune_custom_DIV2K/
├── models/
│   ├── net_g_10000.pth
│   ├── net_g_20000.pth   ← 파인튜닝된 체크포인트
│   └── net_g_100000.pth
├── training_states/      ← Resume용
└── log/
```

### 학습 모니터링

```bash
tensorboard --logdir tb_logger/SPAN_x2_finetune_custom_DIV2K
# http://localhost:6006
```

---

## 6. 모델 품질 평가 (PSNR/SSIM)

파인튜닝된 모델의 SR 품질을 도메인별로 정량 평가합니다.

### 설정 파일 수정: `options/test/SPAN/test_SPAN_x2.yml`

```yaml
path:
  pretrain_network_g: experiments/SPAN_x2_finetune_custom_DIV2K/models/net_g_100000.pth
```

### 실행

```bash
python basicsr/test.py -opt options/test/SPAN/test_SPAN_x2.yml
```

### 결과 확인

```
Validation Maritime
     # psnr: 32.45    Best: 32.45 @ 100000 iter
     # ssim: 0.9012   Best: 0.9012 @ 100000 iter
```

SR 결과 이미지: `results/test_SPAN_x2_finetuned/Maritime/`

---

## 7. 실제 영상에 SR 적용 (640×480 → 1280×960)

파인튜닝된 모델로 FHD 영상 프레임에 SR을 적용하는 **전체 추론 파이프라인**입니다.

### Step 1: FHD 영상에서 640×480 센터크롭 추출

FHD(1920×1080) 프레임에서 객체 중심으로 640×480을 잘라내고,
바운딩박스 라벨도 크롭 좌표로 변환합니다.

```bash
python scripts/make_test_data.py \
    --input_dir  path/to/fhd_frames \
    --label_dir  path/to/labels \
    --output_dir test_data/case1_crop \
    --case crop
```

출력:
```
test_data/case1_crop/
├── images/    ← 640×480 센터크롭 이미지
└── labels/    ← 크롭 좌표 기준 YOLO 라벨
```

### Step 2: 파인튜닝 모델로 SR 적용 (640×480 → 1280×960)

```bash
python scripts/apply_sr.py \
    --input_dir  test_data/case1_crop/images \
    --output_dir test_data/case1_sr/images \
    --model_path experiments/SPAN_x2_finetune_custom_DIV2K/models/net_g_100000.pth \
    --device cuda
```

출력:
```
test_data/case1_sr/images/   ← 1280×960 SR 결과
```

> **라벨 호환성**: SR 적용 후 이미지가 2배 커지므로 라벨의 정규화 좌표(cx, cy, w, h)는
> 그대로 유효합니다. `test_data/case1_crop/labels/`를 그대로 사용하면 됩니다.

### 케이스별 비교

| Case | 스크립트 | 입력 | SR | 검출 입력 |
|------|---------|------|-----|---------|
| Case 1 | make_test_data.py + apply_sr.py | 640×480 크롭 | 파인튜닝 모델 | 1280×960 |
| Case 2 | make_test_data.py + apply_sr.py | 640×480 크롭 | 사전학습 모델만 | 1280×960 |
| Case 3 | make_test_data.py --case baseline | FHD 원본 | 미적용 | 1920×1080 |

Case 2 (사전학습만 사용):
```bash
python scripts/apply_sr.py \
    --input_dir  test_data/case1_crop/images \
    --output_dir test_data/case2_sr/images \
    --model_path experiments/pretrained_models/SPAN_x2_pretrained.pth \
    --device cuda
```

Case 3 (베이스라인):
```bash
python scripts/make_test_data.py \
    --input_dir  path/to/fhd_frames \
    --label_dir  path/to/labels \
    --output_dir test_data/case3_baseline \
    --case baseline
```

---

## 8. 학습 재개 (Resume)

```yaml
# train yml에 추가
auto_resume: true
```

또는 특정 checkpoint:

```yaml
path:
  resume_state: experiments/SPAN_x2_finetune_custom_DIV2K/training_states/50000.state
```

---

## 9. 설정 파일 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `gt_size` | 128 | 학습 패치 크기 (GT 기준, LQ = 64) |
| `batch_size_per_gpu` | 16 | GPU당 배치 크기 |
| `dataset_enlarge_ratio` | 100 | 에포크당 데이터 반복 배수 |
| `total_iter` | 100000 | 전체 학습 반복 수 |
| `lr` | 2e-5 | 파인튜닝 학습률 |
| `milestones` | [50000] | LR 절반 감소 시점 |
| `val_freq` | 5000 | 검증 주기 |
| `save_checkpoint_freq` | 10000 | 체크포인트 저장 주기 |

---

## 10. 자주 발생하는 오류 및 해결법

### `ModuleNotFoundError: No module named 'basicsr'`

```bash
BASICSR_EXT=False python setup.py develop
```

### `FileNotFoundError: SPAN_x2_pretrained.pth`

Section 3을 참고하여 `experiments/pretrained_models/`에 파일을 배치하세요.

### `AssertionError: lq and gt datasets have different number of images`

```bash
python scripts/make_lr.py --domains domain_maritime  # LR 재생성
```

### `CUDA out of memory`

```yaml
batch_size_per_gpu: 8   # 16에서 줄이기
```

---

## 빠른 시작 요약

```bash
# 1. 환경 설치
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install "numpy<2" -r requirements.txt
BASICSR_EXT=False python setup.py develop

# 2. 도메인 HR 이미지 준비 후 LR 생성
python scripts/make_lr.py

# 3. 사전학습 모델 배치
#    → experiments/pretrained_models/SPAN_x2_pretrained.pth

# 4. 파인튜닝 학습
python basicsr/train.py -opt options/train/SPAN/train_SPAN_x2_finetune.yml

# 5. (선택) 품질 평가
python basicsr/test.py -opt options/test/SPAN/test_SPAN_x2.yml

# 6. 실제 영상에 SR 적용 (640×480 → 1280×960)
python scripts/make_test_data.py \
    --input_dir path/to/fhd_frames --label_dir path/to/labels \
    --output_dir test_data/case1_crop --case crop

python scripts/apply_sr.py \
    --input_dir  test_data/case1_crop/images \
    --output_dir test_data/case1_sr/images \
    --model_path experiments/SPAN_x2_finetune_custom_DIV2K/models/net_g_100000.pth \
    --device cuda
```


---

## 목차

1. [환경 설정](#1-환경-설정)
2. [데이터셋 준비](#2-데이터셋-준비)
3. [Pretrained 모델 배치](#3-pretrained-모델-배치)
4. [LR 이미지 생성](#4-lr-이미지-생성)
5. [Fine-tuning 학습](#5-fine-tuning-학습)
6. [모델 테스트 (PSNR/SSIM 평가)](#6-모델-테스트-psnrssim-평가)
7. [추론 (단일 이미지 SR)](#7-추론-단일-이미지-sr)
8. [테스트 데이터 준비 (객체검출 파이프라인)](#8-테스트-데이터-준비-객체검출-파이프라인)
9. [학습 재개 (Resume)](#9-학습-재개-resume)
10. [설정 파일 주요 파라미터](#10-설정-파일-주요-파라미터)
11. [자주 발생하는 오류 및 해결법](#11-자주-발생하는-오류-및-해결법)

---

## 1. 환경 설정

### 요구 사항

| 항목 | 버전 |
|------|------|
| Python | 3.10 이상 |
| PyTorch | 2.1 이상 |
| CUDA | 11.8 또는 12.1 |

### 설치

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. BasicSR 개발 모드 설치 (CUDA extension 제외)
python setup.py develop --no_cuda_ext
```

> **주의**: `python setup.py develop`은 반드시 프로젝트 루트 디렉토리에서 실행해야 합니다.

설치 확인:

```bash
python -c "import basicsr; print('BasicSR OK')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 2. 데이터셋 준비

### 디렉토리 구조

데이터셋은 프로젝트 루트 기준 `datasets/` 폴더에 위치해야 합니다.

```
datasets/
├── domain_maritime/
│   └── HR/          ← 해양 도메인 HR 이미지 (권장: 800장 이상)
├── domain_ground/
│   └── HR/
├── domain_air2ground/
│   └── HR/
├── domain_ground2air/
│   └── HR/
└── DIV2K/
    └── DIV2K_train_HR/   ← DIV2K 공식 800장
```

- HR 이미지 포맷: PNG, JPG, JPEG, BMP, TIF 모두 지원
- LR 이미지(`LR/X2/`)는 다음 단계에서 자동 생성됩니다.

### DIV2K 다운로드 (선택)

```bash
# DIV2K HR 이미지 공식 다운로드
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip -d datasets/DIV2K/
```

---

## 3. Pretrained 모델 배치

Fine-tuning의 시작점인 SPAN x2 사전학습 가중치를 아래 경로에 배치합니다.

```
experiments/
└── pretrained_models/
    └── SPAN_x2_pretrained.pth    ← 여기에 배치
```

SPAN 공식 레포지토리에서 다운로드:
- [SPAN GitHub](https://github.com/hongyuanyu/SPAN) → Releases 또는 Google Drive 링크 참조

```bash
mkdir -p experiments/pretrained_models
# 다운로드 후 이동
mv SPAN_x2_pretrained.pth experiments/pretrained_models/
```

---

## 4. LR 이미지 생성

HR 이미지를 bicubic x2 다운샘플링하여 LR 이미지를 합성합니다.

### 전체 도메인 생성

```bash
python scripts/make_lr.py
```

### 특정 도메인만 생성

```bash
python scripts/make_lr.py --domains domain_maritime DIV2K
```

실행 후 디렉토리 구조:

```
datasets/
├── domain_maritime/
│   ├── HR/
│   └── LR/
│       └── X2/    ← 생성됨
├── domain_ground/
│   ├── HR/
│   └── LR/
│       └── X2/    ← 생성됨
...
```

---

## 5. Fine-tuning 학습

### 학습 설정 파일

`options/train/SPAN/train_SPAN_x2_finetune.yml`

주요 설정:

```yaml
datasets:
  train:
    dataroot_gt:            # GT(HR) 폴더 목록
      - datasets/domain_maritime/HR
      - datasets/DIV2K/DIV2K_train_HR
      # ... (5개 도메인)
    dataroot_lq:            # LQ(LR) 폴더 목록 (위와 순서 일치)
      - datasets/domain_maritime/LR/X2
      - datasets/DIV2K/DIV2K_train_LR_bicubic/X2
    gt_size: 128            # 학습 패치 크기 (GT 기준)
    batch_size_per_gpu: 16

path:
  pretrain_network_g: experiments/pretrained_models/SPAN_x2_pretrained.pth

train:
  total_iter: 100000
  optim_g:
    lr: !!float 2e-5        # Fine-tuning용 낮은 LR
```

### 학습 실행

```bash
python basicsr/train.py -opt options/train/SPAN/train_SPAN_x2_finetune.yml
```

### 학습 출력 위치

```
experiments/
└── SPAN_x2_finetune_custom_DIV2K/
    ├── models/
    │   ├── net_g_10000.pth
    │   ├── net_g_20000.pth
    │   └── ...
    ├── training_states/      ← Resume용 상태 파일
    ├── log/                  ← 학습 로그
    └── visualization/        ← val 이미지 (save_img: true 시)
```

### 학습 모니터링 (TensorBoard)

```bash
tensorboard --logdir tb_logger/SPAN_x2_finetune_custom_DIV2K
```

브라우저에서 `http://localhost:6006` 접속

### 멀티 GPU 학습

```bash
# 2 GPU 예시
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=4321 \
    basicsr/train.py \
    -opt options/train/SPAN/train_SPAN_x2_finetune.yml \
    --launcher pytorch
```

---

## 6. 모델 테스트 (PSNR/SSIM 평가)

### 테스트 설정 파일 수정

`options/test/SPAN/test_SPAN_x2.yml`에서 모델 경로를 수정합니다.

```yaml
path:
  pretrain_network_g: experiments/pretrained_models/SPAN_x2_finetuned.pth
  # 또는 학습된 특정 체크포인트:
  # pretrain_network_g: experiments/SPAN_x2_finetune_custom_DIV2K/models/net_g_100000.pth
```

### 테스트 실행

```bash
python basicsr/test.py -opt options/test/SPAN/test_SPAN_x2.yml
```

### 결과 출력 위치

```
results/
└── test_SPAN_x2_finetuned/
    ├── Maritime/
    │   ├── 0001_test_SPAN_x2_finetuned.png
    │   └── ...
    ├── Ground/
    ├── Air2Ground/
    ├── Ground2Air/
    ├── DIV2K/
    └── log/
        └── test_test_SPAN_x2_finetuned_<timestamp>.log
```

### PSNR/SSIM 결과 확인

로그 파일 또는 콘솔 출력에서 도메인별 지표 확인:

```
Validation Maritime
     # psnr: 32.45    Best: 32.45 @ 100000 iter
     # ssim: 0.9012   Best: 0.9012 @ 100000 iter
```

---

## 7. 추론 (단일 이미지 SR)

학습된 모델로 임의의 이미지에 SR을 적용합니다.

```python
import torch
import cv2
import numpy as np
from basicsr.archs.span_arch import SPAN

# 모델 로드
model = SPAN(
    num_in_ch=3, num_out_ch=3,
    feature_channels=48, upscale=2,
    bias=True, img_range=255.,
    rgb_mean=[0.4488, 0.4371, 0.4040]
)
ckpt = torch.load('experiments/pretrained_models/SPAN_x2_finetuned.pth', map_location='cpu')
model.load_state_dict(ckpt['params_ema'])
model.eval()

# 이미지 로드 및 추론
img = cv2.imread('input.png').astype(np.float32) / 255.
img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)[:, [2,1,0]]  # BGR->RGB

with torch.no_grad():
    sr = model(img_t)

sr_np = sr.squeeze(0).permute(1, 2, 0)[[2,1,0]].clamp(0, 1).numpy()
cv2.imwrite('output_sr.png', (sr_np * 255).astype(np.uint8))
```

---

## 8. 테스트 데이터 준비 (객체검출 파이프라인)

FHD 비디오 프레임에서 객체검출 평가용 테스트 데이터를 생성합니다.

| Case | 설명 | SR 적용 | 검출 입력 해상도 |
|------|------|---------|-----------------|
| Case 1/2 | 센터크롭 640×480 | SPAN x2 적용 | 1280×960 |
| Case 3 | FHD 원본 | 미적용 | 1920×1080 (리사이즈) |

### Case 1/2: 크롭 후 SR

```bash
python scripts/make_test_data.py \
    --input_dir path/to/fhd_frames \
    --label_dir path/to/labels \
    --output_dir test_data/case1_crop \
    --case crop
```

### Case 3: FHD 베이스라인

```bash
python scripts/make_test_data.py \
    --input_dir path/to/fhd_frames \
    --label_dir path/to/labels \
    --output_dir test_data/case3_baseline \
    --case baseline
```

---

## 9. 학습 재개 (Resume)

학습이 중단된 경우 마지막 체크포인트에서 재개합니다.

### 자동 재개

`train_SPAN_x2_finetune.yml`에서 `auto_resume: true` 설정 시 자동으로 최신 state를 불러옵니다.

```yaml
# yml 최상단에 추가
auto_resume: true
```

### 수동 재개

```yaml
path:
  resume_state: experiments/SPAN_x2_finetune_custom_DIV2K/training_states/50000.state
```

```bash
python basicsr/train.py -opt options/train/SPAN/train_SPAN_x2_finetune.yml
```

---

## 10. 설정 파일 주요 파라미터

### 학습 설정 (`train_SPAN_x2_finetune.yml`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `gt_size` | 128 | 학습 패치 크기 (GT 기준, LQ는 64) |
| `batch_size_per_gpu` | 16 | GPU당 배치 크기 |
| `dataset_enlarge_ratio` | 100 | 에포크 당 데이터 반복 배수 |
| `total_iter` | 100000 | 전체 학습 이터레이션 수 |
| `lr` | 2e-5 | 초기 학습률 |
| `milestones` | [50000] | LR 감소 시점 (gamma: 0.5) |
| `ema_decay` | 0.999 | EMA 감쇠율 |
| `val_freq` | 5000 | 검증 주기 (iterations) |
| `save_checkpoint_freq` | 10000 | 체크포인트 저장 주기 |

### 네트워크 구조 (`network_g`)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `feature_channels` | 48 | 중간 특징 채널 수 |
| `upscale` | 2 | 업스케일 배수 |
| `img_range` | 255. | 이미지 픽셀 범위 |
| `rgb_mean` | [0.4488, 0.4371, 0.4040] | ImageNet RGB 평균 |

---

## 11. 자주 발생하는 오류 및 해결법

### `ModuleNotFoundError: No module named 'basicsr'`

```bash
# 프로젝트 루트에서 재설치
python setup.py develop --no_cuda_ext
```

### `FileNotFoundError: ... SPAN_x2_pretrained.pth`

`experiments/pretrained_models/SPAN_x2_pretrained.pth` 파일이 없습니다.
[3단계](#3-pretrained-모델-배치)를 참고하여 가중치를 배치하세요.

### `AssertionError: lq and gt datasets have different number of images`

LR 폴더와 HR 폴더의 파일 수가 다릅니다.

```bash
# LR 재생성
python scripts/make_lr.py --domains domain_maritime
```

### `CUDA out of memory`

`train_SPAN_x2_finetune.yml`에서 배치 크기를 줄입니다:

```yaml
batch_size_per_gpu: 8   # 16 → 8
```

### `RuntimeError: Expected all tensors to be on the same device`

단일 GPU 환경에서는 `num_gpu: 1` 확인:

```yaml
num_gpu: 1
```

---

## 빠른 시작 요약

```bash
# 1. 설치
pip install -r requirements.txt && python setup.py develop --no_cuda_ext

# 2. HR 이미지를 datasets/ 에 배치 후 LR 생성
python scripts/make_lr.py

# 3. pretrained weight 배치
# experiments/pretrained_models/SPAN_x2_pretrained.pth

# 4. 학습
python basicsr/train.py -opt options/train/SPAN/train_SPAN_x2_finetune.yml

# 5. 테스트
# test yml에서 모델 경로 수정 후:
python basicsr/test.py -opt options/test/SPAN/test_SPAN_x2.yml
```
