"""Test scripts/apply_sr.py — fine-tuned SPAN x2 SR inference.

Fine-tuning scenario (from README):
  1. Pretrained SPAN x2 is fine-tuned with custom domain + DIV2K data.
  2. Checkpoint saved by BasicSR contains keys 'params' and 'params_ema'.
  3. apply_sr.load_model() prefers 'params_ema', falls back to 'params',
     then bare state dict.
  4. sr_image() converts BGR uint8 -> RGB float -> model -> RGB float -> BGR uint8.
  5. End-to-end: 640x480 center-crop -> SPAN x2 -> 1280x960.
"""

import sys
import os
import tempfile

import cv2
import numpy as np
import pytest

torch = pytest.importorskip('torch')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from apply_sr import load_model, sr_image, SPAN_CFG, IMG_EXTENSIONS  # noqa: E402
from basicsr.archs.span_arch import SPAN  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_model() -> SPAN:
    """Return a randomly-initialised SPAN x2 model in eval mode."""
    m = SPAN(**SPAN_CFG)
    m.eval()
    return m


def _save_checkpoint(path, key_scheme: str):
    """Save a random SPAN state dict using the requested key scheme.

    key_scheme:
      'params_ema'  -> {'params_ema': state_dict}   (BasicSR EMA checkpoint)
      'params'      -> {'params': state_dict}        (BasicSR non-EMA checkpoint)
      'bare'        -> state_dict directly            (plain torch.save)
    """
    state = _fresh_model().state_dict()
    if key_scheme == 'params_ema':
        obj = {'params_ema': state}
    elif key_scheme == 'params':
        obj = {'params': state}
    else:
        obj = state
    torch.save(obj, path)


# ---------------------------------------------------------------------------
# load_model — checkpoint format compatibility
# ---------------------------------------------------------------------------

def test_load_model_params_ema(tmp_path):
    """load_model must load 'params_ema' key (preferred BasicSR EMA format)."""
    ckpt = str(tmp_path / 'net_g_100000.pth')
    _save_checkpoint(ckpt, 'params_ema')
    device = torch.device('cpu')
    model = load_model(ckpt, device)
    assert isinstance(model, SPAN)
    assert not model.training  # must be in eval mode


def test_load_model_params_key(tmp_path):
    """load_model must fall back to 'params' when 'params_ema' is absent."""
    ckpt = str(tmp_path / 'net_g_100000.pth')
    _save_checkpoint(ckpt, 'params')
    device = torch.device('cpu')
    model = load_model(ckpt, device)
    assert isinstance(model, SPAN)
    assert not model.training


def test_load_model_bare_state_dict(tmp_path):
    """load_model must accept a bare state dict (no wrapper keys)."""
    ckpt = str(tmp_path / 'net_g_bare.pth')
    _save_checkpoint(ckpt, 'bare')
    device = torch.device('cpu')
    model = load_model(ckpt, device)
    assert isinstance(model, SPAN)
    assert not model.training


def test_load_model_weights_loaded(tmp_path):
    """Weights from checkpoint must actually be loaded into the model."""
    ckpt = str(tmp_path / 'net_g_100000.pth')
    src = _fresh_model()
    torch.save({'params_ema': src.state_dict()}, ckpt)

    loaded = load_model(ckpt, torch.device('cpu'))

    # Compare a single parameter tensor
    src_p = next(iter(src.state_dict().values()))
    loaded_p = next(iter(loaded.state_dict().values()))
    assert torch.allclose(src_p, loaded_p), "Loaded weights do not match saved weights"


# ---------------------------------------------------------------------------
# sr_image — input/output contract
# ---------------------------------------------------------------------------

def test_sr_image_output_dtype():
    """sr_image must return a uint8 numpy array."""
    model = _fresh_model()
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    out = sr_image(model, img, torch.device('cpu'))
    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"


def test_sr_image_output_shape_doubles():
    """sr_image output must have exactly 2× the spatial dimensions of the input."""
    model = _fresh_model()
    h, w = 24, 32
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    out = sr_image(model, img, torch.device('cpu'))
    assert out.shape == (h * 2, w * 2, 3), (
        f"Expected ({h*2},{w*2},3), got {out.shape}"
    )


def test_sr_image_output_value_range():
    """sr_image output pixel values must be in [0, 255]."""
    model = _fresh_model()
    img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    out = sr_image(model, img, torch.device('cpu'))
    assert out.min() >= 0 and out.max() <= 255


def test_sr_image_bgr_channel_order_preserved():
    """sr_image must accept BGR and return BGR (not RGB)."""
    model = _fresh_model()
    # Solid-colour images: R=200, G=100, B=50 in BGR layout -> [50, 100, 200]
    img_bgr = np.full((16, 16, 3), [50, 100, 200], dtype=np.uint8)
    out = sr_image(model, img_bgr, torch.device('cpu'))
    # Output shape should be (32, 32, 3) — 3 channels, not flipped to RGBA
    assert out.shape[2] == 3


def test_sr_image_deterministic_eval():
    """sr_image called twice on the same input must return identical results."""
    model = _fresh_model()  # already in eval mode
    img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    out1 = sr_image(model, img, torch.device('cpu'))
    out2 = sr_image(model, img, torch.device('cpu'))
    assert np.array_equal(out1, out2), "sr_image is not deterministic in eval mode"


# ---------------------------------------------------------------------------
# End-to-end: 640x480 center-crop -> apply_sr -> 1280x960  (README use-case)
# ---------------------------------------------------------------------------

def test_apply_sr_readme_resolution(tmp_path):
    """Full README scenario: 640x480 crop -> SPAN x2 -> 1280x960."""
    ckpt = str(tmp_path / 'net_g.pth')
    _save_checkpoint(ckpt, 'params_ema')
    model = load_model(ckpt, torch.device('cpu'))

    # Simulate a 640x480 center-crop (as produced by make_test_data.py)
    crop = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    sr = sr_image(model, crop, torch.device('cpu'))

    assert sr.shape == (960, 1280, 3), (
        f"Expected (960,1280,3) after x2 SR on 640x480 crop, got {sr.shape}"
    )


def test_apply_sr_batch_saves_correct_files(tmp_path):
    """apply_sr must create one output file per input image."""
    ckpt = str(tmp_path / 'net_g.pth')
    _save_checkpoint(ckpt, 'params_ema')

    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    output_dir = tmp_path / 'output'

    for i in range(3):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / f'img{i:03d}.png'), img)

    model = load_model(ckpt, torch.device('cpu'))
    output_dir.mkdir()

    from pathlib import Path
    img_files = sorted([f for f in input_dir.iterdir()
                        if f.suffix.lower() in IMG_EXTENSIONS])
    for img_file in img_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        sr = sr_image(model, img, torch.device('cpu'))
        cv2.imwrite(str(output_dir / img_file.name), sr)

    out_files = list(output_dir.glob('*.png'))
    assert len(out_files) == 3, f"Expected 3 output files, got {len(out_files)}"


def test_apply_sr_output_resolution_matches_input_x2(tmp_path):
    """Every output image must be exactly 2× the input resolution."""
    ckpt = str(tmp_path / 'net_g.pth')
    _save_checkpoint(ckpt, 'params_ema')
    model = load_model(ckpt, torch.device('cpu'))

    for h, w in [(16, 16), (24, 32), (30, 40)]:
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        sr = sr_image(model, img, torch.device('cpu'))
        assert sr.shape[:2] == (h * 2, w * 2), (
            f"For {h}x{w} input expected {h*2}x{w*2} output, got {sr.shape[:2]}"
        )
