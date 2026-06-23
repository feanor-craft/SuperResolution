"""Test SPAN x2 architecture as described in README.

README states:
- SPAN x2 upscales 640x480 centre-crops to 1280x960 for object detection.
- Network config: num_in_ch=3, num_out_ch=3, feature_channels=48,
  upscale=2, bias=True, img_range=255., rgb_mean=[0.4488, 0.4371, 0.4040]
"""

import pytest

torch = pytest.importorskip('torch')

from basicsr.archs.span_arch import SPAN  # noqa: E402


SPAN_CFG = dict(
    num_in_ch=3,
    num_out_ch=3,
    feature_channels=48,
    upscale=2,
    bias=True,
    img_range=255.,
    rgb_mean=[0.4488, 0.4371, 0.4040],
)


def _make_model(train=False):
    model = SPAN(**SPAN_CFG)
    if train:
        model.train()
    else:
        model.eval()
    return model


def test_span_x2_instantiation():
    """SPAN model should instantiate without errors using README config."""
    model = _make_model()
    assert isinstance(model, SPAN)


def test_span_x2_output_shape_small():
    """Forward pass: small image should be doubled in spatial dimensions."""
    model = _make_model()
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 64, 64), f"Expected (1,3,64,64), got {out.shape}"


def test_span_x2_output_shape_readme_usecase():
    """README use-case: 640x480 centre-crop → 1280x960 after SPAN x2."""
    model = _make_model()
    # Use a smaller proxy (divided by 16) to keep test fast on CPU
    x = torch.rand(1, 3, 30, 40)   # proxy for 480x640 (H x W) / 16
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 60, 80), f"Expected (1,3,60,80), got {out.shape}"


def test_span_x2_output_range():
    """Output pixel values should be in [0, 1] range after clamping in apply_sr."""
    model = _make_model()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        out = model(x)
    # The model itself does not clamp; apply_sr.py does. Just check it's finite.
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_span_x2_train_eval_modes():
    """Model should run in both training and eval modes without error."""
    x = torch.rand(1, 3, 16, 16)

    model_eval = _make_model(train=False)
    with torch.no_grad():
        out_eval = model_eval(x)
    assert out_eval.shape == (1, 3, 32, 32)

    model_train = _make_model(train=True)
    out_train = model_train(x)
    assert out_train.shape == (1, 3, 32, 32)


def test_span_x2_upscale_factor():
    """Upscale factor of 2 is required; sanity check via different input sizes."""
    model = _make_model()
    for h, w in [(8, 8), (16, 24), (20, 30)]:
        x = torch.rand(1, 3, h, w)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, h * 2, w * 2), (
            f"For input ({h},{w}) expected output ({h*2},{w*2}), got {out.shape[2:]}"
        )
