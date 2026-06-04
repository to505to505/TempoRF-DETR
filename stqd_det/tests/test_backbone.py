"""Backbone (ResNet-50 + FPN) tests."""

from pathlib import Path

import pytest
import torch

from stqd_det.backbone import FPN_LEVELS, build_backbone

# Skip the heavy "pretrained" test if torchvision can't fetch weights and they
# aren't cached locally - otherwise CI without network would crash.
torch_home_cached = (Path(torch.hub.get_dir()) / "checkpoints").glob("resnet50-*.pth")
_HAS_PRETRAINED = any(True for _ in torch_home_cached)


def test_forward_shapes_random_init():
    bb = build_backbone(out_channels=256, pretrained=False, frozen_bn=False)
    bb.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        pyramid = bb(x)
    assert list(pyramid.keys()) == FPN_LEVELS
    expected_strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}
    H = W = 256
    for name, feat in pyramid.items():
        s = expected_strides[name]
        assert feat.shape == (2, 256, H // s, W // s), f"{name}: {tuple(feat.shape)}"


def test_level_strides_attribute():
    bb = build_backbone(pretrained=False, frozen_bn=False)
    assert bb.level_strides == {"P2": 4, "P3": 8, "P4": 16, "P5": 32}


def test_frozen_bn_puts_bn_layers_in_eval_with_no_grad():
    bb = build_backbone(pretrained=False, frozen_bn=True)
    bb.train()  # put module in training mode; BN must STILL be frozen
    bn_count = 0
    for m in bb.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_count += 1
            assert not m.training
            assert all(not p.requires_grad for p in m.parameters())
    assert bn_count > 0, "expected ResNet-50 to contain BN2d layers"


def test_gradients_flow_through_trainable_conv():
    bb = build_backbone(pretrained=False, frozen_bn=True)
    bb.train()
    x = torch.randn(1, 3, 128, 128, requires_grad=False)
    loss = sum(p.sum() for p in bb(x).values())
    loss.backward()
    # First trainable conv (stem conv1) should receive non-zero gradient.
    grad_norm = bb.stem[0].weight.grad.abs().sum().item()
    assert grad_norm > 0


@pytest.mark.skipif(not _HAS_PRETRAINED, reason="ResNet-50 pretrained weights not cached")
def test_pretrained_weights_loaded_when_requested():
    """Sanity: pretrained init differs from random init."""
    bb_rand = build_backbone(pretrained=False, frozen_bn=False)
    bb_pre = build_backbone(pretrained=True, frozen_bn=False)
    w1 = bb_rand.stem[0].weight
    w2 = bb_pre.stem[0].weight
    assert not torch.allclose(w1, w2)
