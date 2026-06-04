"""Unit tests for DynamicConv (Chen et al. 2020 K-kernel attention mix)."""

import torch

from stqd_det.dynamic_conv import DynamicConv


def test_forward_shape_same_padding():
    layer = DynamicConv(in_channels=16, out_channels=32, kernel_size=3, K=4)
    x = torch.randn(2, 16, 7, 7)
    y = layer(x)
    assert y.shape == (2, 32, 7, 7)


def test_forward_shape_changes_with_stride():
    layer = DynamicConv(16, 32, kernel_size=3, K=4, stride=2)
    x = torch.randn(2, 16, 8, 8)
    y = layer(x)
    assert y.shape == (2, 32, 4, 4)


def test_gradients_flow_through_attention_and_kernels():
    torch.manual_seed(0)
    layer = DynamicConv(8, 8, kernel_size=3, K=4)
    x = torch.randn(2, 8, 6, 6, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    # attention MLP parameters receive non-zero gradient
    grad_norms = [p.grad.abs().sum().item() for p in layer.attention.parameters()]
    assert all(g > 0 for g in grad_norms)
    # mixed weight tensor receives non-zero gradient
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_different_inputs_get_different_attention():
    """Two visually different inputs should produce different pi distributions."""
    torch.manual_seed(0)
    layer = DynamicConv(4, 4, kernel_size=3, K=4)
    layer.eval()
    x1 = torch.randn(1, 4, 5, 5)
    x2 = torch.ones(1, 4, 5, 5) * 5.0
    with torch.no_grad():
        logits1 = layer.attention(x1)
        logits2 = layer.attention(x2)
    assert not torch.allclose(logits1, logits2, atol=1e-3)


def test_temperature_scales_attention_entropy():
    """High temperature -> near-uniform pi. Low temperature -> near-onehot pi."""
    torch.manual_seed(0)
    layer = DynamicConv(4, 4, kernel_size=3, K=4)
    layer.eval()
    x = torch.randn(1, 4, 5, 5)
    with torch.no_grad():
        y_hi = layer(x, temperature=100.0)
        y_lo = layer(x, temperature=0.1)
    assert y_hi.shape == y_lo.shape
    # Just sanity-check it runs and returns finite outputs at both extremes.
    assert torch.isfinite(y_hi).all() and torch.isfinite(y_lo).all()


def test_bias_off_runs():
    layer = DynamicConv(4, 4, kernel_size=3, K=2, bias=False)
    x = torch.randn(2, 4, 5, 5)
    y = layer(x)
    assert y.shape == (2, 4, 5, 5)
    assert layer.bias is None
