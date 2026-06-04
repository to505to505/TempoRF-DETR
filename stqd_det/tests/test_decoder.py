"""Tests for the DiffusionDet-style stenosis decoder."""

from collections import OrderedDict

import torch

from stqd_det.decoder import (
    DynamicHead,
    RCNNHead,
    SinusoidalTimeEmbedding,
    StenosisDecoder,
    _MIN_BOX_SIDE,
    _signed_to_xyxy_abs,
    apply_box_delta,
)
from stqd_det.noise import from_signed, to_signed


def _toy_features(BT: int, C: int, img_size: int):
    """A 4-level FPN-style dict; trailing keys must match StenosisDecoder."""
    return OrderedDict(
        [
            ("P2", torch.randn(BT, C, img_size // 4, img_size // 4)),
            ("P3", torch.randn(BT, C, img_size // 8, img_size // 8)),
            ("P4", torch.randn(BT, C, img_size // 16, img_size // 16)),
            ("P5", torch.randn(BT, C, img_size // 32, img_size // 32)),
        ]
    )


def test_signed_to_xyxy_conversion_range():
    sigma = 2.0
    boxes = torch.tensor([[[0.0, 0.0, sigma, sigma]]])           # -> cx,cy,w,h = 0.5,0.5,1,1
    xyxy = _signed_to_xyxy_abs(boxes, img_size=100, sigma=sigma)
    assert torch.allclose(
        xyxy[0, 0], torch.tensor([0.0, 0.0, 100.0, 100.0]), atol=1e-3
    )


def test_signed_to_xyxy_floors_degenerate_wh():
    """A box with w_signed << -sigma used to collapse to zero area. The safety
    clamp must enforce at least _MIN_BOX_SIDE * img_size pixels of width.
    """
    sigma = 2.0
    img = 100
    # signed (cx=0, cy=0, w=-3, h=-3) -> unclipped w/h would be -0.25 in [0,1].
    boxes = torch.tensor([[[0.0, 0.0, -3.0, -3.0]]])
    xyxy = _signed_to_xyxy_abs(boxes, img_size=img, sigma=sigma)[0, 0]
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    assert float(w) >= _MIN_BOX_SIDE * img - 1e-3
    assert float(h) >= _MIN_BOX_SIDE * img - 1e-3


def test_apply_box_delta_zero_delta_is_identity():
    sigma = 2.0
    prev = torch.tensor([[0.5, 0.5, 0.2, 0.2]])                 # norm cxcywh
    prev_signed = to_signed(prev, sigma)
    delta = torch.zeros_like(prev_signed)
    new_signed = apply_box_delta(prev_signed, delta, sigma)
    new_norm = from_signed(new_signed, sigma)
    assert torch.allclose(new_norm, prev, atol=1e-5)


def test_apply_box_delta_w_stays_positive_under_negative_delta():
    """The previous additive update sent w_signed -> -inf when many small
    negative deltas accumulated. With exp(dw) we can never reach 0."""
    sigma = 2.0
    prev_signed = to_signed(torch.tensor([[0.5, 0.5, 0.2, 0.2]]), sigma)
    # Apply 12 cascade stages of strongly-negative dw.
    for _ in range(12):
        delta = torch.tensor([[0.0, 0.0, -1.0, -1.0]])
        prev_signed = apply_box_delta(prev_signed, delta, sigma)
    new_norm = from_signed(prev_signed, sigma)
    # exp(-1)^12 = exp(-12) ~ 6e-6 -> floored to _MIN_BOX_SIDE.
    assert float(new_norm[0, 2]) >= _MIN_BOX_SIDE - 1e-6
    assert float(new_norm[0, 3]) >= _MIN_BOX_SIDE - 1e-6


def test_apply_box_delta_dw_clipped_to_avoid_overflow():
    sigma = 2.0
    prev_signed = to_signed(torch.tensor([[0.5, 0.5, 0.2, 0.2]]), sigma)
    # Huge positive dw would overflow exp(.) without clamping.
    delta = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
    new_signed = apply_box_delta(prev_signed, delta, sigma)
    assert torch.isfinite(new_signed).all()


def test_apply_box_delta_dx_scales_with_w():
    """The cx shift is `dx * w`, so a bigger box can absorb a bigger dx
    without leaving the image."""
    sigma = 2.0
    small = to_signed(torch.tensor([[0.5, 0.5, 0.05, 0.05]]), sigma)
    big   = to_signed(torch.tensor([[0.5, 0.5, 0.50, 0.50]]), sigma)
    dx = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    new_small = from_signed(apply_box_delta(small, dx, sigma), sigma)
    new_big   = from_signed(apply_box_delta(big,   dx, sigma), sigma)
    # small box moves by 0.05, big by 0.5 (clamped to 1.0)
    assert abs(float(new_small[0, 0]) - 0.55) < 1e-4
    assert abs(float(new_big[0, 0]) - 1.0) < 1e-4


def test_time_embedding_shape():
    emb = SinusoidalTimeEmbedding(d_model=64)
    t = torch.tensor([0, 100, 999], dtype=torch.long)
    out = emb(t)
    assert out.shape == (3, 64)
    # Different timesteps -> different embeddings.
    assert not torch.allclose(out[0], out[1])


def test_dynamic_head_shape():
    dh = DynamicHead(d_model=16, dim_dynamic=8, num_dynamic=2)
    roi = torch.randn(5, 49, 16)        # (B*N, S*S, C)
    pro = torch.randn(5, 16)
    out = dh(roi, pro)
    assert out.shape == (5, 16)


def test_rcnn_head_shapes_one_stage():
    head = RCNNHead(
        d_model=16, num_classes=1, nhead=4,
        dim_feedforward=32, dropout=0.0,
        roi_size=7, dim_dynamic=8, num_dynamic=2,
    )
    BT, N, C, S = 2, 4, 16, 7
    roi = torch.randn(BT * N, C, S, S)
    pro = torch.randn(BT * N, C)
    time_emb = torch.randn(BT, C)
    logits, delta, queries = head(roi, pro, num_proposals=N, time_emb=time_emb)
    assert logits.shape == (BT, N, 1)
    assert delta.shape == (BT, N, 4)
    assert queries.shape == (BT, N, C)


def test_decoder_forward_smoke():
    torch.manual_seed(0)
    BT, N, C, img = 2, 8, 16, 64
    decoder = StenosisDecoder(
        d_model=C,
        num_classes=1,
        num_heads_cascade=2,
        roi_size=7,
        nhead=4,
        dim_feedforward=32,
        dim_dynamic=8,
        img_size=img,
        sigma=2.0,
    )
    feats = _toy_features(BT, C, img)
    init_boxes = torch.zeros(BT, N, 4)                  # all proposals at image centre
    t = torch.tensor([10, 500], dtype=torch.long)
    out = decoder(feats, init_boxes, t)
    K = decoder.num_heads_cascade
    assert out["pred_logits_stages"].shape == (K, BT, N, 1)
    assert out["pred_boxes_stages"].shape == (K, BT, N, 4)
    assert out["final_logits"].shape == (BT, N, 1)
    assert out["final_boxes"].shape == (BT, N, 4)
    assert out["queries_final"].shape == (BT, N, C)


def test_decoder_gradients_flow():
    torch.manual_seed(0)
    BT, N, C, img = 1, 4, 16, 64
    decoder = StenosisDecoder(
        d_model=C, num_classes=1, num_heads_cascade=2, roi_size=4,
        nhead=4, dim_feedforward=32, dim_dynamic=8, img_size=img, sigma=2.0,
    )
    feats = _toy_features(BT, C, img)
    init_boxes = torch.zeros(BT, N, 4, requires_grad=False)
    t = torch.tensor([100], dtype=torch.long)
    out = decoder(feats, init_boxes, t)
    loss = out["final_logits"].sum() + out["final_boxes"].sum()
    loss.backward()
    # First cascade head's box-head weight should receive non-zero grad.
    head0 = decoder.heads[0]
    assert head0.box_head.weight.grad is not None
    assert head0.box_head.weight.grad.abs().sum() > 0
