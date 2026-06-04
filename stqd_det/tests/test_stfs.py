"""Tests for the Spatio-Temporal Feature Sharing module."""

from collections import OrderedDict

import torch

from stqd_det.config import Config
from stqd_det.stfs import (
    RoIAggregator,
    STFS,
    build_chains,
    vote_chains,
    _cxcywh_to_xyxy,
    _pairwise_iou,
    _xyxy_to_cxcywh,
)


def _toy_cfg() -> Config:
    return Config(
        img_size=64,
        T=5,
        num_proposals=4,
        fpn_out_channels=16,
        decoder_roi_size=4,
        gfe_heads=2,
        gfe_dc_kernels=2,
    )


def _toy_features(BT: int, cfg: Config):
    img = cfg.img_size
    return OrderedDict(
        [
            ("P2", torch.randn(BT, cfg.fpn_out_channels, img // 4, img // 4)),
            ("P3", torch.randn(BT, cfg.fpn_out_channels, img // 8, img // 8)),
            ("P4", torch.randn(BT, cfg.fpn_out_channels, img // 16, img // 16)),
            ("P5", torch.randn(BT, cfg.fpn_out_channels, img // 32, img // 32)),
        ]
    )


def test_pairwise_iou_self_returns_identity():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]])
    iou = _pairwise_iou(boxes, boxes)
    assert torch.allclose(torch.diag(iou), torch.ones(2), atol=1e-5)


def test_pairwise_iou_zero_when_disjoint():
    a = torch.tensor([[0.1, 0.1, 0.05, 0.05]])
    b = torch.tensor([[0.9, 0.9, 0.05, 0.05]])
    iou = _pairwise_iou(a, b)
    assert float(iou[0, 0]) == 0.0


def test_xyxy_roundtrip():
    cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    back = _xyxy_to_cxcywh(_cxcywh_to_xyxy(cxcywh))
    assert torch.allclose(back, cxcywh, atol=1e-6)


def test_build_chains_single_box_in_each_frame_makes_one_chain():
    """One box per frame, all overlapping -> all in one chain."""
    boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]]) for _ in range(5)]
    scores = [torch.tensor([0.9]) for _ in range(5)]
    chains = build_chains(boxes, scores)
    assert len(chains) == 1
    assert len(chains[0]) == 5
    assert chains[0] == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]


def test_build_chains_disjoint_boxes_dont_merge():
    """Two non-overlapping boxes per frame -> 2 chains."""
    boxes = [
        torch.tensor([[0.1, 0.1, 0.05, 0.05], [0.9, 0.9, 0.05, 0.05]])
        for _ in range(3)
    ]
    scores = [torch.tensor([0.9, 0.9]) for _ in range(3)]
    chains = build_chains(boxes, scores)
    assert len(chains) == 2
    for c in chains:
        assert len(c) == 3


def test_vote_h_tp_when_chain_spans_all_frames():
    chain_full = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    groups = vote_chains([chain_full], T=5)
    assert groups["H_TP"] == [0]
    assert groups["H_FN"] == []
    assert groups["H_FP"] == []


def test_vote_h_fn_when_chain_misses_some_frames():
    # 3 of 5 frames -> 3 >= T/2=2, so H-FN
    chain_partial = [(0, 0), (2, 0), (4, 0)]
    groups = vote_chains([chain_partial], T=5)
    assert groups["H_FN"] == [0]


def test_vote_h_fp_when_chain_spans_few_frames():
    chain_short = [(0, 0)]
    groups = vote_chains([chain_short], T=5)
    assert groups["H_FP"] == [0]


def test_roi_aggregator_shape_and_grad():
    cfg = _toy_cfg()
    agg = RoIAggregator(cfg.fpn_out_channels, cfg.decoder_roi_size, cfg.gfe_heads, cfg.gfe_dc_kernels)
    q = torch.randn(1, cfg.fpn_out_channels, cfg.decoder_roi_size, cfg.decoder_roi_size, requires_grad=True)
    r = torch.randn(3, cfg.fpn_out_channels, cfg.decoder_roi_size, cfg.decoder_roi_size, requires_grad=True)
    out = agg(q, r)
    assert out.shape == q.shape
    out.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert r.grad is not None and r.grad.abs().sum() > 0


def test_stfs_forward_runs_and_outputs_correct_shapes():
    cfg = _toy_cfg()
    stfs = STFS(cfg, fpn_levels=["P2", "P3", "P4", "P5"], fpn_strides={"P2": 4, "P3": 8, "P4": 16, "P5": 32})
    B, T, N = 1, cfg.T, cfg.num_proposals
    stage1_logits = torch.randn(B * T, N, cfg.num_classes)
    stage1_boxes_signed = torch.randn(B * T, N, 4) * 0.2          # near origin
    features = _toy_features(B * T, cfg)
    out = stfs(stage1_logits, stage1_boxes_signed, features, B, T)
    assert out["stage2_boxes_signed"].shape == stage1_boxes_signed.shape
    assert out["counts_per_frame"].shape == (B, T)
    assert out["n_r"].shape == (B,)
    assert torch.isfinite(out["loss_num"])


def test_stfs_consistency_loss_is_zero_when_counts_match_n_r():
    cfg = _toy_cfg()
    stfs = STFS(cfg, fpn_levels=["P2", "P3", "P4", "P5"], fpn_strides={"P2": 4, "P3": 8, "P4": 16, "P5": 32})
    counts = torch.tensor([[3, 3, 3, 3, 3]])
    n_r = torch.tensor([3])
    loss = stfs.consistency_loss(counts, n_r, beta=cfg.consistency_beta)
    assert float(loss) == 0.0


def test_stfs_consistency_loss_nonzero_when_mismatched():
    cfg = _toy_cfg()
    stfs = STFS(cfg, fpn_levels=["P2", "P3", "P4", "P5"], fpn_strides={"P2": 4, "P3": 8, "P4": 16, "P5": 32})
    counts = torch.tensor([[3, 0, 3, 3, 3]])
    n_r = torch.tensor([3])
    loss = stfs.consistency_loss(counts, n_r, beta=cfg.consistency_beta)
    assert float(loss) > 0
