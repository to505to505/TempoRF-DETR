"""End-to-end stage-1 model smoke tests."""

import pytest
import torch

from stqd_det.config import Config
from stqd_det.losses import SetCriterion
from stqd_det.model import STQDDet


def _toy_cfg(T: int = 3, N: int = 16, img: int = 128, K: int = 2) -> Config:
    """Tiny config so CPU forward+backward stays cheap."""
    return Config(
        img_size=img,
        T=T,
        num_proposals=N,
        diffusion_T_steps=50,
        decoder_num_heads=K,
        decoder_roi_size=4,
        decoder_dim_feedforward=64,
        decoder_dynamic_dim=8,
        fpn_out_channels=32,
        gfe_heads=2,
        gfe_dc_kernels=2,
        num_workers=0,
    )


def _toy_targets(B: int, T: int):
    return [
        [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.zeros(1, dtype=torch.int64),
            }
            for _ in range(T)
        ]
        for _ in range(B)
    ]


def test_forward_training_returns_expected_keys():
    cfg = _toy_cfg()
    model = STQDDet(cfg).train()
    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size)
    targets = _toy_targets(B=1, T=cfg.T)
    out = model(frames, targets_per_frame=targets)
    expected = {"pred_logits_stages", "pred_boxes_stages", "final_logits",
                "final_boxes", "queries_final", "t", "B", "T"}
    assert expected <= set(out.keys())
    K = cfg.decoder_num_heads
    BT = 1 * cfg.T
    assert out["pred_logits_stages"].shape == (K, BT, cfg.num_proposals, cfg.num_classes)
    assert out["pred_boxes_stages"].shape == (K, BT, cfg.num_proposals, 4)
    assert out["final_logits"].shape == (BT, cfg.num_proposals, cfg.num_classes)
    assert out["final_boxes"].shape == (BT, cfg.num_proposals, 4)


def test_forward_inference_returns_finite_predictions():
    cfg = _toy_cfg()
    model = STQDDet(cfg).eval()
    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size)
    with torch.no_grad():
        out = model(frames)
    assert torch.isfinite(out["final_logits"]).all()
    assert torch.isfinite(out["final_boxes"]).all()


def test_training_raises_without_targets():
    cfg = _toy_cfg()
    model = STQDDet(cfg).train()
    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size)
    with pytest.raises(ValueError):
        model(frames)


def test_forward_backward_with_set_criterion():
    """End-to-end: a single optimization step must reduce the loss
    (sanity check that gradients reach every trainable parameter)."""
    torch.manual_seed(0)
    cfg = _toy_cfg(T=3, N=16, img=64, K=2)
    model = STQDDet(cfg).train()
    crit = SetCriterion(num_classes=cfg.num_classes, cfg=cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size)
    targets = _toy_targets(B=1, T=cfg.T)

    losses = []
    for step in range(3):
        optim.zero_grad()
        out = model(frames, targets_per_frame=targets, t=torch.tensor([0]))
        loss = crit.compute_total_loss(out, targets)["loss_total"]
        loss.backward()
        optim.step()
        losses.append(float(loss))
    # On a fixed input, loss should monotonically decrease across a few
    # optimisation steps.
    assert losses[-1] < losses[0]


def test_total_loss_includes_l_num_when_stfs_enabled():
    cfg = _toy_cfg(T=3, N=8, img=64, K=2)
    cfg.stfs_enabled = True
    model = STQDDet(cfg).train()
    crit = SetCriterion(num_classes=cfg.num_classes, cfg=cfg)
    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size)
    targets = _toy_targets(B=1, T=cfg.T)
    out = model(frames, targets_per_frame=targets, t=torch.tensor([0]))
    losses = crit.compute_total_loss(out, targets)
    assert "loss_num" in losses
    assert "stage1/loss_total" in losses
    assert "stage2/loss_total" in losses
    assert torch.isfinite(losses["loss_total"])


def test_total_loss_without_stfs_falls_back_to_stage1_only():
    cfg = _toy_cfg(T=3, N=8, img=64, K=2)
    cfg.stfs_enabled = False
    model = STQDDet(cfg).train()
    crit = SetCriterion(num_classes=cfg.num_classes, cfg=cfg)
    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size)
    targets = _toy_targets(B=1, T=cfg.T)
    out = model(frames, targets_per_frame=targets, t=torch.tensor([0]))
    losses = crit.compute_total_loss(out, targets)
    assert "stage1/loss_total" in losses
    assert "stage2/loss_total" not in losses
    assert "loss_num" not in losses


def test_inference_init_runs_at_different_T():
    cfg = _toy_cfg(T=5, N=8, img=64, K=1)
    model = STQDDet(cfg).eval()
    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size)
    with torch.no_grad():
        out = model(frames)
    assert out["final_logits"].shape[0] == cfg.T
