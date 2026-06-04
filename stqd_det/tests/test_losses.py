"""Tests for the set-prediction criterion."""

import pytest
import torch

from stqd_det.config import Config
from stqd_det.losses import SetCriterion, _sigmoid_focal_loss
from stqd_det.noise import to_signed


def _toy_outputs(K: int, BT: int, N: int, C: int):
    return {
        "pred_logits_stages": torch.zeros(K, BT, N, C),
        "pred_boxes_stages": torch.zeros(K, BT, N, 4),
    }


def test_sigmoid_focal_loss_zero_when_correct():
    """If logits saturate the correct target, loss -> 0."""
    logits = torch.full((4, 1), 10.0)        # sigmoid ~ 1
    targets = torch.ones(4, 1)
    loss = _sigmoid_focal_loss(logits, targets, reduction="sum")
    assert float(loss) < 0.01


def test_sigmoid_focal_loss_nonzero_when_wrong():
    logits = torch.full((4, 1), 10.0)
    targets = torch.zeros(4, 1)
    loss = _sigmoid_focal_loss(logits, targets, reduction="sum")
    assert float(loss) > 1.0


def test_criterion_zero_when_no_targets_no_real_predictions():
    """If no GT boxes anywhere, total loss should be finite and small
    (only the focal classification term, mostly suppressed by 'no
    object' negative entropy)."""
    cfg = Config()
    crit = SetCriterion(num_classes=1, cfg=cfg)
    out = _toy_outputs(K=2, BT=2, N=5, C=1)
    targets = [
        [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}],
        [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}],
    ]
    loss = crit(out, targets)
    assert torch.isfinite(loss["loss_total"])
    assert float(loss["loss_l1"]) == 0.0
    assert float(loss["loss_giou"]) == 0.0


def test_criterion_drops_when_predictions_match_targets_perfectly():
    """Take a tiny scenario, set predictions == GT for one slot, and verify
    the box losses are zero and total loss decreases relative to an
    intentionally bad init."""
    cfg = Config()
    crit = SetCriterion(num_classes=1, cfg=cfg)

    K, BT, N = 1, 1, 3
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])           # one stenosis box
    target_signed = to_signed(gt, cfg.sigma_scale)

    # GOOD predictions: slot 0 == GT, slots 1-2 random (treated as background).
    pred_boxes = torch.zeros(K, BT, N, 4)
    pred_boxes[0, 0, 0] = target_signed[0]
    pred_logits = torch.full((K, BT, N, 1), -5.0)        # all suppressed
    pred_logits[0, 0, 0, 0] = 5.0                        # slot 0 highly confident
    good_out = {"pred_logits_stages": pred_logits, "pred_boxes_stages": pred_boxes}

    targets = [
        [{"boxes": gt, "labels": torch.zeros(1, dtype=torch.int64)}],
    ]
    good_loss = crit(good_out, targets)
    # Matched slot has same box & high confidence -> l1 ~ 0, giou ~ 0.
    assert float(good_loss["loss_l1"]) < 1e-3
    assert float(good_loss["loss_giou"]) < 1e-3

    # BAD predictions: same boxes but everything confident=wrong.
    bad_logits = pred_logits.clone()
    bad_logits[0, 0, 0, 0] = -5.0                        # slot 0 says 'no'
    bad_out = {"pred_logits_stages": bad_logits, "pred_boxes_stages": pred_boxes}
    bad_loss = crit(bad_out, targets)
    assert float(bad_loss["loss_total"]) > float(good_loss["loss_total"])


def test_criterion_gradient_flows():
    cfg = Config()
    crit = SetCriterion(num_classes=1, cfg=cfg)
    K, BT, N = 2, 2, 4
    logits = torch.randn(K, BT, N, 1, requires_grad=True)
    boxes = torch.randn(K, BT, N, 4, requires_grad=True)
    targets = [
        [
            {"boxes": torch.tensor([[0.3, 0.3, 0.1, 0.1]]), "labels": torch.zeros(1, dtype=torch.int64)},
            {"boxes": torch.tensor([[0.6, 0.6, 0.1, 0.1]]), "labels": torch.zeros(1, dtype=torch.int64)},
        ],
    ]
    out = crit({"pred_logits_stages": logits, "pred_boxes_stages": boxes}, targets)
    out["loss_total"].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert boxes.grad is not None and boxes.grad.abs().sum() > 0


def test_criterion_raises_on_bt_b_mismatch():
    cfg = Config()
    crit = SetCriterion(num_classes=1, cfg=cfg)
    # K, BT=3, N=4 - but only B=2 in target list with T=4 (B*T=8, mismatch with BT=3)
    out = _toy_outputs(K=1, BT=3, N=4, C=1)
    targets = [
        [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)} for _ in range(4)],
        [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)} for _ in range(4)],
    ]
    with pytest.raises(ValueError):
        crit(out, targets)
