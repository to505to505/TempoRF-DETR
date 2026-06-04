"""Unit tests for the SQNB noise box generator."""

import math

import pytest
import torch

from stqd_det.noise import (
    cosine_alpha_bar,
    forward_diffuse,
    from_signed,
    pad_to_num_proposals,
    prepare_inference_init,
    prepare_training_noise,
    sample_centered_poisson,
    sequential_prior_perturb,
    to_signed,
)


def test_cosine_schedule_is_monotone_decreasing_in_unit_interval():
    a = cosine_alpha_bar(1000)
    assert a.numel() == 1000
    assert float(a[0]) > 0.99
    assert float(a[-1]) < 0.05
    diffs = a[1:] - a[:-1]
    assert (diffs <= 0).all()


def test_signed_roundtrip_is_identity_on_unit_interval():
    sigma = 2.0
    x = torch.rand(7, 4)
    y = from_signed(to_signed(x, sigma), sigma)
    assert torch.allclose(x, y, atol=1e-6)


def test_centered_poisson_has_zero_mean_unit_variance_at_rate_1():
    torch.manual_seed(0)
    eps = sample_centered_poisson((50000,), torch.device("cpu"), rate=1.0)
    assert abs(float(eps.mean())) < 0.02
    assert abs(float(eps.var(unbiased=False)) - 1.0) < 0.05


def test_forward_diffuse_t0_returns_close_to_b0_only_when_alpha_close_to_one():
    """At t=0 with cosine schedule a_bar~1, B_t should track B_0 closely.

    We don't assert exact equality because centered Poisson with rate 1
    contributes a small residual through sqrt(1-a_bar) ~ sqrt(1e-6). That's tiny.
    """
    torch.manual_seed(0)
    a = cosine_alpha_bar(1000)
    B0 = torch.randn(4, 4)
    t = torch.zeros(4, dtype=torch.long)
    Bt = forward_diffuse(B0, t, a)
    assert torch.allclose(Bt, B0, atol=5e-2)


def test_forward_diffuse_tT_is_dominated_by_noise():
    torch.manual_seed(0)
    a = cosine_alpha_bar(1000)
    B0 = torch.zeros(4, 4)
    t = torch.full((4,), 999, dtype=torch.long)
    Bt = forward_diffuse(B0, t, a)
    # B0 is zero so Bt = sqrt(1-a_bar) * eps; expect non-zero variance.
    assert float(Bt.abs().mean()) > 0.1
    assert torch.isfinite(Bt).all()


def test_pad_to_num_proposals_keeps_real_gts_at_front():
    gt = torch.tensor([
        [0.5, 0.5, 0.1, 0.1],
        [0.2, 0.3, 0.2, 0.2],
    ])
    padded, is_real = pad_to_num_proposals(gt, num_proposals=10)
    assert padded.shape == (10, 4)
    assert is_real[:2].all() and not is_real[2:].any()
    assert torch.allclose(padded[:2], gt)


def test_pad_to_num_proposals_truncates_when_too_many_gts():
    gt = torch.rand(20, 4)
    padded, is_real = pad_to_num_proposals(gt, num_proposals=10)
    assert padded.shape == (10, 4)
    assert is_real.all()


def test_pad_pad_boxes_are_inside_image():
    torch.manual_seed(0)
    gt = torch.empty(0, 4)
    padded, _ = pad_to_num_proposals(gt, num_proposals=200)
    assert (padded[:, 0:2] >= 0).all() and (padded[:, 0:2] <= 1).all()
    assert (padded[:, 2:4] >= 0.2).all() and (padded[:, 2:4] <= 0.8).all()


def test_sequential_prior_preserves_frame1_and_perturbs_others():
    torch.manual_seed(0)
    B1 = torch.randn(2, 8, 4)
    out = sequential_prior_perturb(B1, num_frames=5, alpha=0.01, poisson_rate=1.0)
    assert out.shape == (2, 5, 8, 4)
    assert torch.allclose(out[:, 0], B1)
    # frames 1..4 must differ from frame 0 (Poisson perturbation is non-zero a.s.)
    assert (out[:, 1:] - out[:, 0:1]).abs().sum() > 0
    # ...but only a little (perturbation is alpha-scaled)
    assert (out[:, 1:] - out[:, 0:1]).abs().mean() < 0.1


def test_sequential_prior_num_frames_one_returns_single_frame():
    B1 = torch.randn(1, 4, 4)
    out = sequential_prior_perturb(B1, num_frames=1, alpha=0.01)
    assert out.shape == (1, 1, 4, 4)
    assert torch.allclose(out[:, 0], B1)


def test_prepare_training_noise_shapes():
    torch.manual_seed(0)
    T = 5
    targets = [
        [
            {"boxes": torch.rand(2, 4)},
            {"boxes": torch.rand(1, 4)},
            {"boxes": torch.rand(0, 4)},
            {"boxes": torch.rand(3, 4)},
            {"boxes": torch.rand(1, 4)},
        ]
    ]
    alpha = cosine_alpha_bar(100)
    boxes, t = prepare_training_noise(
        targets,
        num_proposals=300,
        sigma=2.0,
        alpha_bar=alpha,
        sequential_alpha=0.01,
        device=torch.device("cpu"),
    )
    assert boxes.shape == (1, T, 300, 4)
    assert t.shape == (1,)
    assert 0 <= int(t[0]) < 100
    assert torch.isfinite(boxes).all()


def test_prepare_training_noise_respects_explicit_t():
    torch.manual_seed(0)
    targets = [[{"boxes": torch.rand(1, 4)} for _ in range(3)]]
    alpha = cosine_alpha_bar(100)
    t_in = torch.tensor([0], dtype=torch.long)
    boxes, t_out = prepare_training_noise(
        targets,
        num_proposals=10,
        sigma=2.0,
        alpha_bar=alpha,
        sequential_alpha=0.01,
        device=torch.device("cpu"),
        t=t_in,
    )
    assert torch.equal(t_in, t_out)
    # At t=0 the noised boxes should be close to the signed-normalised GT
    # for frame 0. Take the first GT slot and compare.
    gt_signed_first = to_signed(targets[0][0]["boxes"], 2.0)
    assert torch.allclose(boxes[0, 0, :1], gt_signed_first, atol=0.05)


def test_prepare_inference_init_shape():
    boxes = prepare_inference_init(
        batch_size=2,
        num_frames=5,
        num_proposals=300,
        sigma=2.0,
        sequential_alpha=0.01,
        device=torch.device("cpu"),
    )
    assert boxes.shape == (2, 5, 300, 4)
    assert torch.isfinite(boxes).all()
