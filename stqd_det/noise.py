"""Sequential Quantum Noise Box (SQNB) generator for STQD-Det.

Samples the marginal q(B_t | B_0) directly (DiffusionDet-style), but uses
centered Poisson noise (Poisson(1) - 1, mean 0 var 1) instead of Gaussian.
Frame 1 gets a full noise box; later frames inherit it plus a small
per-frame perturbation.

Coords: targets are cxcywh in [0,1]; diffusion runs in signed-normalised
[-sigma, sigma] space (see to_signed / from_signed) so zero-mean noise
doesn't need boundary clamping.
"""

from typing import List, Optional, Tuple

import math
import torch


# schedule 


def cosine_alpha_bar(T_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule. Returns a_bar_t for t = 0..T_steps-1."""
    steps = torch.arange(T_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / T_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    alpha_bar = alpha_bar[:-1].clamp(min=1e-6, max=1.0)
    return alpha_bar.to(torch.float32)


# box-space conversions 


def to_signed(boxes_norm: torch.Tensor, sigma: float) -> torch.Tensor:
    """[0, 1]^4 cxcywh -> [-sigma, sigma]^4 signed-normalised."""
    return (boxes_norm - 0.5) * 2.0 * sigma


def from_signed(boxes_signed: torch.Tensor, sigma: float) -> torch.Tensor:
    """[-sigma, sigma]^4 signed -> [0, 1]^4 cxcywh (clamped)."""
    return (boxes_signed / (2.0 * sigma) + 0.5).clamp(0.0, 1.0)


# noise 


def sample_centered_poisson(
    shape: Tuple[int, ...],
    device: torch.device,
    rate: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """eps = Poisson(rate) - rate.  Mean 0, variance ~ rate."""
    rate_t = torch.full(shape, float(rate), device=device, dtype=dtype)
    eps = torch.poisson(rate_t) - rate_t
    return eps


def forward_diffuse(
    B0_signed: torch.Tensor,
    t: torch.Tensor,
    alpha_bar: torch.Tensor,
    poisson_rate: float = 1.0,
) -> torch.Tensor:
    """Sample B_t ~ q(B_t | B_0). Returns signed-normalised boxes (not clamped).

    B0_signed is (..., 4); t is a long tensor whose shape must be a prefix of
    B0's leading dims (one timestep per box, or per sample/frame).
    """
    lead = B0_signed.shape[:-1]
    if t.shape != lead and (len(t.shape) > len(lead) or t.shape != lead[: len(t.shape)]):
        raise ValueError(
            f"t shape {tuple(t.shape)} must be a prefix of B0 leading dims "
            f"{tuple(lead)}"
        )
    a = alpha_bar[t.clamp(min=0, max=alpha_bar.numel() - 1)]
    for _ in range(len(lead) - len(t.shape)):
        a = a.unsqueeze(-1)
    a = a.unsqueeze(-1)
    eps = sample_centered_poisson(
        B0_signed.shape, device=B0_signed.device, rate=poisson_rate
    )
    return a.sqrt() * B0_signed + (1.0 - a).sqrt() * eps


# proposal padding 


def pad_to_num_proposals(
    gt_boxes_norm: torch.Tensor,
    num_proposals: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad GT boxes (cxcywh in [0,1]) up to num_proposals.

    Pads with random boxes: cx/cy uniform in [0,1], w/h in [0.2, 0.8].
    Returns (num_proposals, 4) boxes and an is_real bool mask (real GTs at
    the front, random pads after).
    """
    if gt_boxes_norm.dim() != 2 or gt_boxes_norm.shape[-1] != 4:
        raise ValueError("gt_boxes_norm must be (N, 4) cxcywh in [0,1]")
    N = gt_boxes_norm.shape[0]
    device = gt_boxes_norm.device
    is_real = torch.zeros(num_proposals, dtype=torch.bool, device=device)
    if N >= num_proposals:
        out = gt_boxes_norm[:num_proposals].clone()
        is_real[:] = True
        return out, is_real

    pad_n = num_proposals - N
    rand = torch.rand((pad_n, 4), device=device, generator=generator)
    rand[:, 2:] = rand[:, 2:] * 0.6 + 0.2  # w/h into [0.2, 0.8]
    out = torch.cat([gt_boxes_norm, rand], dim=0)
    is_real[:N] = True
    return out, is_real


# sequential prior 


def sequential_prior_perturb(
    B_t_frame1_signed: torch.Tensor,
    num_frames: int,
    alpha: float,
    poisson_rate: float = 1.0,
) -> torch.Tensor:
    """Broadcast frame-1 noise boxes across T frames, perturbing frames 1..T-1.

    B_t_frame1_signed is (B, num_proposals, 4). Returns (B, T, num_proposals, 4):
    frame 0 is unchanged; frames 1..T-1 get an independent centered-Poisson
    perturbation of scale alpha.
    """
    if B_t_frame1_signed.dim() != 3:
        raise ValueError(
            "B_t_frame1_signed must be (B, num_proposals, 4); got shape "
            f"{tuple(B_t_frame1_signed.shape)}"
        )
    B, N, _ = B_t_frame1_signed.shape
    out = B_t_frame1_signed.unsqueeze(1).expand(B, num_frames, N, 4).clone()
    if num_frames <= 1:
        return out
    perturb = sample_centered_poisson(
        (B, num_frames - 1, N, 4),
        device=B_t_frame1_signed.device,
        rate=poisson_rate,
    )
    out[:, 1:, :, :] = out[:, 1:, :, :] + alpha * perturb
    return out


# training / inference init 


def prepare_training_noise(
    targets_per_frame: List[List[dict]],
    num_proposals: int,
    sigma: float,
    alpha_bar: torch.Tensor,
    sequential_alpha: float,
    device: torch.device,
    t: Optional[torch.Tensor] = None,
    poisson_rate: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build noised proposals for a batch of T-frame windows.

    targets_per_frame is list[B] of list[T] of {"boxes": (N_n, 4) cxcywh}.
    t is an optional (B,) timestep tensor; sampled uniformly if None.
    Returns noised boxes (B, T, num_proposals, 4) and the (B,) timesteps used.
    """
    B = len(targets_per_frame)
    if B == 0:
        raise ValueError("Empty target batch")
    T = len(targets_per_frame[0])
    T_steps = alpha_bar.numel()

    if t is None:
        t = torch.randint(0, T_steps, (B,), device=device, dtype=torch.long)
    elif t.shape != (B,):
        raise ValueError(f"t must have shape ({B},), got {tuple(t.shape)}")

    # Seed frame 1 (centre) of each clip with its GT; the sequential prior
    # then propagates that noise to the remaining frames.
    seeded: List[torch.Tensor] = []
    for b in range(B):
        gt = targets_per_frame[b][0]["boxes"].to(device)
        padded, _ = pad_to_num_proposals(gt, num_proposals)
        seeded.append(padded)
    B0_norm = torch.stack(seeded, dim=0)                       # (B, N, 4)

    B0_signed = to_signed(B0_norm, sigma)
    Bt_signed_frame1 = forward_diffuse(
        B0_signed, t, alpha_bar, poisson_rate=poisson_rate
    )
    Bt_signed_all = sequential_prior_perturb(
        Bt_signed_frame1,
        num_frames=T,
        alpha=sequential_alpha,
        poisson_rate=poisson_rate,
    )                                                          # (B, T, N, 4)
    return Bt_signed_all, t


def prepare_inference_init(
    batch_size: int,
    num_frames: int,
    num_proposals: int,
    sigma: float,
    sequential_alpha: float,
    device: torch.device,
    poisson_rate: float = 1.0,
) -> torch.Tensor:
    """Seed inference with random proposals.

    Raw Poisson noise around signed-0 gives ~30% boxes with w_signed <= -sigma,
    which from_signed clamps to zero-area regions the model can't refine. So we
    sample cx/cy uniform in [0,1] and w/h in [0.05, 0.40] (typical lesion size
    ~0.07), then map to signed space.
    """
    cxcy = torch.rand((batch_size, num_proposals, 2), device=device)
    wh = torch.rand((batch_size, num_proposals, 2), device=device) * 0.35 + 0.05
    init_norm = torch.cat([cxcy, wh], dim=-1)
    init_signed = (init_norm - 0.5) * 2.0 * sigma  # to_signed inline
    return sequential_prior_perturb(
        init_signed,
        num_frames=num_frames,
        alpha=sequential_alpha,
        poisson_rate=poisson_rate,
    )
