"""Count-consistency loss across a window of frames.

Penalises per-frame box counts that drift from the window median. Uses a
soft (sigmoid) count so it stays differentiable.
"""

from __future__ import annotations

import torch


def num_consistency_loss(
    pred_logits: torch.Tensor,
    threshold: float,
    soft_temp: float = 0.05,
) -> torch.Tensor:
    """pred_logits is (B, T, Q, K). Returns a scalar. Smaller soft_temp ->
    closer to a hard count but weaker gradient.
    """
    if pred_logits.dim() != 4:
        raise ValueError(
            f"pred_logits must be (B, T, Q, K), got {tuple(pred_logits.shape)}"
        )
    B, T, Q, _K = pred_logits.shape
    if T < 2:
        return pred_logits.new_zeros(())

    p = pred_logits.sigmoid().amax(dim=-1)            # (B, T, Q)

    soft_temp = max(float(soft_temp), 1e-6)
    soft_indicator = torch.sigmoid((p - float(threshold)) / soft_temp)  # (B, T, Q)
    n_t = soft_indicator.sum(dim=-1)                  # (B, T)

    # detached median = fixed target; grad only through n_t
    n_r = n_t.detach().median(dim=1, keepdim=True).values  # (B, 1)

    return (n_t - n_r).abs().mean()
