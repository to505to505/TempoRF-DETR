"""Cross-Resolution Relational Contrastive Distillation (CRRCD).

Works on the decoder hidden states at matching query slots from the frozen
HR teacher and the trainable LR student. Two MLPs (FRMs) build teacher-teacher
and teacher-student relations, and a sigmoid-NCE critic matches them.
Slot alignment comes from the KD-DETR specific-sampling hook (see DISTILLATION.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RelationMLP(nn.Module):
    """v = W2 * ReLU(W1 (e_i - e_j))."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        return self.net(diff)


class CRRCDLoss(nn.Module):
    """Cross-Resolution Relational Contrastive Distillation loss."""

    def __init__(
        self,
        hidden_dim: int,
        relation_dim: int,
        frm_hidden_dim: int,
        num_fg: int,
        num_bg: int,
        num_negatives: int,
        temperature: float,
    ):
        super().__init__()
        self.F_t = _RelationMLP(hidden_dim, frm_hidden_dim, relation_dim)
        self.F_ts = _RelationMLP(hidden_dim, frm_hidden_dim, relation_dim)
        self.K_fg = int(num_fg)
        self.K_bg = int(num_bg)
        self.n_neg = int(num_negatives)
        self.tau = float(temperature)

    def forward(
        self,
        teacher_hs: torch.Tensor,   # (B, Q, D)  detached
        student_hs: torch.Tensor,   # (B, Q, D)  with grad
        weights: torch.Tensor,      # (B, Q)     teacher max-fg confidence
    ) -> torch.Tensor:
        assert teacher_hs.shape == student_hs.shape, (
            f"shape mismatch: teacher_hs {tuple(teacher_hs.shape)} vs "
            f"student_hs {tuple(student_hs.shape)}"
        )
        assert weights.shape[:2] == teacher_hs.shape[:2], (
            f"weights {tuple(weights.shape)} must match (B, Q) of hs"
        )

        e_t = teacher_hs.detach()
        e_s = student_hs

        B, Q, D = e_t.shape
        K_fg = min(self.K_fg, Q)
        K_bg = min(self.K_bg, Q)
        if K_fg == 0 or K_bg < 2:
            return e_s.new_zeros(())

        # top-K_fg / bottom-K_bg slots by foreground weight
        fg_idx = weights.topk(K_fg, dim=1).indices                      # (B, K_fg)
        bg_idx = weights.topk(K_bg, dim=1, largest=False).indices       # (B, K_bg)

        def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))

        et_fg = gather(e_t, fg_idx)        # (B, K_fg, D)
        et_bg = gather(e_t, bg_idx)        # (B, K_bg, D)
        es_bg = gather(e_s, bg_idx)        # (B, K_bg, D)  - carries grad

        diff_t = et_fg.unsqueeze(2) - et_bg.unsqueeze(1)
        diff_ts = et_fg.unsqueeze(2) - es_bg.unsqueeze(1)

        v_t = self.F_t(diff_t)             # (B, K_fg, K_bg, R)
        v_ts = self.F_ts(diff_ts)          # (B, K_fg, K_bg, R)

        v_t_n = F.normalize(v_t, dim=-1)
        v_ts_n = F.normalize(v_ts, dim=-1)

        # (B, K_fg, K_bg, K_bg); last two dims are (j, k)
        sim = torch.einsum("bijd,bikd->bijk", v_t_n, v_ts_n) / max(self.tau, 1e-6)

        device = sim.device
        eye = torch.eye(K_bg, device=device, dtype=torch.bool)         # (K_bg, K_bg)
        eye = eye.view(1, 1, K_bg, K_bg)

        # positive = diagonal (j == k)
        pos_sim = torch.diagonal(sim, dim1=-2, dim2=-1)                # (B, K_fg, K_bg)
        log_h_pos = F.logsigmoid(pos_sim)

        # log(1 - sigma(s)) = logsigmoid(-s); drop the diagonal
        log_one_minus_h_neg = F.logsigmoid(-sim).masked_fill(eye, 0.0)

        # subsample negatives per anchor if n_neg is set, else use all off-diagonal
        if 0 < self.n_neg < (K_bg - 1):
            rand = torch.rand_like(sim).masked_fill(eye, -1.0)
            sel = rand.topk(self.n_neg, dim=-1).indices                # (B, K_fg, K_bg, n_neg)
            neg_term = log_one_minus_h_neg.gather(-1, sel).sum(dim=-1)  # (B, K_fg, K_bg)
        else:
            neg_term = log_one_minus_h_neg.sum(dim=-1)                  # (B, K_fg, K_bg)

        loss = -(log_h_pos.mean() + neg_term.mean())
        return loss
