"""HR->LR distillation losses (KL + L1 + GIoU) weighted by a per-query
foreground mask. Student and teacher share object queries, so predictions
are aligned slot-for-slot.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# Box helpers (element-wise; rfdetr.box_ops is pairwise NxM, not what we want here)

def _cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(dim=-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


def _elementwise_giou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Element-wise GIoU between two (..., 4) xyxy box tensors."""
    x1 = torch.maximum(b1[..., 0], b2[..., 0])
    y1 = torch.maximum(b1[..., 1], b2[..., 1])
    x2 = torch.minimum(b1[..., 2], b2[..., 2])
    y2 = torch.minimum(b1[..., 3], b2[..., 3])
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    a1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    a2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)
    union = a1 + a2 - inter

    ex1 = torch.minimum(b1[..., 0], b2[..., 0])
    ey1 = torch.minimum(b1[..., 1], b2[..., 1])
    ex2 = torch.maximum(b1[..., 2], b2[..., 2])
    ey2 = torch.maximum(b1[..., 3], b2[..., 3])
    enc = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0)

    eps = 1e-7
    iou = inter / union.clamp(min=eps)
    giou = iou - (enc - union) / enc.clamp(min=eps)
    return giou


# Class KL  (Bernoulli)

def _bernoulli_kl_per_query(
    student_logits: torch.Tensor,   # (B, Q, K_s)
    teacher_logits: torch.Tensor,   # (B, Q, K_t)
    temperature: float,
) -> torch.Tensor:
    """Per-query mean Bernoulli KL(p_t || p_s).

    When K_s != K_t both sides collapse to a single foreground prob via amax,
    so the target still makes sense if teacher/student class counts differ.
    """
    T = max(temperature, 1e-6)
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        p_t = torch.sigmoid(teacher_logits / T).amax(dim=-1, keepdim=True)  # (B,Q,1)
        p_s = torch.sigmoid(student_logits / T).amax(dim=-1, keepdim=True)  # (B,Q,1)
    else:
        p_t = torch.sigmoid(teacher_logits / T)
        p_s = torch.sigmoid(student_logits / T)

    eps = 1e-6
    p_t = p_t.clamp(min=eps, max=1.0 - eps).detach()
    p_s = p_s.clamp(min=eps, max=1.0 - eps)

    kl = p_t * (p_t.log() - p_s.log()) + (1 - p_t) * ((1 - p_t).log() - (1 - p_s).log())
    # T**2 factor: Hinton convention, keeps gradients comparable across temps.
    return kl.mean(dim=-1) * (T * T)


# Public API

def distillation_loss(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    cfg,
) -> Dict[str, torch.Tensor]:
    """Weighted KL + L1 + GIoU centre-frame distillation losses.

    student_out/teacher_out hold pred_logits (B,Q,K), pred_boxes (B,Q,4) in
    normalised cxcywh; teacher_out also carries foreground_weight (B,Q).
    Returns the three component losses plus their weighted sum loss_distill.
    """
    s_logits = student_out["pred_logits"]
    s_boxes = student_out["pred_boxes"]
    t_logits = teacher_out["pred_logits"]
    t_boxes = teacher_out["pred_boxes"]
    w = teacher_out["foreground_weight"]                  # (B, Q)

    # Drop the no-object slot rfdetr appends (+1), so KL sees real classes only.
    K_student = int(getattr(cfg, "num_classes", s_logits.shape[-1]))
    if s_logits.shape[-1] > K_student:
        s_logits = s_logits[..., :K_student]

    assert s_logits.shape[:2] == t_logits.shape[:2], (
        f"shared queries required: student logits {tuple(s_logits.shape)} "
        f"vs teacher {tuple(t_logits.shape)}"
    )

    eps = 1e-6
    w_sum = w.sum().clamp(min=eps)

    # KL
    kl_per_q = _bernoulli_kl_per_query(
        s_logits, t_logits, cfg.distill_temperature
    )                                                     # (B, Q)
    loss_kl = (w * kl_per_q).sum() / w_sum

    # L1 on normalised cxcywh
    l1_per_q = (s_boxes - t_boxes).abs().sum(dim=-1)      # (B, Q)
    loss_l1 = (w * l1_per_q).sum() / w_sum

    # GIoU on xyxy
    giou_per_q = _elementwise_giou(
        _cxcywh_to_xyxy(s_boxes), _cxcywh_to_xyxy(t_boxes)
    )                                                     # (B, Q)
    loss_giou = (w * (1.0 - giou_per_q)).sum() / w_sum

    # Aux-layer distillation: only the student varies, teacher target is its last layer.
    if cfg.distill_use_aux_layers and "aux_outputs" in student_out:
        n_aux = 0
        aux_kl = s_boxes.new_zeros(())
        aux_l1 = s_boxes.new_zeros(())
        aux_giou = s_boxes.new_zeros(())
        for aux in student_out["aux_outputs"]:
            a_logits = aux["pred_logits"]
            if a_logits.shape[-1] > K_student:
                a_logits = a_logits[..., :K_student]
            a_boxes = aux["pred_boxes"]
            aux_kl = aux_kl + (
                w * _bernoulli_kl_per_query(a_logits, t_logits, cfg.distill_temperature)
            ).sum() / w_sum
            aux_l1 = aux_l1 + (
                w * (a_boxes - t_boxes).abs().sum(dim=-1)
            ).sum() / w_sum
            a_giou = _elementwise_giou(
                _cxcywh_to_xyxy(a_boxes), _cxcywh_to_xyxy(t_boxes)
            )
            aux_giou = aux_giou + (w * (1.0 - a_giou)).sum() / w_sum
            n_aux += 1
        if n_aux > 0:
            loss_kl = loss_kl + aux_kl / n_aux
            loss_l1 = loss_l1 + aux_l1 / n_aux
            loss_giou = loss_giou + aux_giou / n_aux

    loss_distill = (
        cfg.distill_kl_weight * loss_kl
        + cfg.distill_l1_weight * loss_l1
        + cfg.distill_giou_weight * loss_giou
    )
    return {
        "loss_distill_kl": loss_kl,
        "loss_distill_l1": loss_l1,
        "loss_distill_giou": loss_giou,
        "loss_distill": loss_distill,
    }
