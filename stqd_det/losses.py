"""Set prediction criterion for STQD-Det (DiffusionDet-style, per-frame).

Per-frame Hungarian matching, sigmoid focal cls loss over all N proposals,
L1 + GIoU on matched only, deep-supervised over the K cascade stages.
Matcher reused from rf-detr (costs in cxcywh [0,1] space).
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from .config import Config
from .noise import from_signed


def _sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """Sigmoid focal loss (per-element)."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class SetCriterion(nn.Module):
    """Per-frame set criterion with cascade deep supervision.

    num_classes is the foreground class count (1 for stenosis).
    """

    def __init__(self, num_classes: int, cfg: Config):
        super().__init__()
        self.num_classes = num_classes
        self.cfg = cfg
        # matcher cost weights == loss weights (DiffusionDet convention)
        self.matcher = HungarianMatcher(
            cost_class=cfg.cls_weight,
            cost_bbox=cfg.l1_weight,
            cost_giou=cfg.giou_weight,
            focal_alpha=cfg.focal_alpha,
        )

    # per-frame loss 

    def _per_frame_stage_loss(
        self,
        pred_logits: torch.Tensor,             # (B, N, num_classes)
        pred_boxes_signed: torch.Tensor,       # (B, N, 4) signed-normalised
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        B, N, C = pred_logits.shape
        device = pred_logits.device
        # to [0,1] cxcywh for matching / losses
        pred_boxes_norm = from_signed(pred_boxes_signed, self.cfg.sigma_scale)

        # targets come off the DataLoader on CPU; matcher GIoU needs same device
        targets = [
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
            }
            for t in targets
        ]

        outputs_for_match = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes_norm,
        }
        indices = self.matcher(outputs_for_match, targets)

        target_classes = torch.zeros((B, N, C), device=device, dtype=pred_logits.dtype)
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            tgt_labels = targets[b]["labels"][tgt_idx].long()         # (M,)
            tgt_labels = tgt_labels.clamp(min=0, max=C - 1)
            target_classes[b, pred_idx, tgt_labels] = 1.0

        num_boxes_total = sum(len(t["labels"]) for t in targets)
        normaliser = max(num_boxes_total, 1)
        cls_loss = (
            _sigmoid_focal_loss(
                pred_logits,
                target_classes,
                alpha=self.cfg.focal_alpha,
                gamma=self.cfg.focal_gamma,
                reduction="sum",
            )
            / normaliser
        )

        # box losses, matched only
        if num_boxes_total == 0:
            l1 = pred_boxes_norm.sum() * 0.0
            giou_loss = pred_boxes_norm.sum() * 0.0
        else:
            src_boxes = []
            tgt_boxes = []
            for b, (pred_idx, tgt_idx) in enumerate(indices):
                if pred_idx.numel() == 0:
                    continue
                src_boxes.append(pred_boxes_norm[b, pred_idx])
                tgt_boxes.append(targets[b]["boxes"][tgt_idx])
            if src_boxes:
                src = torch.cat(src_boxes, dim=0)
                tgt = torch.cat(tgt_boxes, dim=0).to(src.device)
                l1 = F.l1_loss(src, tgt, reduction="none").sum() / normaliser
                giou = generalized_box_iou(
                    box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt)
                )
                # giou is pairwise; take the diagonal
                giou_loss = (1.0 - torch.diag(giou)).sum() / normaliser
            else:
                l1 = pred_boxes_norm.sum() * 0.0
                giou_loss = pred_boxes_norm.sum() * 0.0

        return {
            "loss_cls": cls_loss,
            "loss_l1": l1,
            "loss_giou": giou_loss,
            "indices": indices,
        }

    # top-level forward 

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets_per_frame: List[List[Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs:  decoder dict with
                        pred_logits_stages (K, B*T, N, num_classes)
                        pred_boxes_stages  (K, B*T, N, 4) signed-normalised
            targets_per_frame: list (B) of list (T) of {"labels", "boxes"}
        Returns:
            dict of {"loss_cls", "loss_l1", "loss_giou", "loss_total"}
        """
        K, BT, N, C = outputs["pred_logits_stages"].shape
        B = len(targets_per_frame)
        T = BT // B
        if B * T != BT:
            raise ValueError(f"BT={BT} not divisible by B={B}")

        # flatten clip targets to index b*T+f
        flat_targets: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            for f in range(T):
                flat_targets.append(targets_per_frame[b][f])

        cls_total = pred_zero = giou_total = l1_total = 0.0
        for k in range(K):
            stage = self._per_frame_stage_loss(
                outputs["pred_logits_stages"][k],
                outputs["pred_boxes_stages"][k],
                flat_targets,
            )
            cls_total = cls_total + stage["loss_cls"]
            l1_total = l1_total + stage["loss_l1"]
            giou_total = giou_total + stage["loss_giou"]

        cls_total = cls_total / K
        l1_total = l1_total / K
        giou_total = giou_total / K
        total = (
            self.cfg.cls_weight * cls_total
            + self.cfg.l1_weight * l1_total
            + self.cfg.giou_weight * giou_total
        )
        return {
            "loss_cls": cls_total,
            "loss_l1": l1_total,
            "loss_giou": giou_total,
            "loss_total": total,
        }

    # two-stage helper 

    def compute_total_loss(
        self,
        model_out: Dict[str, torch.Tensor],
        targets_per_frame: List[List[Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """Sum stage-1 + (optional) stage-2 + L_num into the training loss.

        Stage-2 / L_num are skipped if absent (STFS disabled).
        """
        stage1_losses = self(
            {
                "pred_logits_stages": model_out["stage1"]["pred_logits_stages"],
                "pred_boxes_stages": model_out["stage1"]["pred_boxes_stages"],
            },
            targets_per_frame,
        )
        total = stage1_losses["loss_total"]
        out: Dict[str, torch.Tensor] = {f"stage1/{k}": v for k, v in stage1_losses.items()}

        if "stage2" in model_out:
            stage2_losses = self(
                {
                    "pred_logits_stages": model_out["stage2"]["pred_logits_stages"],
                    "pred_boxes_stages": model_out["stage2"]["pred_boxes_stages"],
                },
                targets_per_frame,
            )
            total = total + stage2_losses["loss_total"]
            out.update({f"stage2/{k}": v for k, v in stage2_losses.items()})

        if "stfs" in model_out and "loss_num" in model_out["stfs"]:
            loss_num = model_out["stfs"]["loss_num"]
            out["loss_num"] = loss_num
            total = total + self.cfg.consistency_weight * loss_num

        out["loss_total"] = total
        return out
