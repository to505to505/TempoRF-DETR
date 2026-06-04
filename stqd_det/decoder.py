"""Stenosis decoder - DiffusionDet-style cascaded RCNN head.

Boxes stay in signed-normalised cxcywh [-sigma, sigma] throughout. Returns
the final stage plus per-stage outputs for deep supervision.
"""

from typing import Dict, List, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from .noise import from_signed, to_signed


# timestep embedding 


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding + 2-layer MLP (DDPM convention)."""

    def __init__(self, d_model: int, hidden: int = None):
        super().__init__()
        hidden = hidden or d_model * 4
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long -> (B, d_model)."""
        device = t.device
        half = self.d_model // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        ang = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        return self.mlp(emb)


# dynamic interaction 


class DynamicHead(nn.Module):
    """Instance-conditioned 1x1 conv pair (DiffusionDet / Sparse R-CNN).

    Query vector predicts the params of two 1x1 convs applied to the RoI feature.
    """

    def __init__(self, d_model: int, dim_dynamic: int = 64, num_dynamic: int = 2):
        super().__init__()
        if num_dynamic != 2:
            raise ValueError("only num_dynamic=2 is implemented (matches paper)")
        self.d_model = d_model
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        # two convs: d_model x dim_dynamic, then dim_dynamic x d_model
        self.num_params = d_model * dim_dynamic + dim_dynamic * d_model
        self.dynamic_layer = nn.Linear(d_model, self.num_params)
        self.norm1 = nn.LayerNorm(dim_dynamic)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        roi_feat: torch.Tensor,           # (B*N, roi_size*roi_size, C)
        pro_feat: torch.Tensor,           # (B*N, C)
    ) -> torch.Tensor:                    # (B*N, C)
        params = self.dynamic_layer(pro_feat)                       # (B*N, P)
        p1, p2 = params.split(
            [self.d_model * self.dim_dynamic, self.dim_dynamic * self.d_model],
            dim=-1,
        )
        BN = roi_feat.shape[0]
        w1 = p1.view(BN, self.d_model, self.dim_dynamic)
        w2 = p2.view(BN, self.dim_dynamic, self.d_model)

        x = torch.bmm(roi_feat, w1)                                 # (B*N, S, dim_dynamic)
        x = self.norm1(x)
        x = self.activation(x)
        x = torch.bmm(x, w2)                                        # (B*N, S, d_model)
        x = self.norm2(x)
        x = self.activation(x)
        return x.mean(dim=1)


# single cascade head 


class RCNNHead(nn.Module):
    """One cascade stage."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        roi_size: int,
        dim_dynamic: int,
        num_dynamic: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.roi_size = roi_size

        # Self-attention over the N proposals (per-frame).
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.inst_interact = DynamicHead(d_model, dim_dynamic, num_dynamic)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # FFN.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)

        # Heads.
        self.cls_head = nn.Linear(d_model, num_classes)
        self.box_head = nn.Linear(d_model, 4)

        # Per-stage timestep conditioning.
        self.time_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        roi_feat: torch.Tensor,                # (B*N, d_model, S, S)
        pro_feat: torch.Tensor,                # (B*N, d_model)
        num_proposals: int,
        time_emb: torch.Tensor,                # (B, d_model)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        BN, C, S, S2 = roi_feat.shape
        assert S == S2 and C == self.d_model

        # self-attn over the N proposals
        B = BN // num_proposals
        q = pro_feat.view(B, num_proposals, C)
        attn_out, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.norm1(q + self.dropout1(attn_out))                 # (B, N, C)

        q = q + self.time_proj(time_emb).unsqueeze(1)               # (B,1,C)

        # dynamic instance interaction with the RoI feature
        pro_feat = q.view(B * num_proposals, C)
        roi_tokens = roi_feat.view(BN, C, S * S).permute(0, 2, 1)   # (B*N, S*S, C)
        inst_out = self.inst_interact(roi_tokens, pro_feat)         # (B*N, C)
        pro_feat = self.norm2(pro_feat + self.dropout2(inst_out))

        ffn = self.linear2(self.dropout3(self.activation(self.linear1(pro_feat))))
        pro_feat = self.norm3(pro_feat + self.dropout4(ffn))

        logits = self.cls_head(pro_feat).view(B, num_proposals, -1)
        delta = self.box_head(pro_feat).view(B, num_proposals, 4)
        return logits, delta, pro_feat.view(B, num_proposals, C)


# decoder wrapper 


_MIN_BOX_SIDE = 0.01      # floor box w/h so RoIAlign never gets a degenerate region
_DELTA_WH_CLIP = 4.0      # clamp dw,dh before exp()


def _signed_to_xyxy_abs(
    boxes_signed: torch.Tensor,
    img_size: int,
    sigma: float,
) -> torch.Tensor:
    """Signed-normalised cxcywh -> absolute xyxy. Clamps w/h to _MIN_BOX_SIDE;
    zero-area boxes otherwise break torchvision's level assignment silently."""
    boxes = from_signed(boxes_signed, sigma)                       # cxcywh [0,1]
    cx, cy, w, h = boxes.unbind(-1)
    w = w.clamp(min=_MIN_BOX_SIDE, max=1.0)
    h = h.clamp(min=_MIN_BOX_SIDE, max=1.0)
    x1 = (cx - w / 2).clamp(0, 1) * img_size
    y1 = (cy - h / 2).clamp(0, 1) * img_size
    x2 = (cx + w / 2).clamp(0, 1) * img_size
    y2 = (cy + h / 2).clamp(0, 1) * img_size
    return torch.stack([x1, y1, x2, y2], dim=-1)


def apply_box_delta(
    prev_signed: torch.Tensor,         # (..., 4) signed-normalised
    delta: torch.Tensor,               # (..., 4) raw decoder output
    sigma: float,
) -> torch.Tensor:
    """Sparse-R-CNN / DiffusionDet-style multiplicative refinement in cxcywh.

    Multiplicative w/h (vs additive) avoids the degeneracy where w_signed
    drifted to -12 across the cascade (2026-05-18 diagnostic).
    """
    prev_norm = from_signed(prev_signed, sigma)                    # (..., 4) in [0,1]
    cx, cy, w, h = prev_norm.unbind(-1)
    w = w.clamp(min=_MIN_BOX_SIDE)
    h = h.clamp(min=_MIN_BOX_SIDE)
    dx, dy, dw, dh = delta.unbind(-1)
    dw = dw.clamp(-_DELTA_WH_CLIP, _DELTA_WH_CLIP)
    dh = dh.clamp(-_DELTA_WH_CLIP, _DELTA_WH_CLIP)

    cx_new = (cx + dx * w).clamp(0.0, 1.0)
    cy_new = (cy + dy * h).clamp(0.0, 1.0)
    w_new = (w * torch.exp(dw)).clamp(min=_MIN_BOX_SIDE, max=1.0)
    h_new = (h * torch.exp(dh)).clamp(min=_MIN_BOX_SIDE, max=1.0)

    new_norm = torch.stack([cx_new, cy_new, w_new, h_new], dim=-1)
    return to_signed(new_norm, sigma)


class StenosisDecoder(nn.Module):
    """Cascaded RCNN head over noised box proposals."""

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 1,
        num_heads_cascade: int = 6,
        roi_size: int = 7,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        dim_dynamic: int = 64,
        num_dynamic: int = 2,
        img_size: int = 512,
        sigma: float = 2.0,
        fpn_levels: List[str] = None,
        fpn_strides: Dict[str, int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_heads_cascade = num_heads_cascade
        self.roi_size = roi_size
        self.img_size = img_size
        self.sigma = sigma

        self.fpn_levels = fpn_levels or ["P2", "P3", "P4", "P5"]
        self.fpn_strides = fpn_strides or {"P2": 4, "P3": 8, "P4": 16, "P5": 32}

        self.roi_align = MultiScaleRoIAlign(
            featmap_names=self.fpn_levels,
            output_size=roi_size,
            sampling_ratio=2,
            canonical_scale=224,
            canonical_level=4,
        )

        self.time_embed = SinusoidalTimeEmbedding(d_model)

        # initial proposal query, from the flattened RoI feature
        self.init_proj = nn.Linear(d_model * roi_size * roi_size, d_model)

        self.heads = nn.ModuleList(
            [
                RCNNHead(
                    d_model=d_model,
                    num_classes=num_classes,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    roi_size=roi_size,
                    dim_dynamic=dim_dynamic,
                    num_dynamic=num_dynamic,
                )
                for _ in range(num_heads_cascade)
            ]
        )

    def _roi_pool(
        self,
        features: Dict[str, torch.Tensor],
        boxes_signed: torch.Tensor,            # (BT, N, 4) signed-normalised
    ) -> torch.Tensor:
        """Return (BT*N, d_model, S, S) RoI-pooled feature."""
        BT, N, _ = boxes_signed.shape
        abs_xyxy = _signed_to_xyxy_abs(boxes_signed, self.img_size, self.sigma)
        boxes_list = [abs_xyxy[b] for b in range(BT)]
        image_shapes = [(self.img_size, self.img_size)] * BT
        roi = self.roi_align(features, boxes_list, image_shapes)   # (BT*N, C, S, S)
        return roi

    def forward(
        self,
        features: Dict[str, torch.Tensor],     # FPN dict, each (BT, C, h, w)
        boxes_signed: torch.Tensor,            # (BT, N, 4) signed-normalised noised init
        t: torch.Tensor,                       # (BT,) long
    ) -> Dict[str, torch.Tensor]:
        """Returns dict of per-stage predictions:
            pred_logits_stages: (K, BT, N, num_classes)
            pred_boxes_stages:  (K, BT, N, 4) signed-normalised
            final_logits:       (BT, N, num_classes)
            final_boxes:        (BT, N, 4) signed-normalised
            queries_final:      (BT, N, d_model) - used by STFS aggregator
        """
        if boxes_signed.dim() != 3 or boxes_signed.shape[-1] != 4:
            raise ValueError(f"boxes_signed must be (BT,N,4); got {tuple(boxes_signed.shape)}")
        BT, N, _ = boxes_signed.shape
        if t.shape != (BT,):
            raise ValueError(f"t must be (BT,); got {tuple(t.shape)}")

        time_emb = self.time_embed(t)                              # (BT, d_model)

        roi = self._roi_pool(features, boxes_signed)               # (BT*N, C, S, S)
        BN = BT * N
        init_q = roi.view(BN, -1)
        pro_feat = self.init_proj(init_q)                          # (BT*N, d_model)

        per_stage_logits: List[torch.Tensor] = []
        per_stage_boxes: List[torch.Tensor] = []
        cur_boxes = boxes_signed

        for stage in self.heads:
            logits, delta, queries = stage(roi, pro_feat, N, time_emb)
            cur_boxes = apply_box_delta(cur_boxes, delta, self.sigma)
            per_stage_logits.append(logits)
            per_stage_boxes.append(cur_boxes)

            # re-pool with refined boxes, carry queries to next stage
            roi = self._roi_pool(features, cur_boxes)
            pro_feat = queries.view(BN, self.d_model)

        return {
            "pred_logits_stages": torch.stack(per_stage_logits, dim=0),  # (K,BT,N,C)
            "pred_boxes_stages": torch.stack(per_stage_boxes, dim=0),    # (K,BT,N,4)
            "final_logits": per_stage_logits[-1],
            "final_boxes": per_stage_boxes[-1],
            "queries_final": queries,
        }
