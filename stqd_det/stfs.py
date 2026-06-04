"""Spatio-Temporal Feature Sharing (STFS).

Links stage-1 detections into cross-frame chains (Hungarian on
1-IoU + L1), votes them into H-TP / H-FN / H-FP, then for H-FN/H-FP
chains injects shared RoI features into the stage-2 noise pool.
Cross-frame matching is plain numpy/scipy; only the aggregator carries
gradients.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops import MultiScaleRoIAlign

from .config import Config
from .dynamic_conv import DynamicConv
from .noise import from_signed, to_signed


# geometry helpers 


def _cxcywh_to_xyxy(box_cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box_cxcywh.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _xyxy_to_cxcywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box_xyxy.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1)


def _pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """boxes_a (M,4) cxcywh, boxes_b (N,4) cxcywh -> IoU matrix (M, N)."""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros(boxes_a.shape[0], boxes_b.shape[0])
    a = _cxcywh_to_xyxy(boxes_a)
    b = _cxcywh_to_xyxy(boxes_b)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter_wh = (rb - lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    union = area_a + area_b - inter + 1e-7
    return inter / union


def _manhattan(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise Manhattan distance between centre points; (M, N)."""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros(boxes_a.shape[0], boxes_b.shape[0])
    return torch.cdist(boxes_a[:, :2], boxes_b[:, :2], p=1)


# matching + voting 


def _filter_by_confidence(
    logits: torch.Tensor,            # (N, num_classes) raw logits
    boxes_norm: torch.Tensor,        # (N, 4) cxcywh in [0, 1]
    conf_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = logits.sigmoid().max(dim=-1).values
    keep = scores >= conf_thresh
    return boxes_norm[keep], scores[keep], keep


def build_chains(
    boxes_per_frame: List[torch.Tensor],     # T lists, each (N_t, 4) cxcywh [0,1]
    scores_per_frame: List[torch.Tensor],
    iou_match: float = 1.0,                  # weight on (1 - IoU)
    l1_match: float = 1e-4,                  # weight on Manhattan (tiebreak)
    iou_floor: float = 0.1,                  # below this, refuse to match
) -> List[List[Tuple[int, int]]]:
    """Grow chains by Hungarian-matching consecutive frames.

    Returns a list of chains, each a list of (frame_idx, box_idx). Every
    box lands in exactly one chain.
    """
    T = len(boxes_per_frame)
    chains: List[List[Tuple[int, int]]] = []
    # prev-frame box-index -> chain id, only for chains still alive at f-1.
    active_chain: Dict[int, int] = {}

    for f in range(T):
        cur_boxes = boxes_per_frame[f]
        n_cur = cur_boxes.shape[0]
        if n_cur == 0:
            active_chain = {}
            continue

        if not active_chain:
            active_chain = {}
            for i in range(n_cur):
                chains.append([(f, i)])
                active_chain[i] = len(chains) - 1
            continue

        prev_indices = sorted(active_chain.keys())
        prev_chain_ids = [active_chain[i] for i in prev_indices]
        prev_boxes = boxes_per_frame[f - 1][torch.tensor(prev_indices, dtype=torch.long)]
        iou = _pairwise_iou(prev_boxes, cur_boxes)
        manh = _manhattan(prev_boxes, cur_boxes)
        cost_np = (iou_match * (1.0 - iou) + l1_match * manh).cpu().numpy()
        iou_np = iou.cpu().numpy()
        cost_np = np.where(iou_np < iou_floor, 1e6, cost_np)
        r_idx, c_idx = linear_sum_assignment(cost_np)

        matched_cur: set = set()
        new_active: Dict[int, int] = {}
        for r, c in zip(r_idx, c_idx):
            if cost_np[r, c] >= 1e6:
                continue
            chain_id = prev_chain_ids[r]
            chains[chain_id].append((f, int(c)))
            matched_cur.add(int(c))
            new_active[int(c)] = chain_id
        for i in range(n_cur):
            if i not in matched_cur:
                chains.append([(f, i)])
                new_active[i] = len(chains) - 1
        active_chain = new_active

    return chains


def vote_chains(
    chains: List[List[Tuple[int, int]]],
    T: int,
) -> Dict[str, List[int]]:
    """Classify chains by frame span into H_TP / H_FN / H_FP.

    Returns a dict of chain-index lists.
    """
    tp, fn, fp = [], [], []
    threshold_high = T          # full span
    threshold_low = max(T // 2, 1)
    for ci, chain in enumerate(chains):
        n = len(chain)
        if n >= threshold_high:
            tp.append(ci)
        elif n >= threshold_low:
            fn.append(ci)
        else:
            fp.append(ci)
    return {"H_TP": tp, "H_FN": fn, "H_FP": fp}


# RoI feature aggregator 


class RoIAggregator(nn.Module):
    """MHA + DynamicConv + 2-layer FC RoI feature aggregator."""

    def __init__(self, channels: int, roi_size: int, heads: int, dc_kernels: int):
        super().__init__()
        self.channels = channels
        self.roi_size = roi_size
        self.mha = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(channels)
        self.dc = DynamicConv(channels * 2, channels, kernel_size=3, K=dc_kernels)
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)
        self.act = nn.ReLU(inplace=True)
        self.ln2 = nn.LayerNorm(channels)

    def forward(
        self,
        query_feat: torch.Tensor,            # (1, C, S, S) - the missing frame's RoI
        reference_feats: torch.Tensor,       # (M, C, S, S) - present frames' RoIs
    ) -> torch.Tensor:                       # (1, C, S, S) - context-enhanced query
        C, S = self.channels, self.roi_size
        if query_feat.shape[1:] != (C, S, S):
            raise ValueError(
                f"query_feat shape {tuple(query_feat.shape)} != (1,{C},{S},{S})"
            )
        if reference_feats.shape[1:] != (C, S, S):
            raise ValueError(
                f"reference_feats shape {tuple(reference_feats.shape)} != (M,{C},{S},{S})"
            )
        q = query_feat.view(1, C, S * S).permute(0, 2, 1)            # (1, S*S, C)
        kv = reference_feats.view(-1, C, S * S).permute(0, 2, 1)     # (M, S*S, C)
        kv = kv.reshape(1, -1, C)                                    # (1, M*S*S, C)

        attn, _ = self.mha(q, kv, kv, need_weights=False)            # (1, S*S, C)
        q_prime = self.ln1(attn + q)

        # DC over concat(query, attended-query), same recipe as GFE.
        f_q = q.permute(0, 2, 1).view(1, C, S, S)
        f_qp = q_prime.permute(0, 2, 1).view(1, C, S, S)
        dc_in = torch.cat([f_q, f_qp], dim=1)
        dc_out = self.dc(dc_in)                                      # (1, C, S, S)
        dc_tok = dc_out.view(1, C, S * S).permute(0, 2, 1)
        ffn = self.fc2(self.act(self.fc1(dc_tok)))
        out_tok = self.ln2(ffn + q)
        return out_tok.permute(0, 2, 1).view(1, C, S, S)


# STFS module 


class STFS(nn.Module):
    """End-to-end STFS for a batch of clips.

    Forward takes stage-1 boxes/logits + FPN features and returns the
    stage-2 box init (H-FN/H-FP injected), per-frame counts and n_r.
    """

    def __init__(self, cfg: Config, fpn_levels: List[str], fpn_strides: Dict[str, int]):
        super().__init__()
        self.cfg = cfg
        self.aggregator = RoIAggregator(
            channels=cfg.fpn_out_channels,
            roi_size=cfg.decoder_roi_size,
            heads=cfg.gfe_heads,
            dc_kernels=cfg.gfe_dc_kernels,
        )
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=fpn_levels,
            output_size=cfg.decoder_roi_size,
            sampling_ratio=2,
            canonical_scale=224,
            canonical_level=4,
        )
        # Mirrors decoder.init_proj so projected queries are compatible.
        self.query_proj = nn.Linear(
            cfg.fpn_out_channels * cfg.decoder_roi_size * cfg.decoder_roi_size,
            cfg.fpn_out_channels,
        )

    # per-clip helpers 

    def _build_h_fn_rois(
        self,
        chain: List[Tuple[int, int]],
        boxes_per_frame_norm: List[torch.Tensor],
        T: int,
    ) -> Dict[str, torch.Tensor]:
        """For an H-FN chain: union bbox of present frames + the list of
        missing frame indices to pool it on."""
        present_frames = [f for f, _ in chain]
        present_boxes = torch.stack(
            [boxes_per_frame_norm[f][i] for f, i in chain], dim=0
        )
        xyxy = _cxcywh_to_xyxy(present_boxes)
        x1 = xyxy[:, 0].min()
        y1 = xyxy[:, 1].min()
        x2 = xyxy[:, 2].max()
        y2 = xyxy[:, 3].max()
        union = _xyxy_to_cxcywh(torch.stack([x1, y1, x2, y2]))
        missing = torch.tensor(
            [f for f in range(T) if f not in present_frames], dtype=torch.long
        )
        return {"union_box": union, "missing_frames": missing, "present_frames": present_frames}

    def _build_h_fp_roi(
        self,
        chain: List[Tuple[int, int]],
        boxes_per_frame_norm: List[torch.Tensor],
        alpha: float,
    ) -> torch.Tensor:
        """H-FP box: mean cxcywh over the chain, w/h scaled by alpha."""
        cs = torch.stack([boxes_per_frame_norm[f][i] for f, i in chain], dim=0)
        cx_cy = cs[:, :2].mean(dim=0)
        wh = cs[:, 2:].mean(dim=0) * alpha
        return torch.cat([cx_cy, wh], dim=0).clamp(0.0, 1.0)

    # consistency loss 

    def consistency_loss(
        self,
        counts_per_frame: torch.Tensor,        # (B, T)
        n_r: torch.Tensor,                     # (B,)
        beta: float,
    ) -> torch.Tensor:
        T = counts_per_frame.shape[1]
        diff = (counts_per_frame.float() - n_r.unsqueeze(1).float()).abs()
        return (diff / (T + beta)).mean()

    # main forward 

    def forward(
        self,
        stage1_logits: torch.Tensor,                      # (B*T, N, num_classes)
        stage1_boxes_signed: torch.Tensor,                # (B*T, N, 4) signed
        features: Dict[str, torch.Tensor],                # FPN dict (B*T, C, h, w)
        B: int,
        T: int,
    ) -> Dict[str, torch.Tensor]:
        BT, N, _ = stage1_boxes_signed.shape
        if BT != B * T:
            raise ValueError(f"BT={BT} != B*T = {B}*{T}")
        sigma = self.cfg.sigma_scale
        img = self.cfg.img_size
        stage1_boxes_norm = from_signed(stage1_boxes_signed, sigma)        # (BT, N, 4) in [0,1]

        # Stage-2 init starts as a copy of stage-1; injections overwrite slots.
        stage2_boxes_signed = stage1_boxes_signed.clone()

        counts_per_frame = torch.zeros((B, T), dtype=torch.long, device=stage1_boxes_signed.device)
        n_r = torch.zeros((B,), dtype=torch.long, device=stage1_boxes_signed.device)

        for b in range(B):
            clip_boxes_per_frame: List[torch.Tensor] = []
            clip_scores_per_frame: List[torch.Tensor] = []
            kept_indices_per_frame: List[torch.Tensor] = []
            for f in range(T):
                idx = b * T + f
                kept_boxes, kept_scores, keep = _filter_by_confidence(
                    stage1_logits[idx],
                    stage1_boxes_norm[idx],
                    self.cfg.stfs_conf_thresh,
                )
                clip_boxes_per_frame.append(kept_boxes)
                clip_scores_per_frame.append(kept_scores)
                kept_indices_per_frame.append(torch.nonzero(keep, as_tuple=False).flatten())
                counts_per_frame[b, f] = kept_boxes.shape[0]

            chains = build_chains(clip_boxes_per_frame, clip_scores_per_frame)
            groups = vote_chains(chains, T)
            n_r[b] = len(groups["H_TP"]) + len(groups["H_FN"])

            # H-FN: share features into the frames that missed the box.
            for ci in groups["H_FN"]:
                chain = chains[ci]
                info = self._build_h_fn_rois(chain, clip_boxes_per_frame, T)
                if info["missing_frames"].numel() == 0:
                    continue
                ref_feats_list = []
                for pf in info["present_frames"]:
                    feat_b = {k: v[b * T + pf : b * T + pf + 1] for k, v in features.items()}
                    box_xyxy = _cxcywh_to_xyxy(info["union_box"].unsqueeze(0)) * img
                    pooled = self.roi_align(feat_b, [box_xyxy], [(img, img)])
                    ref_feats_list.append(pooled)
                ref_feats = torch.cat(ref_feats_list, dim=0)
                missing_pixel_box = _cxcywh_to_xyxy(info["union_box"].unsqueeze(0)) * img
                for mf in info["missing_frames"]:
                    mf_idx = b * T + int(mf)
                    feat_mf = {k: v[mf_idx : mf_idx + 1] for k, v in features.items()}
                    q_feat = self.roi_align(feat_mf, [missing_pixel_box], [(img, img)])
                    agg = self.aggregator(q_feat, ref_feats)                # (1, C, S, S)
                    # Overwrite the lowest-confidence slot with the union box.
                    scores_mf = stage1_logits[mf_idx].sigmoid().max(dim=-1).values
                    worst_slot = int(scores_mf.argmin())
                    stage2_boxes_signed[mf_idx, worst_slot] = to_signed(
                        info["union_box"].unsqueeze(0), sigma
                    )[0]

            # H-FP: seed the avg-padded box (no aggregator path).
            for ci in groups["H_FP"]:
                chain = chains[ci]
                pad_box = self._build_h_fp_roi(
                    chain, clip_boxes_per_frame, self.cfg.stfs_alpha_pad
                )
                for f, _ in chain:
                    mf_idx = b * T + f
                    scores_mf = stage1_logits[mf_idx].sigmoid().max(dim=-1).values
                    worst_slot = int(scores_mf.argmin())
                    stage2_boxes_signed[mf_idx, worst_slot] = to_signed(
                        pad_box.unsqueeze(0), sigma
                    )[0]

        loss_num = self.consistency_loss(
            counts_per_frame, n_r, self.cfg.consistency_beta
        )

        return {
            "stage2_boxes_signed": stage2_boxes_signed,
            "counts_per_frame": counts_per_frame,
            "n_r": n_r,
            "loss_num": loss_num,
        }
