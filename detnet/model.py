"""Stenosis-DetNet model: SFF box head + a Faster R-CNN wrapper around it.

Reuses torchvision's RoIHeads helpers for sample selection / loss.
Deviation from the paper: we keep the standard RPN instead of Guided Anchoring.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection._utils import (
    BalancedPositiveNegativeSampler,
    BoxCoder,
    Matcher,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from .config import Config


# SFF block
class SFFBlock(nn.Module):
    """One multi-head self-attention block over the whole pool of box features,
    with a residual add. Input/output dim is d_in (flattened ROI feature)."""

    def __init__(self, d_in: int, d_model: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % heads == 0, (
            f"d_model ({d_model}) must be divisible by heads ({heads})"
        )
        self.d_in = d_in
        self.d_model = d_model
        self.heads = heads
        self.d_head = d_model // heads

        self.to_q = nn.Linear(d_in, d_model)
        self.to_k = nn.Linear(d_in, d_model)
        self.to_v = nn.Linear(d_in, d_model)

        # back to d_in for the residual add
        self.proj_out = nn.Linear(d_model, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N_tok, d_in) flat box features across all (b, t, m); returns same shape
        if x.numel() == 0:
            return x
        N = x.shape[0]
        H, Dh = self.heads, self.d_head

        # (N, d_model) -> (N, H, Dh) -> (H, N, Dh)
        q = self.to_q(x).reshape(N, H, Dh).transpose(0, 1)
        k = self.to_k(x).reshape(N, H, Dh).transpose(0, 1)
        v = self.to_v(x).reshape(N, H, Dh).transpose(0, 1)

        # Scaled dot-product attention per head: (H, N, N) @ (H, N, Dh) -> (H, N, Dh)
        scale = Dh ** -0.5
        attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                  # (H, N, Dh)
        out = out.transpose(0, 1).contiguous().reshape(N, H * Dh)  # (N, d_model)
        out = self.proj_out(out)                        # (N, d_in)
        return x + out


class SFFBoxHead(nn.Module):
    """ROI-Align the boxes, run one global SFF block over the whole (B,T,M) pool,
    and return flat features in the original proposal order."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.T
        self.C = cfg.backbone_out_channels
        self.wr = cfg.roi_size
        self.hr = cfg.roi_size
        self.d_in = self.C * self.wr * self.hr        # 256 * 7 * 7 = 12544

        self.sff = SFFBlock(
            d_in=self.d_in,
            d_model=cfg.sff_token_dim,
            heads=cfg.sff_heads,
            dropout=cfg.sff_dropout,
        )

        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=cfg.roi_size,
            sampling_ratio=2,
        )

    @property
    def out_features(self) -> int:
        return self.d_in

    def forward(
        self,
        per_frame_features: List[Dict[str, torch.Tensor]],
        proposals_per_bt: List[torch.Tensor],
        batch_idx_per_bt: List[int],
        frame_idx_per_bt: List[int],
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Pool + fuse across all (b, t) groups.

        proposals_per_bt holds (S_i, 4) xyxy pixel boxes per group; returns
        (features, sizes) with features (sum_i S_i, d_in) in input order.
        """
        T = self.T
        sizes = [p.shape[0] for p in proposals_per_bt]

        any_level = next(iter(per_frame_features[0].values()))
        B = any_level.shape[0]

        # MultiScaleRoIAlign wants a List[Tensor] of length B per call, so we
        # call it once per frame t with that frame's boxes for each b.
        per_b_per_t_boxes: List[List[torch.Tensor]] = [
            [
                proposals_per_bt[0].new_zeros((0, 4))
                for _ in range(T)
            ]
            for _ in range(B)
        ]
        # drop each group's boxes into its [b][t] slot; concat if a (b,t) repeats
        for i, (b, t) in enumerate(zip(batch_idx_per_bt, frame_idx_per_bt)):
            existing = per_b_per_t_boxes[b][t]
            if existing.shape[0] == 0:
                per_b_per_t_boxes[b][t] = proposals_per_bt[i]
            else:
                per_b_per_t_boxes[b][t] = torch.cat(
                    [existing, proposals_per_bt[i]], dim=0,
                )

        image_shapes = [image_size] * B

        # pool per frame; keep split sizes to reassemble groups in order later
        pooled_per_t: List[torch.Tensor] = []
        per_t_split_sizes: List[List[int]] = []   # per_t_split_sizes[t][b] = N_b^t
        for t in range(T):
            boxes_per_b = [per_b_per_t_boxes[b][t] for b in range(B)]
            per_t_split_sizes.append([bx.shape[0] for bx in boxes_per_b])
            pooled_t = self.roi_pool(
                per_frame_features[t], boxes_per_b, image_shapes,
            )
            pooled_per_t.append(pooled_t)

        # flatten everything into one (N_tok, d_in) tensor for SFF
        flat_features: List[torch.Tensor] = []
        slice_layout: List[Tuple[int, int, int]] = []  # (t, b, count) for reassembly
        for t in range(T):
            offset = 0
            for b in range(B):
                count = per_t_split_sizes[t][b]
                if count > 0:
                    block = pooled_per_t[t][offset : offset + count]
                    # (count, C, wr, hr) -> (count, d_in)
                    flat_features.append(block.reshape(count, self.d_in))
                slice_layout.append((t, b, count))
                offset += count

        if len(flat_features) == 0:
            empty = pooled_per_t[0].new_zeros((0, self.d_in))
            return empty, sizes
        tokens = torch.cat(flat_features, dim=0)        # (N_tok, d_in)

        fused = self.sff(tokens)                        # (N_tok, d_in)

        # reassemble into per-(b, t) blocks keyed by (t, b)
        bt_to_block: Dict[Tuple[int, int], torch.Tensor] = {}
        cursor = 0
        for (t, b, count) in slice_layout:
            if count == 0:
                bt_to_block[(t, b)] = fused.new_zeros((0, self.d_in))
            else:
                bt_to_block[(t, b)] = fused[cursor : cursor + count]
                cursor += count

        # reorder back into proposals_per_bt order, slicing concatenated
        # (b, t) groups in turn; bt_cursor tracks how much of each is consumed
        out_blocks: List[torch.Tensor] = []
        bt_cursor: Dict[Tuple[int, int], int] = {}
        for i, (b, t) in enumerate(zip(batch_idx_per_bt, frame_idx_per_bt)):
            S_i = sizes[i]
            cur = bt_cursor.get((t, b), 0)
            block = bt_to_block[(t, b)][cur : cur + S_i]
            bt_cursor[(t, b)] = cur + S_i
            out_blocks.append(block)
        if len(out_blocks) == 0:
            return fused.new_zeros((0, self.d_in)), sizes
        out = torch.cat(out_blocks, dim=0)
        return out, sizes


# VideoFasterRCNN
class VideoFasterRCNN(nn.Module):
    """End-to-end Stenosis-DetNet detector. Forward matches psstt.model.VideoFasterRCNN
    for drop-in comparison. SCA runs separately in detnet.evaluate, not here."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.num_classes_with_bg = cfg.num_classes + 1

        weights = (
            FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if cfg.pretrained_coco
            else None
        )
        frcnn = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=None,
            rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train,
            rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
            rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
            box_score_thresh=cfg.box_score_thresh,
            box_nms_thresh=cfg.box_nms_thresh,
            box_detections_per_img=cfg.box_detections_per_img,
            box_fg_iou_thresh=cfg.box_fg_iou_thresh,
            box_bg_iou_thresh=cfg.box_bg_iou_thresh,
            box_batch_size_per_image=cfg.box_batch_size_per_image,
            box_positive_fraction=cfg.box_positive_fraction,
        )
        self.backbone = frcnn.backbone
        self.rpn = frcnn.rpn
        self.tv_transform = frcnn.transform

        # reuse RoIHeads' matcher / sampler / box-coder
        rh = frcnn.roi_heads
        self.box_coder: BoxCoder = rh.box_coder
        self.proposal_matcher: Matcher = rh.proposal_matcher
        self.fg_bg_sampler: BalancedPositiveNegativeSampler = rh.fg_bg_sampler
        self.box_score_thresh = float(rh.score_thresh)
        self.box_nms_thresh = float(rh.nms_thresh)
        self.box_detections_per_img = int(rh.detections_per_img)

        self.sff_head = SFFBoxHead(cfg)
        self.box_predictor = FastRCNNPredictor(
            in_channels=self.sff_head.out_features,
            num_classes=self.num_classes_with_bg,
        )

    def get_param_groups(self):
        backbone_params, other_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(p)
            else:
                other_params.append(p)
        return [
            {"params": backbone_params, "lr": self.cfg.lr * self.cfg.lr_backbone_mult},
            {"params": other_params, "lr": self.cfg.lr},
        ]

    # torchvision-style training-sample helpers (per frame)
    def _assign_targets_to_proposals(
        self,
        proposals: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        matched_idxs, labels = [], []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
            proposals, gt_boxes, gt_labels,
        ):
            if gt_boxes_in_image.numel() == 0:
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device,
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device,
                )
            else:
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes_in_image, proposals_in_image,
                )
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1
            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def _subsample(self, labels: List[torch.Tensor]) -> List[torch.Tensor]:
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds: List[torch.Tensor] = []
        for img_sampled_pos_inds, img_sampled_neg_inds in zip(
            sampled_pos_inds, sampled_neg_inds,
        ):
            img_sampled_inds = torch.where(img_sampled_pos_inds | img_sampled_neg_inds)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(
        self,
        proposals: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ):
        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        proposals = [
            torch.cat([p, gt_b]) if gt_b.numel() > 0 else p
            for p, gt_b in zip(proposals, gt_boxes)
        ]
        matched_idxs, labels = self._assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels,
        )
        sampled_inds = self._subsample(labels)
        matched_gt_boxes, out_proposals, out_labels = [], [], []
        for i, idxs in enumerate(sampled_inds):
            out_proposals.append(proposals[i][idxs])
            out_labels.append(labels[i][idxs])
            if gt_boxes[i].numel() == 0:
                matched_gt_boxes.append(
                    torch.zeros((idxs.shape[0], 4), device=proposals[i].device,
                                dtype=proposals[i].dtype),
                )
            else:
                matched_gt_boxes.append(gt_boxes[i][matched_idxs[i][idxs]])
        reg_targets = self.box_coder.encode(matched_gt_boxes, out_proposals)
        return out_proposals, out_labels, reg_targets

    def _backbone_per_frame(
        self,
        images_btchw: torch.Tensor,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        B, T = images_btchw.shape[:2]
        x = images_btchw.reshape(B * T, *images_btchw.shape[2:])
        feats = self.backbone(x)
        per_frame: List[Dict[str, torch.Tensor]] = []
        for t in range(T):
            d: Dict[str, torch.Tensor] = {}
            for k, v in feats.items():
                vv = v.reshape(B, T, *v.shape[1:])
                d[k] = vv[:, t].contiguous()
            per_frame.append(d)
        return per_frame, feats

    def _image_list(self, images_btchw: torch.Tensor) -> ImageList:
        B, T, _, H, W = images_btchw.shape
        flat = images_btchw.reshape(B * T, 3, H, W)
        sizes = [(H, W)] * (B * T)
        return ImageList(flat, sizes)

    def _postprocess_detections_for_frame(
        self,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
    ):
        device = class_logits.device
        num_classes = self.num_classes_with_bg
        boxes_per_image = [p.shape[0] for p in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes, all_scores, all_labels = [], [], []
        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes,
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device).view(1, -1).expand_as(scores)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            inds = torch.where(scores > self.box_score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            keep = box_ops.batched_nms(boxes, scores, labels, self.box_nms_thresh)
            keep = keep[: self.box_detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels

    def forward(
        self,
        images_btchw: torch.Tensor,
        targets: Optional[List[List[Dict[str, torch.Tensor]]]] = None,
    ):
        """images_btchw: (B, T, 3, H, W) normalised frames; targets needed for training.

        Train returns a dict of scalar losses. Eval returns 'centre' (list[B] dets)
        and, when cfg.supervise_all_frames, 'all_frames' (list[B] of list[T]).
        """
        cfg = self.cfg
        is_train = self.training and targets is not None
        B, T = images_btchw.shape[:2]
        H, W = images_btchw.shape[3], images_btchw.shape[4]
        device = images_btchw.device

        per_frame_features, feats_flat = self._backbone_per_frame(images_btchw)
        image_list = self._image_list(images_btchw)

        # RPN
        if is_train:
            assert targets is not None
            rpn_targets: List[Dict[str, torch.Tensor]] = []
            for b in range(B):
                for t in range(T):
                    td = targets[b][t]
                    rpn_targets.append({
                        "boxes": td["boxes"].to(device),
                        "labels": td["labels"].to(device),
                    })
        else:
            rpn_targets = None
        proposals_flat, rpn_losses = self.rpn(image_list, feats_flat, rpn_targets)

        # which frames to run the head on: all frames at eval (SCA needs them)
        # and when supervising all frames; otherwise just the centre frame
        if cfg.supervise_all_frames or not is_train:
            ref_frames = list(range(T))
        else:
            ref_frames = [T // 2]

        # build (b, t) work list
        bt_proposals: List[torch.Tensor] = []
        bt_b: List[int] = []
        bt_t: List[int] = []
        bt_targets: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            for t in ref_frames:
                flat_idx = b * T + t
                bt_proposals.append(proposals_flat[flat_idx])
                bt_b.append(b)
                bt_t.append(t)
                if is_train:
                    bt_targets.append({
                        "boxes": targets[b][t]["boxes"].to(device),
                        "labels": targets[b][t]["labels"].to(device),
                    })

        # training-sample selection per (b, t)
        if is_train:
            sampled_props, sampled_labels, reg_targets = self.select_training_samples(
                bt_proposals, bt_targets,
            )
            sff_proposals = sampled_props
        else:
            sff_proposals = bt_proposals
            sampled_labels = None
            reg_targets = None

        # SFF head: one global self-attention pool across (b, t, m)
        sff_feats, _sizes = self.sff_head(
            per_frame_features=per_frame_features,
            proposals_per_bt=sff_proposals,
            batch_idx_per_bt=bt_b,
            frame_idx_per_bt=bt_t,
            image_size=(H, W),
        )

        class_logits, box_regression = self.box_predictor(sff_feats)

        if is_train:
            loss_cls, loss_box = fastrcnn_loss(
                class_logits, box_regression, sampled_labels, reg_targets,
            )
            losses = dict(rpn_losses)
            losses["loss_classifier"] = loss_cls
            losses["loss_box_reg"] = loss_box
            return losses

        # eval: postprocess and split predictions back per (b, t)
        image_shapes = [(H, W)] * len(sff_proposals)
        all_boxes, all_scores, all_labels = self._postprocess_detections_for_frame(
            class_logits, box_regression, sff_proposals, image_shapes,
        )

        per_bt: List[Dict[str, torch.Tensor]] = []
        for boxes, scores, labels in zip(all_boxes, all_scores, all_labels):
            per_bt.append({"boxes": boxes, "scores": scores, "labels": labels})

        # pack back into (b, t) layout
        per_b_per_t: List[List[Dict[str, torch.Tensor]]] = [
            [{} for _ in ref_frames] for _ in range(B)
        ]
        for i, (b, t) in enumerate(zip(bt_b, bt_t)):
            ref_idx = ref_frames.index(t)
            per_b_per_t[b][ref_idx] = per_bt[i]

        out: Dict[str, object] = {}
        if len(ref_frames) == T:
            out["all_frames"] = per_b_per_t
        centre = T // 2
        if centre in ref_frames:
            ci = ref_frames.index(centre)
            out["centre"] = [per_b_per_t[b][ci] for b in range(B)]
        else:
            out["centre"] = [per_b_per_t[b][0] for b in range(B)]
        return out
