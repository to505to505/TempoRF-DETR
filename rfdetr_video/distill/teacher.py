"""Frozen RF-DETR-Large teacher with a forward_video that folds T into the batch."""

from __future__ import annotations

from typing import Dict

import torch

from rfdetr_video.distill.frozen_teacher import FrozenRFDETRTeacher

from ..config import Config


class VideoFrozenRFDETRTeacher(FrozenRFDETRTeacher):
    """Teacher exposing a batched ``forward_video`` over T frames."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)  # type: ignore[arg-type]

    @torch.no_grad()
    def forward_video(self, frames_hr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """frames_hr: (B, T, 3, S, S). Same keys as forward(), leading dim B*T."""
        assert frames_hr.dim() == 5, (
            f"forward_video expects (B, T, 3, S, S), got "
            f"{tuple(frames_hr.shape)}"
        )
        B, T, C, H, W = frames_hr.shape
        return self.forward(frames_hr.reshape(B * T, C, H, W))

    @torch.no_grad()
    def forward_video_general(
        self,
        frames_hr: torch.Tensor,
        refpoint_w: torch.Tensor,
        query_feat_w: torch.Tensor,
        min_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """General-sampling variant: flatten video frames, then forward_general."""
        assert frames_hr.dim() == 5
        B, T, C, H, W = frames_hr.shape
        return self.forward_general(
            frames_hr.reshape(B * T, C, H, W),
            refpoint_w, query_feat_w,
            min_weight=min_weight,
        )
