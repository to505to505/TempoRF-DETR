"""Distillation utilities for Video RF-DETR."""

from rfdetr_video.distill.crrcd import CRRCDLoss
from rfdetr_video.distill.losses import distillation_loss

from .teacher import VideoFrozenRFDETRTeacher

__all__ = [
    "CRRCDLoss",
    "distillation_loss",
    "VideoFrozenRFDETRTeacher",
]
