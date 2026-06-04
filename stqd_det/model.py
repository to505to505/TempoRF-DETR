"""STQDDet forward pass: frames -> ResNet-50/FPN -> GFE on P5 -> diffusion decoder.

Training seeds noised proposals from GT; inference samples around the image
centre. Decoder outputs are signed-normalised cxcywh (see stqd_det.noise).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .backbone import build_backbone
from .config import Config
from .decoder import StenosisDecoder
from .gfe import GFE
from .noise import (
    cosine_alpha_bar,
    prepare_inference_init,
    prepare_training_noise,
)
from .stfs import STFS


class STQDDet(nn.Module):
    """backbone -> GFE -> diffusion decoder (+ optional STFS stage 2).

    frames: (B, T, 3, H, W) ImageNet-normalised.
    targets_per_frame: list[B] of list[T] of {"boxes","labels"}, cxcywh-norm,
    only needed for training. Returns the decoder output dict.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(
            out_channels=cfg.fpn_out_channels,
            pretrained=True,
            frozen_bn=cfg.backbone_frozen_bn,
        )
        self.gfe = GFE(
            channels=cfg.fpn_out_channels,
            heads=cfg.gfe_heads,
            dc_kernels=cfg.gfe_dc_kernels,
            dropout=cfg.gfe_dropout,
        )
        self.decoder = StenosisDecoder(
            d_model=cfg.fpn_out_channels,
            num_classes=cfg.num_classes,
            num_heads_cascade=cfg.decoder_num_heads,
            roi_size=cfg.decoder_roi_size,
            nhead=cfg.gfe_heads,
            dim_feedforward=cfg.decoder_dim_feedforward,
            dropout=cfg.decoder_dropout,
            dim_dynamic=cfg.decoder_dynamic_dim,
            num_dynamic=cfg.decoder_dynamic_num,
            img_size=cfg.img_size,
            sigma=cfg.sigma_scale,
            fpn_levels=self.backbone.level_names,
            fpn_strides=self.backbone.level_strides,
        )
        # buffer so the schedule moves with the model on .to(device)
        self.register_buffer(
            "alpha_bar", cosine_alpha_bar(cfg.diffusion_T_steps), persistent=False
        )

        self.stfs = (
            STFS(
                cfg,
                fpn_levels=self.backbone.level_names,
                fpn_strides=self.backbone.level_strides,
            )
            if cfg.stfs_enabled
            else None
        )

    # feature extraction 

    def _extract_features(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """frames (B,T,3,H,W) -> FPN dict, P5 enhanced by GFE. Leading dim is B*T."""
        B, T, C_in, H, W = frames.shape
        BT = B * T
        flat = frames.view(BT, C_in, H, W)
        pyramid = self.backbone(flat)                              # OrderedDict
        # GFE wants (B, T, C, h, w) for the windowed MHA
        P5 = pyramid["P5"]
        C = P5.shape[1]
        h, w = P5.shape[2], P5.shape[3]
        P5_BT = P5.view(B, T, C, h, w)
        P5_gfe = self.gfe(P5_BT).view(BT, C, h, w)
        pyramid["P5"] = P5_gfe
        return pyramid

    # forward 

    def forward(
        self,
        frames: torch.Tensor,
        targets_per_frame: Optional[List[List[Dict[str, torch.Tensor]]]] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T = frames.shape[:2]
        BT = B * T
        device = frames.device
        pyramid = self._extract_features(frames)

        if self.training and targets_per_frame is None:
            raise ValueError("targets_per_frame required during training")
        if targets_per_frame is not None:
            boxes_signed, t_chosen = prepare_training_noise(
                targets_per_frame,
                num_proposals=self.cfg.num_proposals,
                sigma=self.cfg.sigma_scale,
                alpha_bar=self.alpha_bar,
                sequential_alpha=self.cfg.sequential_alpha,
                device=device,
                t=t,
            )
        else:
            # inference: noise around centre at the max timestep
            boxes_signed = prepare_inference_init(
                batch_size=B,
                num_frames=T,
                num_proposals=self.cfg.num_proposals,
                sigma=self.cfg.sigma_scale,
                sequential_alpha=self.cfg.sequential_alpha,
                device=device,
            )
            t_chosen = torch.full(
                (B,), self.cfg.diffusion_T_steps - 1, dtype=torch.long, device=device
            )

        boxes_flat = boxes_signed.view(BT, self.cfg.num_proposals, 4)
        t_flat = t_chosen.unsqueeze(1).expand(B, T).reshape(BT).contiguous()

        # Stage 1 
        stage1 = self.decoder(pyramid, boxes_flat, t_flat)
        out: Dict[str, torch.Tensor] = {
            "stage1": stage1,
            "t": t_chosen,
            "B": B,
            "T": T,
        }

        # STFS -> Stage 2 
        if self.stfs is not None:
            stfs_out = self.stfs(
                stage1_logits=stage1["final_logits"],
                stage1_boxes_signed=stage1["final_boxes"].detach(),
                features=pyramid,
                B=B,
                T=T,
            )
            stage2_init = stfs_out["stage2_boxes_signed"]
            stage2 = self.decoder(pyramid, stage2_init, t_flat)
            out["stage2"] = stage2
            out["stfs"] = stfs_out
            out["final_logits"] = stage2["final_logits"]
            out["final_boxes"] = stage2["final_boxes"]
        else:
            out["final_logits"] = stage1["final_logits"]
            out["final_boxes"] = stage1["final_boxes"]
        # back-compat: tests/evaluator read these stage-1 keys at the top level
        out["pred_logits_stages"] = stage1["pred_logits_stages"]
        out["pred_boxes_stages"] = stage1["pred_boxes_stages"]
        out["queries_final"] = stage1["queries_final"]
        return out
