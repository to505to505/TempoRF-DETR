"""Centralised configuration for STQD-Det."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Paths
    data_root: Path = Path("data/dataset2_split")
    output_dir: Path = Path("rfdetr_video/runs")

    # Image / sequence
    img_size: int = 512
    T: int = 5

    # Detection
    num_classes: int = 1
    num_proposals: int = 300            # paper inherits from DiffusionDet; same as RF-DETR baseline
    pixel_mean: tuple = (0.485, 0.456, 0.406)
    pixel_std: tuple = (0.229, 0.224, 0.225)

    # Backbone + FPN
    fpn_out_channels: int = 256
    backbone_frozen_bn: bool = True     # standard for detection: freeze BN stats during fine-tuning

    # Diffusion (Quantum Poisson noise box)
    diffusion_T_steps: int = 1000       # DDPM-style; only marginal q(B_t|B_0) is sampled
    diffusion_sampling_steps: int = 1   # DiffusionDet default = 1 inference step
    sigma_scale: float = 2.0            # box-noise scale at t=T (signed-normalised coords in [-2, 2])
    sequential_alpha: float = 0.01      # frame-to-frame perturbation scale for SQNB Eq. 4

    # GFE (Global Feature Enhancement)
    gfe_heads: int = 8
    gfe_dropout: float = 0.0
    gfe_dc_kernels: int = 4             # K parallel dynamic-conv kernels

    # Decoder (DiffusionDet RCNNHead cascade)
    decoder_num_heads: int = 6
    decoder_dim_feedforward: int = 2048
    decoder_dropout: float = 0.0
    decoder_roi_size: int = 7
    decoder_dynamic_dim: int = 64
    decoder_dynamic_num: int = 2

    # STFS
    stfs_enabled: bool = True
    stfs_conf_thresh: float = 0.5       # paper: stage-1 boxes kept above 0.5
    stfs_iou_match: float = 1e-4        # primary cost weight on (1 - IoU); secondary cost weight on L1
    stfs_alpha_pad: float = 2.0         # H-FP RoI padding coefficient alpha (paper Eq. 19)
    consistency_weight: float = 1.0
    consistency_beta: float = 1.0       # eps in L_num denominator (paper Eq. 21 - paper writes beta, prevent div-by-zero)

    # Loss weights (DiffusionDet defaults)
    cls_weight: float = 2.0
    l1_weight: float = 5.0
    giou_weight: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Training
    epochs: int = 50
    batch_size: int = 1                 # paper batch = 1 sequence (xT frames)
    grad_accum_steps: int = 4           # effective batch = 4 (matches RF-DETR baseline)
    num_workers: int = 4
    lr: float = 2.5e-5                  # paper hyperparameter
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"
    warmup_iters: int = 500
    lr_step_milestones: tuple = (30, 40)
    lr_gamma: float = 0.1

    # EMA + checkpoint selection + early stopping (mirrors rfdetr_video)
    ema_enabled: bool = True
    ema_decay: float = 0.999
    selection_smooth_k: int = 3
    selection_weights: tuple = (0.5, 0.3, 0.2)
    early_stop_enabled: bool = True
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.0

    # Logging
    wandb_project: str = "stqd-det"
    wandb_enabled: bool = True
    run_name: Optional[str] = None
    log_interval: int = 50
    eval_interval: int = 2

    # Misc
    seed: int = 42
    amp: bool = True
