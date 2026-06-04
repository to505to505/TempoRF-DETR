"""STQD-Det evaluation adapter for the shared headline-metric pipeline.

Boxes come back as absolute-pixel xyxy at the original image resolution
(same convention as rfdetr_video.evaluate.PostProcess).
"""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch

from .config import Config
from .model import STQDDet
from .noise import from_signed


@torch.no_grad()
def predict_centre(
    model: STQDDet,
    frames: torch.Tensor,
    orig_w: int,
    orig_h: int,
    centre: int = None,
    amp: bool = True,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Run on a single (T, 3, H, W) window; return centre-frame boxes/scores.

    centre defaults to T // 2. boxes_xyxy is (N, 4) pixel xyxy at the
    original resolution; scores is (N,) sigmoid confidences.
    """
    if frames.dim() != 4 or frames.shape[1] != 3:
        raise ValueError(
            f"frames must be (T, 3, H, W); got {tuple(frames.shape)}"
        )
    cfg: Config = model.cfg
    if centre is None:
        centre = frames.shape[0] // 2

    device = next(model.parameters()).device
    images = frames.unsqueeze(0).to(device, non_blocking=True)
    use_amp = amp and device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_amp):
        out = model(images)

    # final_* are (BT=1*T, N, *); take the centre frame.
    final_logits = out["final_logits"][centre]
    final_boxes = out["final_boxes"][centre]

    scores = final_logits.sigmoid().max(dim=-1).values

    # signed-normalised cxcywh -> [0,1] cxcywh -> pixel xyxy
    boxes_norm = from_signed(final_boxes, cfg.sigma_scale)
    cx, cy, w, h = boxes_norm.unbind(-1)
    x1 = (cx - w / 2).clamp(0, 1) * orig_w
    y1 = (cy - h / 2).clamp(0, 1) * orig_h
    x2 = (cx + w / 2).clamp(0, 1) * orig_w
    y2 = (cy + h / 2).clamp(0, 1) * orig_h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes_xyxy, scores


_IOU_5095 = np.arange(0.5, 1.0, 0.05)


def _to_imagenet_tensor(img: np.ndarray, size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    arr = np.stack([img, img, img], axis=0)
    return (torch.from_numpy(arr) - mean) / std


def _yolo_xyxy_pixel(lbl_path: Path, w: int, h: int) -> np.ndarray:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    lab = np.loadtxt(lbl_path, dtype=np.float32).reshape(-1, 5)
    if lab.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
    return np.column_stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]).astype(np.float32)


@torch.no_grad()
def evaluate_on_split(model: STQDDet, img_dir: Path, lbl_dir: Path) -> Dict[str, float]:
    """MICRO pooled AP30/AP50/AP75/AP5095/F1/P/R for a single split.

    Uses the same centre-frame extraction as the headline-metric script
    so train-loop and ablation numbers stay comparable.
    """
    from rfdetr_video.sequence_dataset import build_sequence_index
    from rfdetr_video.sequence_eval import evaluate_map, f1_confidence_sweep

    cfg = model.cfg
    mean_t = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)
    centre = cfg.T // 2

    sequences = build_sequence_index(Path(img_dir))
    dets, gts = [], []
    for pid, sid, paths in sequences:
        n = len(paths)
        if n == 0:
            continue
        windows = (
            [list(paths) + [paths[-1]] * (cfg.T - n)]
            if n < cfg.T
            else [paths[s : s + cfg.T] for s in range(n - cfg.T + 1)]
        )
        for win in windows:
            frames, orig_w, orig_h = [], None, None
            for p in win:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(p)
                if orig_w is None:
                    orig_h, orig_w = img.shape[:2]
                frames.append(_to_imagenet_tensor(img, cfg.img_size, mean_t, std_t))
            frames_t = torch.stack(frames, dim=0)
            boxes_xyxy, scores = predict_centre(
                model, frames_t, orig_w=orig_w, orig_h=orig_h, centre=centre, amp=True,
            )
            gt = _yolo_xyxy_pixel(Path(lbl_dir) / (win[centre].stem + ".txt"), orig_w, orig_h)
            dets.append({"boxes": boxes_xyxy.cpu().numpy(), "scores": scores.cpu().numpy()})
            gts.append(gt)

    ap30 = evaluate_map(dets, gts, 0.3)
    ap50 = evaluate_map(dets, gts, 0.5)
    ap75 = evaluate_map(dets, gts, 0.75)
    ap5095 = float(np.mean([evaluate_map(dets, gts, t) for t in _IOU_5095]))
    f1, p, r, thr = f1_confidence_sweep(dets, gts)
    return {
        "AP@0.3": float(ap30), "AP@0.5": float(ap50), "AP@0.75": float(ap75),
        "AP@0.5:0.95": float(ap5095),
        "F1": float(f1), "precision": float(p), "recall": float(r),
        "best_conf": float(thr),
        "n_centre_frames": len(dets),
    }


def build_model_from_run(run_dir, device=None, checkpoint_path=None) -> STQDDet:
    """Rebuild STQDDet from run_dir/config.json and a checkpoint.

    checkpoint_path defaults to best.pth, falling back to last.pth.
    """
    import json
    from pathlib import Path

    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        raw = json.load(f)
    # JSON gives lists back; Config wants tuples
    for tup_key in ("pixel_mean", "pixel_std", "selection_weights",
                    "lr_step_milestones"):
        if tup_key in raw and isinstance(raw[tup_key], list):
            raw[tup_key] = tuple(raw[tup_key])
    if "data_root" in raw:
        raw["data_root"] = Path(raw["data_root"])
    if "output_dir" in raw:
        raw["output_dir"] = Path(raw["output_dir"])
    # drop unknown fields so an older config.json still loads
    cfg_fields = set(Config.__dataclass_fields__.keys())
    raw = {k: v for k, v in raw.items() if k in cfg_fields}
    cfg = Config(**raw)

    model = STQDDet(cfg)
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
    else:
        # fall back to last.pth for --skip-eval runs (no best.pth)
        ckpt_path = run_dir / "best.pth"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "last.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("ema_state_dict") or ckpt["state_dict"]
    # EMA shadow is params only, so merge over live weights to keep buffers
    if "ema_state_dict" in ckpt:
        merged = dict(model.state_dict())
        merged.update({k: v for k, v in state.items() if k in merged})
        model.load_state_dict(merged)
    else:
        model.load_state_dict(state)
    if device is not None:
        model.to(device)
    model.eval()
    return model
