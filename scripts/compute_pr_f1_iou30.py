"""Micro-pooled F1 / P / R at IoU=0.30 for the methods in the paper.

Two modes:
  * --from-dump  reads thesis/figdata/<key>__<ds>.json (TempoRF, STQD, Post-Net)
  * --infer      runs inference on a run dir (detnet | psstt | prompt | postnet | tempo)

Usage:
  python _pr_at_iou30.py --from-dump tempo_R1
  python _pr_at_iou30.py --from-dump stqd
  python _pr_at_iou30.py --from-dump postnet
  python _pr_at_iou30.py --infer detnet  detnet/runs/detnet_v1_T5
  python _pr_at_iou30.py --infer psstt   psstt/runs/psstt_20ep
  python _pr_at_iou30.py --infer prompt  rfdetr_video/runs/video_5_prompt_T5
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".claude/worktrees/stqd-det"))

from rfdetr_video.sequence_eval import f1_confidence_sweep
from rfdetr_video.sequence_dataset import build_sequence_index

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = [
    ("dataset2_split_test",
     ROOT / "data/dataset2_split/test/images",
     ROOT / "data/dataset2_split/test/labels"),
    ("cadica_50plus_new",
     ROOT / "data/cadica_50plus_new/images",
     ROOT / "data/cadica_50plus_new/labels"),
]
FIGDATA = ROOT / "thesis/figdata"


# shared helpers 
def yolo_to_xyxy(lbl_path, w, h):
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    lab = np.loadtxt(lbl_path, dtype=np.float32).reshape(-1, 5)
    if lab.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
    return np.column_stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]).astype(np.float32)


def to_imagenet(img, size, mean, std):
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return (torch.from_numpy(np.stack([img] * 3, axis=0)) - mean) / std


# from-dump mode 
def from_dump(key):
    for ds_name, _, _ in DATASETS:
        path = FIGDATA / f"{key}__{ds_name}.json"
        with open(path) as f:
            d = json.load(f)
        dets = [{"boxes": np.array(fr["boxes"], dtype=np.float32).reshape(-1, 4),
                 "scores": np.array(fr["scores"], dtype=np.float32)}
                for fr in d["frames"]]
        gts = [np.array(fr["gt_boxes"], dtype=np.float32).reshape(-1, 4) for fr in d["frames"]]
        f1, p, r, thr = f1_confidence_sweep(dets, gts, iou_threshold=0.30)
        print(f"{key:14s} | {ds_name:22s} | P@0.3={p:.4f}  R@0.3={r:.4f}  F1@0.3={f1:.4f}  thr={thr:.3f}")


# infer mode 
class DetNetPSSTTPredictor:
    """Both detnet and psstt expose VideoFasterRCNN.forward returning {'centre': [...]}."""
    def __init__(self, kind, run_dir):
        if kind == "detnet":
            from detnet.config import Config
            from detnet.model import VideoFasterRCNN
        elif kind == "psstt":
            from psstt.config import Config
            from psstt.model import VideoFasterRCNN
        else:
            raise ValueError(kind)
        with open(run_dir / "config.json") as f:
            raw = json.load(f)
        cfg = Config()
        for k, v in raw.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, Path(v) if isinstance(getattr(cfg, k), Path) and v else v)
                except Exception:
                    pass
        cfg.pretrained_coco = False
        self.cfg = cfg
        self.model = VideoFasterRCNN(cfg).to(DEVICE)
        ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        msg = self.model.load_state_dict(sd, strict=False)
        print(f"  {kind} loaded missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        self.model.eval()
        self.T = cfg.T
        self.img_size = cfg.img_size
        self.mean = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

    @torch.no_grad()
    def predict_centre(self, frames, orig_w, orig_h):
        images = frames.unsqueeze(0).to(DEVICE)
        with torch.amp.autocast("cuda", enabled=True):
            out = self.model(images, targets=None)
        centre = out["centre"][0]
        boxes = centre["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = centre["scores"].detach().cpu().numpy().astype(np.float32)
        sx = orig_w / self.img_size
        sy = orig_h / self.img_size
        if boxes.shape[0] > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        return boxes, scores


class TempoVideoRFDETRPredictor:
    """For postnet, prompt - VideoRFDETR with adapt_mode."""
    def __init__(self, run_dir):
        from rfdetr_video.config import Config, apply_adapt_mode
        from rfdetr_video.model import VideoRFDETR
        with open(run_dir / "config.json") as f:
            raw = json.load(f)
        cfg = Config()
        for k, v in raw.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, Path(v) if isinstance(getattr(cfg, k), Path) and v else v)
                except Exception:
                    pass
        cfg = apply_adapt_mode(cfg)
        self.cfg = cfg
        self.model = VideoRFDETR(cfg).to(DEVICE)
        ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        msg = self.model.load_state_dict(sd, strict=False)
        print(f"  videoRFDETR loaded missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        self.model.eval()
        self.T = cfg.T
        self.img_size = cfg.img_size
        self.mean = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

    @torch.no_grad()
    def predict_centre(self, frames, orig_w, orig_h):
        images = frames.unsqueeze(0).to(DEVICE)
        with torch.amp.autocast("cuda", enabled=True):
            out = self.model(images, query_mode="student")
        logits = out["pred_logits"][0, self.T // 2]
        boxes = out["pred_boxes"][0, self.T // 2]
        scores = logits.sigmoid().amax(dim=-1)
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = (cx - bw / 2) * orig_w
        y1 = (cy - bh / 2) * orig_h
        x2 = (cx + bw / 2) * orig_w
        y2 = (cy + bh / 2) * orig_h
        return torch.stack([x1, y1, x2, y2], -1).cpu().numpy().astype(np.float32), scores.cpu().numpy().astype(np.float32)


def infer(kind, run_dir):
    run_dir = Path(run_dir)
    if kind in ("detnet", "psstt"):
        predictor = DetNetPSSTTPredictor(kind, run_dir)
    elif kind in ("prompt", "postnet", "tempo"):
        predictor = TempoVideoRFDETRPredictor(run_dir)
    else:
        raise ValueError(kind)

    for ds_name, img_dir, lbl_dir in DATASETS:
        sequences = build_sequence_index(img_dir)
        centre = predictor.T // 2
        dets, gts = [], []
        t0 = time.time()
        for vi, (pid, sid, paths) in enumerate(sequences):
            n = len(paths)
            if n == 0:
                continue
            if n < predictor.T:
                windows = [list(paths) + [paths[-1]] * (predictor.T - n)]
            else:
                windows = [paths[s:s + predictor.T] for s in range(n - predictor.T + 1)]
            for win in windows:
                frames = []
                orig_h = orig_w = None
                for p in win:
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if orig_h is None:
                        orig_h, orig_w = img.shape[:2]
                    frames.append(to_imagenet(img, predictor.img_size, predictor.mean, predictor.std))
                frames_t = torch.stack(frames, dim=0)
                boxes, scores = predictor.predict_centre(frames_t, orig_w, orig_h)
                centre_path = win[centre]
                gt = yolo_to_xyxy(lbl_dir / (centre_path.stem + ".txt"), orig_w, orig_h)
                dets.append({"boxes": boxes, "scores": scores})
                gts.append(gt)
            if (vi + 1) % 20 == 0 or (vi + 1) == len(sequences):
                print(f"     videos {vi+1}/{len(sequences)}  windows={len(dets)}  elapsed={time.time()-t0:.0f}s", flush=True)
        f1, p, r, thr = f1_confidence_sweep(dets, gts, iou_threshold=0.30)
        print(f"{kind:14s} | {ds_name:22s} | P@0.3={p:.4f}  R@0.3={r:.4f}  F1@0.3={f1:.4f}  thr={thr:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dump", type=str, default=None,
                    help="dump key (tempo_R1 | stqd | postnet)")
    ap.add_argument("--infer", nargs=2, metavar=("kind", "run_dir"),
                    help="kind=detnet|psstt|prompt|postnet|tempo; run_dir path relative to submission root")
    args = ap.parse_args()
    if args.from_dump:
        from_dump(args.from_dump)
    elif args.infer:
        kind, run_dir = args.infer
        infer(kind, ROOT / run_dir)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
