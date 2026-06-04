"""Dump per-frame predictions + per-IoU AP sweep to JSON for the figure scripts.

kind is "tempo" (rfdetr_video) or "stqd" (stqd_det, from the worktree).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.resolve()
# append (not insert) so main-repo rfdetr_video wins; worktree is only for stqd_det
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".claude/worktrees/stqd-det"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

DATASETS = [
    ("dataset2_split_test",
     ROOT / "data/dataset2_split/test/images",
     ROOT / "data/dataset2_split/test/labels"),
    ("cadica_50plus_new",
     ROOT / "data/cadica_50plus_new/images",
     ROOT / "data/cadica_50plus_new/labels"),
]


# utilities 
def yolo_to_xyxy(lbl_path: Path, w: int, h: int) -> np.ndarray:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    lab = np.loadtxt(lbl_path, dtype=np.float32).reshape(-1, 5)
    if lab.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
    return np.column_stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]).astype(np.float32)


def to_tensor(img: np.ndarray, size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return (torch.from_numpy(np.stack([img] * 3, axis=0)) - mean) / std


# per-model predict helpers 
class TempoPredictor:
    def __init__(self, run_dir: Path):
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
        model = VideoRFDETR(cfg).to(DEVICE)
        ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        msg = model.load_state_dict(sd, strict=False)
        print(f"  TempoPredictor loaded  missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        model.eval()
        self.model = model
        self.T = cfg.T
        self.img_size = cfg.img_size
        self.mean = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

    @torch.no_grad()
    def predict_centre(self, frames: torch.Tensor, orig_w: int, orig_h: int):
        # frames: (T, 3, H, W) at img_size
        images = frames.unsqueeze(0).to(DEVICE)  # (1, T, 3, H, W)
        with torch.amp.autocast("cuda", enabled=True):
            out = self.model(images, query_mode="student")
        logits = out["pred_logits"][0, self.T // 2]  # (Q, K)
        boxes = out["pred_boxes"][0, self.T // 2]    # (Q, 4) cxcywh normalised
        scores = logits.sigmoid().amax(dim=-1)
        # cxcywh -> xyxy in image pixels
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = (cx - bw / 2) * orig_w
        y1 = (cy - bh / 2) * orig_h
        x2 = (cx + bw / 2) * orig_w
        y2 = (cy + bh / 2) * orig_h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy().astype(np.float32)
        return boxes_xyxy, scores.cpu().numpy().astype(np.float32)


class STQDPredictor:
    def __init__(self, run_dir: Path):
        from stqd_det.evaluate import build_model_from_run, predict_centre
        self._predict_centre = predict_centre
        self.model = build_model_from_run(run_dir, device=DEVICE)
        self.cfg = self.model.cfg
        self.T = self.cfg.T
        self.img_size = self.cfg.img_size
        self.mean = torch.tensor(self.cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(self.cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)
        print(f"  STQDPredictor loaded T={self.T} img_size={self.img_size}")

    @torch.no_grad()
    def predict_centre(self, frames: torch.Tensor, orig_w: int, orig_h: int):
        boxes_xyxy, scores = self._predict_centre(
            self.model, frames, orig_w=orig_w, orig_h=orig_h,
            centre=self.T // 2, amp=True,
        )
        if hasattr(boxes_xyxy, "detach"):
            boxes_xyxy = boxes_xyxy.detach().cpu().numpy()
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        return boxes_xyxy.astype(np.float32), scores.astype(np.float32)


# IoU + AP 
def iou_matrix(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_area = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    b_area = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = a_area[:, None] + b_area[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def compute_ap(recalls, precisions):
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map(all_detections, all_ground_truths, iou_threshold=0.5):
    # all_detections: list of {boxes,scores}; all_ground_truths: list of (N,4) arrays
    if sum(len(g) for g in all_ground_truths) == 0:
        return 0.0
    scores_all, tp_all, fp_all = [], [], []
    total_gt = 0
    for det, gts in zip(all_detections, all_ground_truths):
        boxes = det["boxes"]
        scores = det["scores"]
        total_gt += len(gts)
        order = np.argsort(-scores)
        boxes_ord = boxes[order]
        scores_ord = scores[order]
        if len(gts) == 0:
            tp = np.zeros(len(boxes_ord))
            fp = np.ones(len(boxes_ord))
        else:
            ious = iou_matrix(boxes_ord, gts)
            matched = np.zeros(len(gts), dtype=bool)
            tp = np.zeros(len(boxes_ord))
            fp = np.zeros(len(boxes_ord))
            for i in range(len(boxes_ord)):
                if ious.shape[1] == 0:
                    fp[i] = 1
                    continue
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= iou_threshold and not matched[j]:
                    tp[i] = 1
                    matched[j] = True
                else:
                    fp[i] = 1
        scores_all.append(scores_ord)
        tp_all.append(tp)
        fp_all.append(fp)
    scores = np.concatenate(scores_all) if scores_all else np.array([])
    tp = np.concatenate(tp_all) if tp_all else np.array([])
    fp = np.concatenate(fp_all) if fp_all else np.array([])
    if len(scores) == 0 or total_gt == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.cumsum(tp[order])
    fp = np.cumsum(fp[order])
    recall = tp / max(total_gt, 1)
    precision = tp / np.maximum(tp + fp, 1e-6)
    return compute_ap(recall, precision)


# main 
def run_one(predictor, ds_name: str, img_dir: Path, lbl_dir: Path, out_path: Path):
    from rfdetr_video.sequence_dataset import build_sequence_index
    sequences = build_sequence_index(img_dir)
    centre = predictor.T // 2

    all_dets, all_gts = [], []
    frames_out = []

    t0 = time.time()
    for vi, (pid, sid, paths) in enumerate(sequences):
        n = len(paths)
        if n == 0:
            continue
        if n < predictor.T:
            windows = [list(paths) + [paths[-1]] * (predictor.T - n)]
        else:
            windows = [paths[s:s + predictor.T] for s in range(n - predictor.T + 1)]

        for wi, win in enumerate(windows):
            frames = []
            orig_h = orig_w = None
            for p in win:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(p)
                if orig_h is None:
                    orig_h, orig_w = img.shape[:2]
                frames.append(to_tensor(img, predictor.img_size, predictor.mean, predictor.std))
            frames_t = torch.stack(frames, dim=0)
            boxes, scores = predictor.predict_centre(frames_t, orig_w, orig_h)
            centre_path = win[centre]
            gt = yolo_to_xyxy(lbl_dir / (centre_path.stem + ".txt"), orig_w, orig_h)

            all_dets.append({"boxes": boxes, "scores": scores})
            all_gts.append(gt)

            frames_out.append({
                "video": f"{pid}_v{sid}",
                "win_idx": wi,
                "centre_path": str(centre_path),
                "orig_w": int(orig_w),
                "orig_h": int(orig_h),
                "boxes": boxes.tolist(),
                "scores": scores.tolist(),
                "gt_boxes": gt.tolist(),
            })
        if (vi + 1) % 20 == 0 or (vi + 1) == len(sequences):
            print(f"     videos {vi + 1}/{len(sequences)}  windows={len(frames_out)}  elapsed={time.time()-t0:.0f}s", flush=True)

    # dump frames before AP so a crash in evaluate_map doesn't lose inference
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"dataset": ds_name, "iou_sweep": {}, "frames": frames_out}, f)
    print(f"  -> wrote frames-only {out_path}  ({len(frames_out)} frames)", flush=True)

    iou_sweep = {f"{t:.2f}": evaluate_map(all_dets, all_gts, t) for t in IOU_THRESHOLDS}
    print("  iou sweep:", {k: round(v, 4) for k, v in iou_sweep.items()}, flush=True)

    with open(out_path, "w") as f:
        json.dump({"dataset": ds_name, "iou_sweep": iou_sweep, "frames": frames_out}, f)
    print(f"  -> wrote final {out_path}", flush=True)


def main(kind: str, run_dir: Path, out_key: str):
    out_dir = ROOT / "thesis" / "figdata"
    if kind == "tempo":
        predictor = TempoPredictor(run_dir)
    elif kind == "stqd":
        predictor = STQDPredictor(run_dir)
    else:
        raise ValueError(kind)
    for ds_name, img_dir, lbl_dir in DATASETS:
        print(f"\n## {out_key} / {ds_name}")
        run_one(predictor, ds_name, img_dir, lbl_dir, out_dir / f"{out_key}__{ds_name}.json")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python _dump_preds_for_figs.py <kind=tempo|stqd> <run_dir> <out_key>")
        sys.exit(1)
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3])
