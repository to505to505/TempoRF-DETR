"""Pick the best STQD-Det checkpoint by MICRO AP30 on the valid split.

Evaluates every epoch_*.pth in <run_dir>/ckpts, writes selection.json, and
symlinks the winner as best.pth. We select on valid; the final test number
comes from re-running _eval_stfs_ablations.py afterwards.

    python -m stqd_det.tools.select_best_checkpoint --run-dir <run>
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

from rfdetr_video.sequence_dataset import build_sequence_index
from rfdetr_video.sequence_eval import evaluate_map, f1_confidence_sweep

from stqd_det.evaluate import build_model_from_run, predict_centre


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_5095 = np.arange(0.5, 1.0, 0.05)


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


def eval_split(model, img_dir: Path, lbl_dir: Path) -> Dict[str, float]:
    """MICRO pooled metrics on a single split."""
    cfg = model.cfg
    mean_t = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)
    centre = cfg.T // 2

    sequences = build_sequence_index(img_dir)
    dets_all: List[Dict[str, np.ndarray]] = []
    gts_all: List[np.ndarray] = []

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
            frames, orig_w = [], None
            for p in win:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(p)
                if orig_w is None:
                    orig_h, orig_w = img.shape[:2]
                frames.append(_to_imagenet_tensor(img, cfg.img_size, mean_t, std_t))
            frames_t = torch.stack(frames, dim=0)
            boxes, scores = predict_centre(
                model, frames_t, orig_w=orig_w, orig_h=orig_h, centre=centre, amp=True,
            )
            centre_path = win[centre]
            gt = _yolo_xyxy_pixel(lbl_dir / (centre_path.stem + ".txt"), orig_w, orig_h)
            dets_all.append({"boxes": boxes.cpu().numpy(), "scores": scores.cpu().numpy()})
            gts_all.append(gt)

    ap30 = evaluate_map(dets_all, gts_all, 0.3)
    ap50 = evaluate_map(dets_all, gts_all, 0.5)
    ap75 = evaluate_map(dets_all, gts_all, 0.75)
    ap5095 = float(np.mean([evaluate_map(dets_all, gts_all, t) for t in IOU_5095]))
    f1, p, r, thr = f1_confidence_sweep(dets_all, gts_all)
    return {
        "AP30": ap30, "AP50": ap50, "AP75": ap75, "AP5095": ap5095,
        "F1": f1, "P": p, "R": r, "thr": thr, "n_centre_frames": len(dets_all),
    }


def main():
    ap = argparse.ArgumentParser(description="Select the best STQD-Det checkpoint on the valid split.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ckpts-dir", type=Path, default=None,
                    help="Defaults to <run_dir>/ckpts")
    ap.add_argument("--data-root", type=Path, default=Path("data/dataset2_split"),
                    help="Validation dataset root (split = 'valid').")
    ap.add_argument("--score-key", type=str, default="AP30",
                    help="Which MICRO metric to maximise.")
    args = ap.parse_args()

    run_dir = args.run_dir
    ckpts_dir = args.ckpts_dir or (run_dir / "ckpts")
    img_dir = args.data_root / "valid" / "images"
    lbl_dir = args.data_root / "valid" / "labels"
    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)

    epoch_pat = re.compile(r"^epoch_(\d+)\.pth$")
    ckpts = []
    for p in sorted(ckpts_dir.iterdir()):
        m = epoch_pat.match(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort()
    if not ckpts:
        raise RuntimeError(f"No epoch_*.pth under {ckpts_dir}")
    print(f"[select] {len(ckpts)} checkpoints found in {ckpts_dir}")

    results = []
    for epoch, ck in ckpts:
        t0 = time.time()
        model = build_model_from_run(run_dir, device=DEVICE, checkpoint_path=ck)
        with torch.no_grad():
            metrics = eval_split(model, img_dir, lbl_dir)
        dt = time.time() - t0
        metrics["epoch"] = epoch
        metrics["checkpoint"] = str(ck)
        metrics["eval_seconds"] = dt
        print(f"[select] epoch={epoch:3d}  AP30={metrics['AP30']:.4f}  "
              f"AP50={metrics['AP50']:.4f}  F1={metrics['F1']:.4f}  "
              f"({dt:.1f}s)")
        results.append(metrics)
        # free GPU mem before next ckpt
        del model
        torch.cuda.empty_cache()

    best = max(results, key=lambda m: m[args.score_key])
    print(f"\n[select] BEST: epoch={best['epoch']}  "
          f"{args.score_key}={best[args.score_key]:.4f}  "
          f"AP50={best['AP50']:.4f}  F1={best['F1']:.4f}")

    out = {
        "selection_metric": args.score_key,
        "valid_split": str(img_dir),
        "best": best,
        "per_epoch": results,
    }
    (run_dir / "selection.json").write_text(json.dumps(out, indent=2))
    print(f"[select] wrote -> {run_dir/'selection.json'}")

    # symlink so _eval_stfs_ablations.py finds it
    target = run_dir / "best.pth"
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(Path("ckpts") / f"epoch_{best['epoch']}.pth")
    print(f"[select] linked best.pth -> ckpts/epoch_{best['epoch']}.pth")


if __name__ == "__main__":
    main()
