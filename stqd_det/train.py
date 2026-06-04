"""STQD-Det training entry-point.

CLI mirrors rfdetr_video.train so the run-management tooling treats both
models the same (same output files: best.pth, last.pth, train.csv, etc).
"""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from rfdetr_video.ema import ModelEMA
from rfdetr_video.schedule import build_scheduler
from rfdetr_video.selection import (
    EarlyStopper,
    SmoothedTracker,
    composite_selection_score,
)

from .config import Config
from .dataset import get_dataloader
from .losses import SetCriterion
from .model import STQDDet


os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")


# helpers 


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_lr(optimizer, step: int, warmup_iters: int, base_lrs: List[float]) -> None:
    if step >= warmup_iters:
        return
    alpha = step / max(warmup_iters, 1)
    for pg, base in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base * alpha


def save_train_csv(run_dir: Path, history: list) -> None:
    if not history:
        return
    fieldnames = sorted({k for row in history for k in row.keys()})
    with open(run_dir / "train.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_best_txt(run_dir: Path, best_metrics: dict, best_epoch: int, cfg: Config) -> None:
    with open(run_dir / "best.txt", "w") as f:
        f.write("STQD-Det Best Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Best mAP30:    {best_metrics.get('AP@0.3', 0):.5f}\n")
        f.write(f"Best mAP50:    {best_metrics.get('AP@0.5', 0):.5f}\n")
        f.write(f"Best epoch:    {best_epoch}\n")
        f.write("\n--- Metrics ---\n")
        for k, v in sorted(best_metrics.items()):
            f.write(f"{k:35s}  {v}\n")
        f.write("\n--- Config ---\n")
        for k, v in sorted(asdict(cfg).items()):
            f.write(f"{str(k):35s}  {v}\n")


def serialise_config(cfg: Config) -> dict:
    out = {}
    for k, v in asdict(cfg).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


# argparse 


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train STQD-Det")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--T", type=int, default=None, dest="T")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--num-proposals", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--lr-schedule", type=str, default=None, choices=["cosine", "multistep"])
    p.add_argument("--warmup-iters", type=int, default=None)
    p.add_argument("--diffusion-T-steps", type=int, default=None)
    p.add_argument("--consistency-weight", type=float, default=None)
    p.add_argument("--no-stfs", action="store_true")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Run a tiny 1-epoch + 5-batch training pass (for sanity).")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip per-epoch eval; useful while evaluate.py is incomplete.")
    return p


def cfg_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    # Map argparse names -> cfg fields.
    overrides: Dict[str, object] = {}
    for k in (
        "T", "img_size", "num_proposals", "epochs", "batch_size",
        "grad_accum_steps", "num_workers", "lr", "weight_decay",
        "lr_schedule", "warmup_iters", "diffusion_T_steps",
        "consistency_weight", "ema_decay", "early_stop_patience",
        "eval_interval", "seed", "run_name",
    ):
        v = getattr(args, k, None)
        if v is not None:
            overrides[k] = v
    if args.data_root is not None:
        overrides["data_root"] = args.data_root
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.no_stfs:
        overrides["stfs_enabled"] = False
    if args.no_ema:
        overrides["ema_enabled"] = False
    if args.no_early_stop:
        overrides["early_stop_enabled"] = False
    if args.no_amp:
        overrides["amp"] = False
    if args.no_wandb:
        overrides["wandb_enabled"] = False
    return replace(cfg, **overrides)


# train loop 


def train_one_epoch(
    model: STQDDet,
    loader,
    criterion: SetCriterion,
    optimizer,
    scaler,
    cfg: Config,
    device,
    epoch: int,
    global_step: int,
    base_lrs: List[float],
    ema: ModelEMA,
    smoke: bool = False,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_cls = total_l1 = total_giou = total_num = 0.0
    n_steps = 0
    t0 = time.time()

    accum = max(cfg.grad_accum_steps, 1)
    for it, batch in enumerate(loader):
        if smoke and it >= 5:
            break
        warmup_lr(optimizer, global_step, cfg.warmup_iters, base_lrs)

        frames = batch["frames"].to(device, non_blocking=True)
        targets = batch["targets"]                              # python list, kept CPU

        with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
            out = model(frames, targets_per_frame=targets)
            losses = criterion.compute_total_loss(out, targets)
            loss = losses["loss_total"] / accum

        if cfg.amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (it + 1) % accum == 0:
            if cfg.amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        total_loss += float(losses["loss_total"].detach())
        for sink, key in (
            ("total_cls", "stage1/loss_cls"),
            ("total_l1", "stage1/loss_l1"),
            ("total_giou", "stage1/loss_giou"),
        ):
            val = losses.get(key, losses.get(key.split("/")[-1], 0.0))
            if isinstance(val, torch.Tensor):
                val = float(val.detach())
            else:
                val = float(val)
            if sink == "total_cls":
                total_cls += val
            elif sink == "total_l1":
                total_l1 += val
            else:
                total_giou += val
        ln = losses.get("loss_num", 0.0)
        total_num += float(ln.detach()) if isinstance(ln, torch.Tensor) else float(ln)
        n_steps += 1
        global_step += 1

        if it % cfg.log_interval == 0:
            print(
                f"  ep{epoch:03d} step {it:4d} loss={float(losses['loss_total'].detach()):.3f} "
                f"cls={total_cls/n_steps:.3f} l1={total_l1/n_steps:.3f} "
                f"giou={total_giou/n_steps:.3f} num={total_num/n_steps:.4f}",
                flush=True,
            )

    elapsed = time.time() - t0
    avg = max(n_steps, 1)
    return {
        "train_loss": total_loss / avg,
        "loss_cls": total_cls / avg,
        "loss_l1": total_l1 / avg,
        "loss_giou": total_giou / avg,
        "loss_num": total_num / avg,
        "epoch_seconds": elapsed,
        "global_step": global_step,
    }


def evaluate_loader(model, _loader_unused, cfg, device) -> Dict[str, float]:
    """Per-epoch eval: micro-pooled AP30/AP50/F1 on valid.

    Same centre-frame path as _eval_stfs_ablations.py so numbers match.
    _loader_unused is kept for API parity with the old loss-only stub.
    """
    from .evaluate import evaluate_on_split

    img_dir = cfg.data_root / "valid" / "images"
    lbl_dir = cfg.data_root / "valid" / "labels"
    if not img_dir.is_dir():
        # no valid split -> return zeros so the loop keeps going
        return {"val_loss": 0.0, "AP@0.3": 0.0, "AP@0.5": 0.0, "F1": 0.0,
                "precision": 0.0, "recall": 0.0}
    model.eval()
    metrics = evaluate_on_split(model, img_dir, lbl_dir)
    metrics["val_loss"] = 0.0  # train loop still expects this CSV column
    return metrics


# main 


def main() -> None:
    args = build_parser().parse_args()
    cfg = cfg_from_args(args)
    set_seed(cfg.seed)

    run_dir = cfg.output_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(serialise_config(cfg), f, indent=2)
    print(f"[stqd_det] writing run to {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataloader("train", cfg)
    val_loader = (
        None if args.skip_eval else get_dataloader("valid", cfg, shuffle=False, drop_last=False)
    )

    model = STQDDet(cfg).to(device)
    criterion = SetCriterion(num_classes=cfg.num_classes, cfg=cfg).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema_enabled else None

    history: List[dict] = []
    best_score = float("-inf")
    best_metrics: Dict[str, float] = {}
    best_epoch = 0
    smoothed = SmoothedTracker(cfg.selection_smooth_k)
    early = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta) if cfg.early_stop_enabled else None

    global_step = 0
    epochs = 1 if args.smoke else cfg.epochs
    for epoch in range(1, epochs + 1):
        print(f"[stqd_det] === epoch {epoch}/{epochs} ===", flush=True)
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg, device,
            epoch, global_step, base_lrs, ema, smoke=args.smoke,
        )
        global_step = train_stats["global_step"]
        scheduler.step()

        row = {"epoch": epoch, **{k: v for k, v in train_stats.items() if k != "global_step"}}

        if val_loader is not None and (epoch % cfg.eval_interval == 0 or args.smoke):
            if ema is not None:
                with ema.applied_to(model):
                    metrics = evaluate_loader(model, val_loader, cfg, device)
            else:
                metrics = evaluate_loader(model, val_loader, cfg, device)
            for k, v in metrics.items():
                row[k] = v
            score_raw = composite_selection_score(metrics, cfg.selection_weights)
            score = smoothed.add(score_raw)
            row["sel"] = score_raw
            row["sel_smoothed"] = score
            if score > best_score:
                best_score = score
                best_metrics = dict(metrics)
                best_epoch = epoch
                ckpt = {"state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics}
                if ema is not None:
                    ckpt["ema_state_dict"] = ema.shadow
                torch.save(ckpt, run_dir / "best.pth")
                write_best_txt(run_dir, best_metrics, best_epoch, cfg)
            if early is not None and early.update(score):
                print(f"[stqd_det] early stop after epoch {epoch}", flush=True)
                history.append(row)
                break

        history.append(row)
        # Always save last.pth + train.csv + history.json each epoch.
        torch.save({"state_dict": model.state_dict(), "epoch": epoch}, run_dir / "last.pth")
        save_train_csv(run_dir, history)
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"[stqd_det] training complete. best_epoch={best_epoch} best_score={best_score:.4f}")


if __name__ == "__main__":
    main()
