#!/usr/bin/env python3
"""Cluster bootstrap on per-video AP from ablation_results.json files.

Treats each video as a patient cluster, resamples with replacement.
Note: this is video-level macro AP, not the micro-pooled AP the paper reports
(we don't have per-frame preds saved to bootstrap the micro version).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
import sys as _sys
_sys.path.insert(0, str(ROOT))

# (display_name, ablation_results.json path)
MODELS = [
    ("TempoRF-DETR (full)",   ROOT / "rfdetr_video/runs/video_overfit_R1/ablation_results.json"),
    ("TempoRF-DETR ETF+KD",   ROOT / "rfdetr_video/runs/video_5_etf_consistency_distill/ablation_results.json"),
    ("TempoRF-DETR ETF only", ROOT / "rfdetr_video/runs/video_5_etf_consistency/ablation_results.json"),
    ("Post-Network Tuning",   ROOT / "rfdetr_video/runs/video_5_postnet_T5/ablation_results.json"),
    ("Prompt Tuning",         ROOT / "rfdetr_video/runs/video_5_prompt_T5/ablation_results.json"),
    ("STQD-Det",              ROOT / "rfdetr_video/runs/stqd_det_T5_dataset2/ablation_results.json"),
    ("PS-STT",                ROOT / "psstt/runs/psstt_20ep/ablation_results.json"),
    ("Stenosis-DetNet",       ROOT / "detnet/runs/detnet_v1_T5/ablation_results.json"),
]

DATASETS = ["dataset2_split_test", "cadica_50plus_new"]
DATASET_LABEL = {"dataset2_split_test": "RIPCID-test (in-dist)",
                 "cadica_50plus_new":    "CADICA (OOD)"}

METRIC = "AP30"
N_BOOT = 10_000
SEED = 12345


def load_per_video(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Return {dataset -> {'videos': [...], 'AP30': np.array, ...}}."""
    with open(path) as f:
        data = json.load(f)
    # rfdetr_video runs nest under ablations[0]; psstt/detnet have datasets at root
    if "ablations" in data:
        datasets_list = data["ablations"][0]["datasets"]
    else:
        datasets_list = data["datasets"]
    out = {}
    for ds in datasets_list:
        name = ds["dataset"]
        rows = ds["rows"]
        out[name] = {
            "videos": [r["video"] for r in rows],
            "AP30": np.array([r["AP30"] for r in rows], dtype=float),
            "AP50": np.array([r["AP50"] for r in rows], dtype=float),
            "F1":   np.array([r["F1"]   for r in rows], dtype=float),
            "P":    np.array([r["P"]    for r in rows], dtype=float),
            "R":    np.array([r["R"]    for r in rows], dtype=float),
        }
    return out


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator,
                   n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Cluster bootstrap of mean. Returns (point estimate, lo95, hi95)."""
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot_means, 2.5)), \
        float(np.percentile(boot_means, 97.5))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                          n_boot: int = N_BOOT) -> dict:
    """Paired cluster bootstrap of (a - b). Returns delta, CI, p-value."""
    assert len(a) == len(b), "paired vectors must have same length"
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    point = float((a - b).mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    # two-sided bootstrap p-value (Davison & Hinkley)
    if point >= 0:
        p = 2.0 * (np.sum(diffs <= 0) + 1) / (n_boot + 1)
    else:
        p = 2.0 * (np.sum(diffs >= 0) + 1) / (n_boot + 1)
    p = min(p, 1.0)
    return {"delta": point, "ci_lo": lo, "ci_hi": hi, "p": float(p)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"Cluster bootstrap on per-video {METRIC} | n_boot={N_BOOT}, seed={SEED}")
    print("=" * 88)

    # Load all
    loaded: dict[str, dict] = {}
    for name, path in MODELS:
        if not path.exists():
            print(f"  [skip] {name}: file not found at {path}")
            continue
        loaded[name] = load_per_video(path)

    # 1) Per-model AP30 with 95% CI on each dataset
    print(f"\n## Per-model mean {METRIC} with 95% cluster bootstrap CI (resampling videos)")
    print()
    for ds in DATASETS:
        print(f"### {DATASET_LABEL[ds]}")
        print(f"{'Model':30s}  {'n_vid':>5s}  {'mean ' + METRIC:>10s}  {'95% CI':>20s}")
        print("-" * 72)
        for name, _ in MODELS:
            if name not in loaded or ds not in loaded[name]:
                continue
            vals = loaded[name][ds][METRIC]
            mean, lo, hi = bootstrap_mean(vals, rng)
            print(f"{name:30s}  {len(vals):>5d}  {mean:>10.4f}  [{lo:.4f}, {hi:.4f}]")
        print()

    # 2) Paired bootstrap vs TempoRF-DETR (full)
    ref = "TempoRF-DETR (full)"
    print(f"\n## Paired bootstrap: {ref} vs each baseline (delta = ours - baseline)")
    print()
    for ds in DATASETS:
        print(f"### {DATASET_LABEL[ds]}")
        print(f"{'Comparison':40s}  {'delta ' + METRIC:>8s}  {'95% CI':>18s}  {'p (2-sided)':>12s}")
        print("-" * 86)
        ref_vids = loaded[ref][ds]["videos"]
        ref_ap = loaded[ref][ds][METRIC]
        for name, _ in MODELS:
            if name == ref or name not in loaded or ds not in loaded[name]:
                continue
            other_vids = loaded[name][ds]["videos"]
            other_ap = loaded[name][ds][METRIC]
            # align by video name, row order isn't guaranteed across runs
            ref_to_idx = {v: i for i, v in enumerate(other_vids)}
            try:
                aligned_other = np.array([other_ap[ref_to_idx[v]] for v in ref_vids])
            except KeyError as e:
                print(f"  [skip] {name}: video {e} not found")
                continue
            res = paired_bootstrap_diff(ref_ap, aligned_other, rng)
            sig = " *" if res["p"] < 0.05 else "  "
            print(f"vs {name:37s}  {res['delta']:+.4f}  [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  {res['p']:>12.4f}{sig}")
        print()


if __name__ == "__main__":
    main()
