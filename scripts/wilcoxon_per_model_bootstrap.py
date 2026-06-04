#!/usr/bin/env python3
"""Paired significance tests vs TempoRF-DETR (full): Wilcoxon, frame-weighted
cluster bootstrap, and length-stratified bootstrap. Sig threshold p < 0.05.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).parent.parent.resolve()
import sys as _sys
_sys.path.insert(0, str(ROOT))

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
REF = "TempoRF-DETR (full)"


def load_per_video(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    datasets_list = data["ablations"][0]["datasets"] if "ablations" in data else data["datasets"]
    out = {}
    for ds in datasets_list:
        rows = ds["rows"]
        out[ds["dataset"]] = {
            "videos":   [r["video"] for r in rows],
            "n_frames": np.array([r["n_frames"] for r in rows], dtype=float),
            "AP30":     np.array([r["AP30"]     for r in rows], dtype=float),
        }
    return out


def align(ref_vids, other_vids, other_vals):
    idx = {v: i for i, v in enumerate(other_vids)}
    return np.array([other_vals[idx[v]] for v in ref_vids])


# ---- Method 1: Wilcoxon signed-rank ----------------------------------------
def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    # zero_method='wilcox' drops zeros (standard); use 'pratt' to keep them
    if np.all(diff == 0):
        return {"delta_median": 0.0, "W": 0.0, "p": 1.0}
    res = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                       correction=False, method="auto")
    return {"delta_median": float(np.median(diff)),
            "W": float(res.statistic),
            "p": float(res.pvalue)}


# ---- Method 2: Frame-weighted cluster bootstrap ----------------------------
def frame_weighted_paired_bootstrap(
    a: np.ndarray, b: np.ndarray, w: np.ndarray,
    rng: np.random.Generator, n_boot: int = N_BOOT,
) -> dict:
    """Cluster bootstrap of weighted_mean(a) - weighted_mean(b), per-video AP
    weighted by n_frames (approximates micro AP)."""
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    aw = w * a
    bw = w * b
    num_a = aw[idx].sum(axis=1)
    num_b = bw[idx].sum(axis=1)
    den = w[idx].sum(axis=1)
    wmean_a = num_a / den
    wmean_b = num_b / den
    diffs = wmean_a - wmean_b
    point = float((aw.sum() - bw.sum()) / w.sum())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    if point >= 0:
        p = 2.0 * (np.sum(diffs <= 0) + 1) / (n_boot + 1)
    else:
        p = 2.0 * (np.sum(diffs >= 0) + 1) / (n_boot + 1)
    return {"delta": point, "ci_lo": lo, "ci_hi": hi, "p": min(p, 1.0)}


# ---- Method 3: Sequence-stratified cluster bootstrap -----------------------
def stratified_paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_frames: np.ndarray,
    rng: np.random.Generator, n_boot: int = N_BOOT, n_strata: int = 3,
) -> dict:
    """Bootstrap within strata defined by video-length quantiles."""
    quantiles = np.linspace(0, 1, n_strata + 1)[1:-1]
    thresholds = np.quantile(n_frames, quantiles)
    strata = np.digitize(n_frames, thresholds)  # 0..n_strata-1

    stratum_indices = [np.where(strata == s)[0] for s in range(n_strata)]
    n_total = len(a)

    diffs = np.empty(n_boot)
    for k in range(n_boot):
        sampled = []
        for inds in stratum_indices:
            if len(inds) == 0:
                continue
            sampled.append(rng.choice(inds, size=len(inds), replace=True))
        sampled = np.concatenate(sampled)
        diffs[k] = (a[sampled] - b[sampled]).mean()

    point = float((a - b).mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    if point >= 0:
        p = 2.0 * (np.sum(diffs <= 0) + 1) / (n_boot + 1)
    else:
        p = 2.0 * (np.sum(diffs >= 0) + 1) / (n_boot + 1)
    return {"delta": point, "ci_lo": lo, "ci_hi": hi, "p": min(p, 1.0)}


def fmt_sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "   "


def main() -> None:
    rng_w = np.random.default_rng(SEED)
    rng_s = np.random.default_rng(SEED + 1)

    print(f"Paired statistical tests vs {REF}, metric={METRIC}, n_boot={N_BOOT}")
    print("=" * 110)

    # Load all
    loaded = {}
    for name, path in MODELS:
        if path.exists():
            loaded[name] = load_per_video(path)

    for ds in DATASETS:
        print(f"\n## {DATASET_LABEL[ds]}")
        print(f"  n_videos = {len(loaded[REF][ds]['videos'])}, "
              f"n_frames range = [{int(loaded[REF][ds]['n_frames'].min())}, "
              f"{int(loaded[REF][ds]['n_frames'].max())}], "
              f"total frames = {int(loaded[REF][ds]['n_frames'].sum())}")
        print()
        # Header
        print(f"{'Comparison':32s}  | {'Wilcoxon':>20s} | {'Frame-weighted boot':>40s} | {'Stratified boot':>40s}")
        print(f"{'':32s}  | {'deltamed':>8s} {'p':>8s}    | {'delta':>6s}  {'95% CI':>20s} {'p':>8s}    | {'delta':>6s}  {'95% CI':>20s} {'p':>8s}")
        print("-" * 175)

        ref_vids = loaded[REF][ds]["videos"]
        ref_ap   = loaded[REF][ds][METRIC]
        ref_nf   = loaded[REF][ds]["n_frames"]

        for name, _ in MODELS:
            if name == REF or name not in loaded:
                continue
            other_ap = align(ref_vids, loaded[name][ds]["videos"], loaded[name][ds][METRIC])

            w_res = wilcoxon_test(ref_ap, other_ap)
            fw_res = frame_weighted_paired_bootstrap(ref_ap, other_ap, ref_nf, rng_w)
            st_res = stratified_paired_bootstrap(ref_ap, other_ap, ref_nf, rng_s)

            print(f"vs {name:29s}  | "
                  f"{w_res['delta_median']:+8.4f} {w_res['p']:8.4f}{fmt_sig(w_res['p'])} | "
                  f"{fw_res['delta']:+6.4f}  [{fw_res['ci_lo']:+.4f}, {fw_res['ci_hi']:+.4f}] {fw_res['p']:8.4f}{fmt_sig(fw_res['p'])} | "
                  f"{st_res['delta']:+6.4f}  [{st_res['ci_lo']:+.4f}, {st_res['ci_hi']:+.4f}] {st_res['p']:8.4f}{fmt_sig(st_res['p'])}")

    print()
    print("Sig codes: *** p<0.001, ** p<0.01, * p<0.05")
    print()
    print("Methods recap:")
    print("  Wilcoxon            : non-parametric paired test on per-video AP30, no resampling")
    print("  Frame-weighted boot : 10k cluster bootstrap; per-video AP weighted by n_frames (~micro)")
    print("  Stratified boot     : 10k cluster bootstrap; videos stratified into 3 length quantiles")


if __name__ == "__main__":
    main()
