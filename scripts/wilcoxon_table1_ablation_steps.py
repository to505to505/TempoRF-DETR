#!/usr/bin/env python3
"""Progressive ablation tests for Table I: does each added module give a significant gain?

Walks the ablation chain (Row 4..7) and runs three paired, video-level tests on AP30:
Wilcoxon signed-rank, frame-weighted bootstrap, and sequence-stratified bootstrap.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).parent.parent.resolve()
import sys as _sys
_sys.path.insert(0, str(ROOT))
RUN = lambda name: ROOT / f"rfdetr_video/runs/{name}/ablation_results.json"

# Row in Table I -> run path
ROWS = {
    "Row 4: ETF, RIPCID-only init":    RUN("video_5_etf_consistency_dataset2"),
    "Row 5: ETF, ARCADE init":         RUN("video_5_etf_consistency"),
    "Row 6: ETF + KD":                 RUN("video_5_etf_consistency_distill"),
    "Row 7: ETF + KD + CRRCD (full)":  RUN("video_overfit_R1"),
}

# (later_row, earlier_row, claim_label)
PROGRESSIONS = [
    ("Row 5: ETF, ARCADE init",        "Row 4: ETF, RIPCID-only init",  "ARCADE init effect"),
    ("Row 6: ETF + KD",                "Row 5: ETF, ARCADE init",        "Adding KD"),
    ("Row 7: ETF + KD + CRRCD (full)", "Row 6: ETF + KD",                "Adding CRRCD"),
    ("Row 7: ETF + KD + CRRCD (full)", "Row 5: ETF, ARCADE init",        "Combined KD + CRRCD"),
    ("Row 7: ETF + KD + CRRCD (full)", "Row 4: ETF, RIPCID-only init",  "Full pipeline vs ETF-only/RIPCID-init"),
]

DATASETS = ["dataset2_split_test", "cadica_50plus_new"]
LABEL = {"dataset2_split_test": "RIPCID-test (in-dist)",
         "cadica_50plus_new":    "CADICA (OOD)"}
N_BOOT = 10_000
SEED = 12345


def load_per_video(path):
    with open(path) as f:
        data = json.load(f)
    ds_list = data["ablations"][0]["datasets"] if "ablations" in data else data["datasets"]
    out = {}
    for ds in ds_list:
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


def wilcoxon(a, b):
    diff = a - b
    if np.all(diff == 0):
        return {"delta_med": 0.0, "p": 1.0}
    res = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"delta_med": float(np.median(diff)), "p": float(res.pvalue)}


def frame_weighted_boot(a, b, w, rng, n_boot=N_BOOT):
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = ((a - b) * w)[idx].sum(axis=1) / w[idx].sum(axis=1)
    point = ((a - b) * w).sum() / w.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    if point >= 0: p = 2.0 * (np.sum(diffs <= 0) + 1) / (n_boot + 1)
    else:          p = 2.0 * (np.sum(diffs >= 0) + 1) / (n_boot + 1)
    return {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi), "p": float(min(p, 1.0))}


def stratified_boot(a, b, n_frames, rng, n_boot=N_BOOT, n_strata=3):
    qs = np.quantile(n_frames, np.linspace(0, 1, n_strata + 1)[1:-1])
    strata = np.digitize(n_frames, qs)
    idx_per = [np.where(strata == s)[0] for s in range(n_strata)]
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        samp = np.concatenate([rng.choice(i, size=len(i), replace=True) for i in idx_per if len(i) > 0])
        diffs[k] = (a[samp] - b[samp]).mean()
    point = (a - b).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    if point >= 0: p = 2.0 * (np.sum(diffs <= 0) + 1) / (n_boot + 1)
    else:          p = 2.0 * (np.sum(diffs >= 0) + 1) / (n_boot + 1)
    return {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi), "p": float(min(p, 1.0))}


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "   "


def main():
    rng_w = np.random.default_rng(SEED)
    rng_s = np.random.default_rng(SEED + 1)

    print("PROGRESSIVE ABLATION TESTS - module-by-module additions")
    print("=" * 110)
    print(f"Metric: AP30 | n_boot: {N_BOOT} | seed: {SEED}")
    print()

    loaded = {name: load_per_video(path) for name, path in ROWS.items() if path.exists()}
    for name, path in ROWS.items():
        if not path.exists():
            print(f"  MISSING: {name} -> {path}")

    for ds in DATASETS:
        print(f"\n## {LABEL[ds]}")
        print(f"{'Claim':40s} | {'micro paper':>11s} | {'Wilcoxon':>20s} | {'Frame-weighted':>40s} | {'Stratified':>40s}")
        print(f"{'':40s} | {'':>11s} | {'deltamed':>9s} {'p':>9s} | {'delta':>6s}  {'95% CI':>20s} {'p':>8s} | {'delta':>6s}  {'95% CI':>20s} {'p':>8s}")
        print("-" * 175)

        for later, earlier, claim in PROGRESSIONS:
            if later not in loaded or earlier not in loaded:
                continue
            ref_vids = loaded[later][ds]["videos"]
            a = loaded[later][ds]["AP30"]
            w = loaded[later][ds]["n_frames"]
            b = align(ref_vids, loaded[earlier][ds]["videos"], loaded[earlier][ds]["AP30"])

            # micro-pooled number for the paper, not a test output
            mp = (a * w).sum() / w.sum() - (b * w).sum() / w.sum()

            wr  = wilcoxon(a, b)
            fwr = frame_weighted_boot(a, b, w, rng_w)
            sr  = stratified_boot(a, b, w, rng_s)

            print(f"{claim:40s} | {mp:+11.4f} | "
                  f"{wr['delta_med']:+9.4f} {wr['p']:8.4f}{sig(wr['p'])} | "
                  f"{fwr['delta']:+6.4f}  [{fwr['ci_lo']:+.4f}, {fwr['ci_hi']:+.4f}] {fwr['p']:8.4f}{sig(fwr['p'])} | "
                  f"{sr['delta']:+6.4f}  [{sr['ci_lo']:+.4f}, {sr['ci_hi']:+.4f}] {sr['p']:8.4f}{sig(sr['p'])}")

    print()
    print("Sig codes: *** p<0.001, ** p<0.01, * p<0.05")


if __name__ == "__main__":
    main()
