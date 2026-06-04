"""Generate the three paper figures.

Inputs (all in thesis/figdata/):
  tempo_R1__dataset2_split_test.json
  tempo_R1__cadica_50plus_new.json
  stqd__dataset2_split_test.json
  stqd__cadica_50plus_new.json
  postnet__dataset2_split_test.json     (optional)
  postnet__cadica_50plus_new.json       (optional)

Outputs (all in thesis/):
  fig_methods_bar.pdf
  fig_ap_iou.pdf
  fig_qualitative.pdf
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
import sys as _sys
_sys.path.insert(0, str(ROOT))
FIGDATA = ROOT / "thesis" / "figdata"
OUT = ROOT / "thesis"

# IEEE-friendly fonts
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 9,
    "figure.dpi": 200,
})


# bar chart 
def fig_methods_bar():
    """Grouped bar chart: AP30 on RIPCID vs CADICA, one bar per method."""
    methods = ["DetNet", "PS-STT", "STQD-Det", "Post-Net", "Prompt", "TempoRF-DETR"]
    ripcid  = [0.328, 0.271, 0.377, 0.525, 0.530, 0.581]
    cadica  = [0.089, 0.060, 0.170, 0.353, 0.345, 0.416]
    colors = ["#bdbdbd", "#9e9e9e", "#757575", "#8ecae6", "#219ebc", "#fb8500"]

    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.bar(x - width/2, ripcid, width, label="RIPCID-test", color=colors, edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, cadica, width, label="CADICA (OOD)", color=colors, edgecolor="black", linewidth=0.4, hatch="//", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("micro AP$_{30}$")
    ax.set_ylim(0, 0.70)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#cccccc", edgecolor="black", label="RIPCID-test"),
        Patch(facecolor="#cccccc", edgecolor="black", hatch="//", label="CADICA (OOD)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", frameon=False)
    plt.tight_layout()
    out = OUT / "fig_methods_bar.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


# AP-vs-IoU curves 
def _load_iou_sweep(path: Path) -> tuple[list, list]:
    with open(path) as f:
        d = json.load(f)
    items = sorted(((float(k), v) for k, v in d["iou_sweep"].items()), key=lambda x: x[0])
    return [t for t, _ in items], [v for _, v in items]


def fig_ap_iou():
    """Two-panel AP-vs-IoU plot: RIPCID-test (left) and CADICA (right)."""
    methods = [
        ("TempoRF-DETR (ours)",  "tempo_R1", "#fb8500", "-",  "o"),
        ("STQD-Det",             "stqd",     "#757575", "--", "s"),
        ("Post-Net",             "postnet",  "#219ebc", ":",  "^"),
    ]
    datasets = [
        ("RIPCID-test (in-distribution)", "dataset2_split_test"),
        ("CADICA (out-of-distribution)",  "cadica_50plus_new"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, (ds_label, ds_key) in zip(axes, datasets):
        for label, prefix, color, ls, marker in methods:
            path = FIGDATA / f"{prefix}__{ds_key}.json"
            if not path.exists():
                print(f"  (skip missing {path.name})")
                continue
            xs, ys = _load_iou_sweep(path)
            ax.plot(xs, ys, label=label, color=color, linestyle=ls,
                    marker=marker, markersize=4, linewidth=1.4)
        ax.set_title(ds_label)
        ax.set_xlabel("IoU threshold")
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
        ax.set_xlim(0.28, 0.77)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("micro AP")
    axes[0].set_ylim(0.0, 0.65)
    axes[0].legend(loc="upper right", frameon=False)
    plt.tight_layout()
    out = OUT / "fig_ap_iou.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


# qualitative panel 
def _iou(a, b):
    """IoU of single boxes a, b (xyxy)."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2]-a[0]) * (a[3]-a[1])
    bb = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(aa + bb - inter, 1e-6)


def _pick_best_examples(tempo_frames, stqd_frames, score_thr_tempo=0.45, score_thr_stqd=0.35,
                        iou_thr_match=0.35, max_candidates=200, who_wins="tempo"):
    """who_wins='tempo' -> frames where tempo hits GT and stqd misses; 'stqd' is the mirror case."""
    if who_wins not in ("tempo", "stqd"):
        raise ValueError(who_wins)
    tempo_by_key = {(f["video"], f["win_idx"]): f for f in tempo_frames}
    stqd_by_key  = {(f["video"], f["win_idx"]): f for f in stqd_frames}

    keys = sorted(set(tempo_by_key) & set(stqd_by_key))
    candidates = []
    for k in keys:
        ft = tempo_by_key[k]; fs = stqd_by_key[k]
        gts = np.array(ft["gt_boxes"], dtype=np.float32) if ft["gt_boxes"] else np.zeros((0,4))
        if gts.shape[0] == 0:
            continue
        if len(ft["scores"]) == 0:
            continue
        t_scores = np.array(ft["scores"]); t_boxes = np.array(ft["boxes"]).reshape(-1, 4)
        keep_t = t_scores >= score_thr_tempo
        if not keep_t.any():
            continue
        t_boxes, t_scores = t_boxes[keep_t], t_scores[keep_t]
        if len(t_boxes) == 0:
            continue
        best_t_iou = max(_iou(b, g) for b in t_boxes for g in gts) if len(t_boxes) and len(gts) else 0.0
        s_scores = np.array(fs["scores"]); s_boxes = np.array(fs["boxes"]).reshape(-1, 4)
        keep_s = s_scores >= score_thr_stqd
        s_boxes_kept = s_boxes[keep_s] if keep_s.any() else np.zeros((0,4))
        s_scores_kept = s_scores[keep_s] if keep_s.any() else np.zeros((0,))
        if len(s_boxes_kept) > 0 and len(gts) > 0:
            best_s_iou = max(_iou(b, g) for b in s_boxes_kept for g in gts)
        else:
            best_s_iou = 0.0
        # prefer winner IoU high, loser IoU low, 1 GT, few FP boxes by the winner
        n_fp_tempo = max(0, len(t_boxes) - 1)
        n_fp_stqd  = max(0, len(s_boxes_kept) - 1)
        if who_wins == "tempo":
            if best_t_iou < 0.5:
                continue
            score = 2.0 * best_t_iou - 1.5 * best_s_iou - 0.1 * n_fp_tempo - 0.1 * max(0, len(gts) - 1)
        else:  # stqd
            if best_s_iou < 0.5:
                continue
            score = 2.0 * best_s_iou - 1.5 * best_t_iou - 0.1 * n_fp_stqd - 0.1 * max(0, len(gts) - 1)
        candidates.append({
            "key": k, "score": float(score),
            "best_t_iou": float(best_t_iou), "best_s_iou": float(best_s_iou),
            "n_gt": int(len(gts)), "n_tempo_kept": int(len(t_boxes)),
            "n_stqd_kept": int(len(s_boxes_kept)),
            "centre_path": ft["centre_path"],
            "gt_area_pct": float(((gts[:, 2]-gts[:, 0]) * (gts[:, 3]-gts[:, 1])).sum() / (ft["orig_w"] * ft["orig_h"]) * 100.0),
        })
    candidates.sort(key=lambda c: -c["score"])
    return candidates[:max_candidates]


def _dashed_line(img, p1, p2, color, thickness=2, dash=8, gap=4):
    """Draw a dashed line between p1 and p2 on img (in-place)."""
    x1, y1 = p1; x2, y2 = p2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    n = max(1, length // (dash + gap))
    for i in range(n + 1):
        t0 = i * (dash + gap) / length
        t1 = min(1.0, (i * (dash + gap) + dash) / length)
        if t0 >= 1.0:
            break
        px0 = int(round(x1 + (x2 - x1) * t0))
        py0 = int(round(y1 + (y2 - y1) * t0))
        px1 = int(round(x1 + (x2 - x1) * t1))
        py1 = int(round(y1 + (y2 - y1) * t1))
        cv2.line(img, (px0, py0), (px1, py1), color, thickness, cv2.LINE_AA)


def _draw_dashed_rect(img, p1, p2, color, thickness=3, dash=10, gap=5):
    x1, y1 = p1; x2, y2 = p2
    _dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash, gap)
    _dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash, gap)
    _dashed_line(img, (x2, y2), (x1, y2), color, thickness, dash, gap)
    _dashed_line(img, (x1, y2), (x1, y1), color, thickness, dash, gap)


def _draw_overlay(img, gts, tempo_kept, stqd_kept, font_scale=0.45):
    """img BGR uint8; tempo_kept/stqd_kept are (box, score). GT drawn last so it stays visible
    under an overlapping prediction box."""
    out = img.copy()
    # STQD blue
    for box, score in stqd_kept:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 80, 80), 2)
        cv2.putText(out, f"S {score:.2f}", (x1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 80, 80), 1, cv2.LINE_AA)
    # TempoRF red
    for box, score in tempo_kept:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"T {score:.2f}", (x1, min(out.shape[0]-3, y2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
    # GT yellow dashed, last
    for box in gts:
        x1, y1, x2, y2 = map(int, box)
        _draw_dashed_rect(out, (x1, y1), (x2, y2), (0, 255, 255), thickness=3, dash=10, gap=5)
    return out


def _render_panel(tile_paths, n_rows=2, n_cols=4, tile_size=320, gap=8,
                  label_h=28, legend_h=0):
    """tile_paths: list of dicts {img_bgr, label}. Returns BGR image."""
    H = n_rows * tile_size + (n_rows + 1) * gap + legend_h
    W = n_cols * tile_size + (n_cols + 1) * gap
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    for i, td in enumerate(tile_paths):
        r, c = divmod(i, n_cols)
        y0 = gap + r * (tile_size + gap)
        x0 = gap + c * (tile_size + gap)
        img = td["img"]
        h, w = img.shape[:2]
        s = tile_size / max(h, w)
        new_w, new_h = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sq = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
        ry = (tile_size - new_h) // 2
        rx = (tile_size - new_w) // 2
        sq[ry:ry+new_h, rx:rx+new_w] = resized
        canvas[y0:y0+tile_size, x0:x0+tile_size] = sq
        lab = td.get("label", "")
        if lab:
            cv2.rectangle(canvas, (x0, y0), (x0 + tile_size, y0 + label_h), (40, 40, 40), -1)
            cv2.putText(canvas, lab, (x0 + 6, y0 + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _draw_legend_strip(W, height=44, font_scale=0.65):
    """Horizontal legend strip (BGR): GT dashed, TempoRF, STQD-Det."""
    strip = np.full((height, W, 3), 255, dtype=np.uint8)
    items = [
        ("GT (dashed)", (0, 255, 255), "dashed"),
        ("TempoRF-DETR (ours)", (0, 0, 255), "solid"),
        ("STQD-Det",    (255, 80, 80), "solid"),
    ]
    pad = 16
    box_sz = 22
    gap_in = 8
    spacings = []
    for it in items:
        txt = it[0]
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        spacings.append(box_sz + gap_in + tw)
    total_w = sum(spacings) + (len(items) - 1) * pad
    x = (W - total_w) // 2
    y = height // 2
    for (txt, color, style), sw in zip(items, spacings):
        if style == "dashed":
            _draw_dashed_rect(strip, (x, y - box_sz//2), (x + box_sz, y + box_sz//2),
                              color, thickness=3, dash=6, gap=3)
        else:
            cv2.rectangle(strip, (x, y - box_sz//2), (x + box_sz, y + box_sz//2), color, -1)
            cv2.rectangle(strip, (x, y - box_sz//2), (x + box_sz, y + box_sz//2), (0, 0, 0), 1)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(strip, txt, (x + box_sz + gap_in, y + th // 2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        x += sw + pad
    return strip


def _render_overlay_for_pick(ft, fs, score_t=0.45, score_s=0.35):
    img = cv2.imread(ft["centre_path"], cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gts = ft["gt_boxes"]
    t_kept = [(b, s) for b, s in zip(ft["boxes"], ft["scores"]) if s >= score_t]
    s_kept = [(b, s) for b, s in zip(fs["boxes"], fs["scores"]) if s >= score_s]
    return _draw_overlay(img, gts, t_kept, s_kept)


def _crop_around_gt(img, gts, target_aspect=2.6, scale_h=0.55):
    """Crop a target_aspect:1 region centred on the GT bbox(es). scale_h = crop height / image height."""
    h, w = img.shape[:2]
    if not gts:
        return img
    xs = [c for g in gts for c in (g[0], g[2])]
    ys = [c for g in gts for c in (g[1], g[3])]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    ch = int(h * scale_h)
    cw = int(ch * target_aspect)
    if cw > w:
        cw = w
        ch = int(cw / target_aspect)
    if ch > h:
        ch = h
        cw = int(ch * target_aspect)
    x1 = int(max(0, min(w - cw, cx - cw / 2)))
    y1 = int(max(0, min(h - ch, cy - ch / 2)))
    return img[y1:y1+ch, x1:x1+cw]


def _render_overlay_mode(ft, fs, mode, score_t=0.45, score_s=0.35,
                         crop=False, crop_aspect=2.6, crop_scale=0.55):
    """Draw a single overlay kind, mode in {gt, tempo, stqd}, optionally cropped around the lesion."""
    img = cv2.imread(ft["centre_path"], cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gts = ft["gt_boxes"]
    t_kept = [(b, s) for b, s in zip(ft["boxes"], ft["scores"]) if s >= score_t]
    s_kept = [(b, s) for b, s in zip(fs["boxes"], fs["scores"]) if s >= score_s]
    if mode == "gt":
        overlay = _draw_overlay(img, gts, [], [])
    elif mode == "tempo":
        overlay = _draw_overlay(img, [], t_kept, [])
    elif mode == "stqd":
        overlay = _draw_overlay(img, [], [], s_kept)
    else:
        raise ValueError(mode)
    if crop:
        return _crop_around_gt(overlay, gts, crop_aspect, crop_scale)
    return overlay


def fig_qualitative_inspect(n_top=24):
    """Debug grid of the top n_top candidates per dataset (both tempo-wins and stqd-wins) for manual picking."""
    for ds_key, ds_label in [
        ("dataset2_split_test", "RIPCID-test"),
        ("cadica_50plus_new",   "CADICA"),
    ]:
        with open(FIGDATA / f"tempo_R1__{ds_key}.json") as f:
            tempo = json.load(f)
        with open(FIGDATA / f"stqd__{ds_key}.json") as f:
            stqd = json.load(f)
        tempo_by_key = {(f["video"], f["win_idx"]): f for f in tempo["frames"]}
        stqd_by_key  = {(f["video"], f["win_idx"]): f for f in stqd["frames"]}
        for who in ("tempo", "stqd"):
            cand = _pick_best_examples(tempo["frames"], stqd["frames"], who_wins=who)
            print(f"\n{ds_label} [{who}-wins]: {len(cand)} candidates; rendering top {n_top}")
            tiles = []
            for i, p in enumerate(cand[:n_top]):
                ft = tempo_by_key[p["key"]]; fs = stqd_by_key[p["key"]]
                ov = _render_overlay_for_pick(ft, fs)
                label = f"#{i:02d} {p['key'][0]}_v{p['key'][1]:02d}  T={p['best_t_iou']:.2f} S={p['best_s_iou']:.2f}"
                tiles.append({"img": ov, "label": label})
            cols = 6
            rows = (len(tiles) + cols - 1) // cols
            panel = _render_panel(tiles, n_rows=rows, n_cols=cols, tile_size=280, gap=6)
            out = OUT / f"qual_inspect_{ds_key}_{who}wins.png"
            cv2.imwrite(str(out), panel)
            print(f"  wrote {out}  ({len(tiles)} tiles)")
            list_out = OUT / f"qual_candidates_{ds_key}_{who}wins.json"
            with open(list_out, "w") as f:
                json.dump(cand[:n_top], f, indent=2)
            print(f"  wrote {list_out}")


def fig_qualitative(ripcid_tempo_picks=None, ripcid_stqd_picks=None,
                    cadica_tempo_picks=None, cadica_stqd_picks=None,
                    score_t=0.45, score_s=0.35, tile_size: int = 420):
    """2 TempoRF-wins + 2 STQD-Det-wins per dataset, 4 cols x 2 rows."""
    panel_tiles = []
    pick_map = [
        ("dataset2_split_test", ripcid_tempo_picks, ripcid_stqd_picks, "RIPCID"),
        ("cadica_50plus_new",   cadica_tempo_picks, cadica_stqd_picks, "CADICA"),
    ]
    for ds_key, t_picks, s_picks, ds_short in pick_map:
        with open(FIGDATA / f"tempo_R1__{ds_key}.json") as f:
            tempo = json.load(f)
        with open(FIGDATA / f"stqd__{ds_key}.json") as f:
            stqd = json.load(f)
        tempo_by_key = {(f["video"], f["win_idx"]): f for f in tempo["frames"]}
        stqd_by_key  = {(f["video"], f["win_idx"]): f for f in stqd["frames"]}
        for who, picks in (("tempo", t_picks or [0, 1]), ("stqd", s_picks or [0, 1])):
            cand = _pick_best_examples(tempo["frames"], stqd["frames"], who_wins=who)
            print(f"\n{ds_short} [{who}-wins]: chosen {len(picks)} of {len(cand)} candidates")
            for ix in picks:
                if ix >= len(cand):
                    print(f"  skip ix={ix} (only {len(cand)} candidates)"); continue
                p = cand[ix]
                ft = tempo_by_key[p["key"]]; fs = stqd_by_key[p["key"]]
                print(f"  #{ix}  {p['key']}  T_iou={p['best_t_iou']:.2f}  "
                      f"S_iou={p['best_s_iou']:.2f}  area={p['gt_area_pct']:.2f}%  "
                      f"path={Path(p['centre_path']).name}")
                ov = _render_overlay_for_pick(ft, fs, score_t=score_t, score_s=score_s)
                tag = "T>S" if who == "tempo" else "S>T"
                label = (f"{ds_short} | {p['key'][0]} | {tag} | "
                         f"T={p['best_t_iou']:.2f} S={p['best_s_iou']:.2f}")
                panel_tiles.append({"img": ov, "label": label})
    panel = _render_panel(panel_tiles, n_rows=2, n_cols=4,
                          tile_size=tile_size, gap=10, label_h=32)
    legend = _draw_legend_strip(W=panel.shape[1], height=48, font_scale=0.7)
    final = np.vstack([panel, legend])
    out_png = OUT / "fig_qualitative.png"
    cv2.imwrite(str(out_png), final)
    from PIL import Image
    Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB)).save(str(OUT / "fig_qualitative.pdf"), "PDF", resolution=150)
    print(f"wrote {out_png}")
    print(f"wrote {OUT / 'fig_qualitative.pdf'}")


def fig_qualitative_triptych(ripcid_tempo_picks=None, ripcid_stqd_picks=None,
                              cadica_tempo_picks=None, cadica_stqd_picks=None,
                              score_t=0.45, score_s=0.35,
                              tile_w=520, tile_h=520,
                              split=False):
    """Triptych: col1=GT, col2=TempoRF only, col3=STQD only, one title strip per row.
    split=True writes per-dataset files (ripcid/cadica) so each fits one appendix page."""
    row_specs = []
    pick_map = [
        ("dataset2_split_test", ripcid_tempo_picks, ripcid_stqd_picks, "RIPCID"),
        ("cadica_50plus_new",   cadica_tempo_picks, cadica_stqd_picks, "CADICA"),
    ]
    for ds_key, t_picks, s_picks, ds_short in pick_map:
        with open(FIGDATA / f"tempo_R1__{ds_key}.json") as f:
            tempo = json.load(f)
        with open(FIGDATA / f"stqd__{ds_key}.json") as f:
            stqd = json.load(f)
        tempo_by_key = {(f["video"], f["win_idx"]): f for f in tempo["frames"]}
        stqd_by_key  = {(f["video"], f["win_idx"]): f for f in stqd["frames"]}
        for who, picks in (("tempo", t_picks or [0, 1]), ("stqd", s_picks or [0, 1])):
            cand = _pick_best_examples(tempo["frames"], stqd["frames"], who_wins=who)
            for ix in picks:
                if ix >= len(cand):
                    print(f"  skip {ds_short}/{who}/{ix} (only {len(cand)} cand)"); continue
                p = cand[ix]
                row_specs.append({
                    "ft": tempo_by_key[p["key"]],
                    "fs": stqd_by_key[p["key"]],
                    "p": p, "ds_short": ds_short, "winner": who,
                })

    def _render_triptych(specs, out_basename):
        n_rows = len(specs)
        n_cols = 3
        gap = 10
        col_header_h = 36
        row_title_h = 32
        legend_h = 44

        W = n_cols * tile_w + (n_cols + 1) * gap
        H = col_header_h + n_rows * (row_title_h + tile_h + gap) + gap + legend_h
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)

        col_titles = ["Ground truth", "TempoRF-DETR (ours)", "STQD-Det"]
        col_colors = [(0, 180, 180), (0, 0, 200), (180, 60, 60)]
        for c, (title, color) in enumerate(zip(col_titles, col_colors)):
            x0 = gap + c * (tile_w + gap)
            cv2.rectangle(canvas, (x0, 0), (x0 + tile_w, col_header_h), (235, 235, 235), -1)
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(canvas, title, (x0 + (tile_w - tw) // 2, (col_header_h + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        for r, spec in enumerate(specs):
            ft, fs, p = spec["ft"], spec["fs"], spec["p"]
            y_title = col_header_h + r * (row_title_h + tile_h + gap)
            y_tile = y_title + row_title_h
            tag = "TempoRF-DETR wins" if spec["winner"] == "tempo" else "STQD-Det wins"
            winner_color = (0, 0, 200) if spec["winner"] == "tempo" else (180, 60, 60)
            cv2.rectangle(canvas, (0, y_title), (W, y_title + row_title_h),
                          (248, 248, 248), -1)
            segments = [
                (f"{spec['ds_short']}  |  {p['key'][0]}  |  ", (40, 40, 40), 0.62, 1),
                (tag, winner_color, 0.66, 2),
                (f"  |  IoU(TempoRF)={p['best_t_iou']:.2f}   IoU(STQD-Det)={p['best_s_iou']:.2f}   "
                 f"|  bbox area={p['gt_area_pct']:.2f}%", (40, 40, 40), 0.62, 1),
            ]
            total_w = 0; sizes = []
            for txt, _c, fs_, th_ in segments:
                (tw, _h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs_, th_)
                sizes.append(tw); total_w += tw
            x = max(8, (W - total_w) // 2)
            baseline_y = y_title + (row_title_h + 14) // 2
            for (txt, color, fs_, th_), sw in zip(segments, sizes):
                cv2.putText(canvas, txt, (x, baseline_y),
                            cv2.FONT_HERSHEY_SIMPLEX, fs_, color, th_, cv2.LINE_AA)
                x += sw
            for c, mode in enumerate(("gt", "tempo", "stqd")):
                x0 = gap + c * (tile_w + gap)
                ov = _render_overlay_mode(ft, fs, mode, score_t=score_t, score_s=score_s)
                ih, iw = ov.shape[:2]
                s = min(tile_w / iw, tile_h / ih)
                new_w, new_h = int(round(iw * s)), int(round(ih * s))
                resized = cv2.resize(ov, (new_w, new_h), interpolation=cv2.INTER_AREA)
                tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
                ry = (tile_h - new_h) // 2; rx = (tile_w - new_w) // 2
                tile[ry:ry+new_h, rx:rx+new_w] = resized
                canvas[y_tile:y_tile+tile_h, x0:x0+tile_w] = tile
                cv2.rectangle(canvas, (x0, y_tile), (x0 + tile_w, y_tile + tile_h),
                              (180, 180, 180), 1)

        legend = _draw_legend_strip(W=W, height=legend_h, font_scale=0.65)
        canvas[H - legend_h:H, :] = legend

        out_png = OUT / f"{out_basename}.png"
        cv2.imwrite(str(out_png), canvas)
        from PIL import Image
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(
            str(OUT / f"{out_basename}.pdf"), "PDF", resolution=150)
        print(f"wrote {out_png}  ({W}x{H})")
        print(f"wrote {OUT / (out_basename + '.pdf')}")

    if split:
        ripcid_specs = [s for s in row_specs if s["ds_short"] == "RIPCID"]
        cadica_specs = [s for s in row_specs if s["ds_short"] == "CADICA"]
        if ripcid_specs:
            _render_triptych(ripcid_specs, "fig_qualitative_ripcid")
        if cadica_specs:
            _render_triptych(cadica_specs, "fig_qualitative_cadica")
        return

    n_rows = len(row_specs)
    n_cols = 3
    gap = 10
    col_header_h = 36     # column titles strip
    row_title_h = 32      # per-row title above the three tiles
    legend_h = 44

    W = n_cols * tile_w + (n_cols + 1) * gap
    H = col_header_h + n_rows * (row_title_h + tile_h + gap) + gap + legend_h
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    col_titles = ["Ground truth", "TempoRF-DETR (ours)", "STQD-Det"]
    col_colors = [(0, 180, 180), (0, 0, 200), (180, 60, 60)]
    for c, (title, color) in enumerate(zip(col_titles, col_colors)):
        x0 = gap + c * (tile_w + gap)
        cv2.rectangle(canvas, (x0, 0), (x0 + tile_w, col_header_h), (235, 235, 235), -1)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(canvas, title, (x0 + (tile_w - tw) // 2, (col_header_h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    for r, spec in enumerate(row_specs):
        ft, fs, p = spec["ft"], spec["fs"], spec["p"]
        y_title = col_header_h + r * (row_title_h + tile_h + gap)
        y_tile = y_title + row_title_h
        tag = "TempoRF-DETR wins" if spec["winner"] == "tempo" else "STQD-Det wins"
        winner_color = (0, 0, 200) if spec["winner"] == "tempo" else (180, 60, 60)
        cv2.rectangle(canvas, (0, y_title), (W, y_title + row_title_h),
                      (248, 248, 248), -1)
        # per-segment colour, so build the title piecewise
        segments = [
            (f"{spec['ds_short']}  |  {p['key'][0]}  |  ", (40, 40, 40), 0.62, 1),
            (tag, winner_color, 0.66, 2),
            (f"  |  IoU(TempoRF)={p['best_t_iou']:.2f}   IoU(STQD-Det)={p['best_s_iou']:.2f}   "
             f"|  bbox area={p['gt_area_pct']:.2f}%", (40, 40, 40), 0.62, 1),
        ]
        total_w = 0
        sizes = []
        for txt, _c, fs_, th_ in segments:
            (tw, _h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs_, th_)
            sizes.append(tw); total_w += tw
        x = max(8, (W - total_w) // 2)
        baseline_y = y_title + (row_title_h + 14) // 2
        for (txt, color, fs_, th_), sw in zip(segments, sizes):
            cv2.putText(canvas, txt, (x, baseline_y),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_, color, th_, cv2.LINE_AA)
            x += sw
        for c, mode in enumerate(("gt", "tempo", "stqd")):
            x0 = gap + c * (tile_w + gap)
            ov = _render_overlay_mode(ft, fs, mode, score_t=score_t, score_s=score_s)
            ih, iw = ov.shape[:2]
            sx = tile_w / iw; sy = tile_h / ih
            s = min(sx, sy)
            new_w, new_h = int(round(iw * s)), int(round(ih * s))
            resized = cv2.resize(ov, (new_w, new_h), interpolation=cv2.INTER_AREA)
            tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
            ry = (tile_h - new_h) // 2
            rx = (tile_w - new_w) // 2
            tile[ry:ry+new_h, rx:rx+new_w] = resized
            canvas[y_tile:y_tile+tile_h, x0:x0+tile_w] = tile
            cv2.rectangle(canvas, (x0, y_tile), (x0 + tile_w, y_tile + tile_h),
                          (180, 180, 180), 1)

    legend = _draw_legend_strip(W=W, height=legend_h, font_scale=0.65)
    canvas[H - legend_h:H, :] = legend

    out_png = OUT / "fig_qualitative.png"
    cv2.imwrite(str(out_png), canvas)
    from PIL import Image
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(
        str(OUT / "fig_qualitative.pdf"), "PDF", resolution=150)
    print(f"wrote {out_png}  ({W}x{H})")
    print(f"wrote {OUT / 'fig_qualitative.pdf'}")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "bar":
        fig_methods_bar()
    elif target == "iou":
        fig_ap_iou()
    elif target == "qual-inspect":
        fig_qualitative_inspect(n_top=24)
    elif target in ("qual", "qual-triptych"):
        # picks: --ripcid-t 0,1 --ripcid-s 0,1 --cadica-t 0,1 --cadica-s 0,1
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--ripcid-t", type=str, default="0,1")
        ap.add_argument("--ripcid-s", type=str, default="0,1")
        ap.add_argument("--cadica-t", type=str, default="0,1")
        ap.add_argument("--cadica-s", type=str, default="0,1")
        ap.add_argument("--split", action="store_true",
                        help="write fig_qualitative_ripcid + fig_qualitative_cadica instead of one figure")
        args, _ = ap.parse_known_args(sys.argv[2:])
        parse = lambda s: [int(x) for x in s.split(",")]
        if target == "qual-triptych":
            fig_qualitative_triptych(ripcid_tempo_picks=parse(args.ripcid_t),
                                     ripcid_stqd_picks =parse(args.ripcid_s),
                                     cadica_tempo_picks=parse(args.cadica_t),
                                     cadica_stqd_picks =parse(args.cadica_s),
                                     split=args.split)
        else:
            fig_qualitative(ripcid_tempo_picks=parse(args.ripcid_t),
                            ripcid_stqd_picks =parse(args.ripcid_s),
                            cadica_tempo_picks=parse(args.cadica_t),
                            cadica_stqd_picks =parse(args.cadica_s))
    elif target == "all":
        fig_methods_bar()
        fig_ap_iou()


if __name__ == "__main__":
    main()
