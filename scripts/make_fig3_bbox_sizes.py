"""Compute bbox-size statistics for RIPCID vs ARCADE for the methods section note."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).parent.parent.resolve()
import sys as _sys
_sys.path.insert(0, str(ROOT))

# Datasets to compare. ARCADE uses single-class label files.
RUNS = {
    "RIPCID (train+val+test)": [
        ROOT / "data/dataset2_split/train/labels",
        ROOT / "data/dataset2_split/valid/labels",
        ROOT / "data/dataset2_split/test/labels",
    ],
    "RIPCID test": [ROOT / "data/dataset2_split/test/labels"],
    "ARCADE (train+val+test)": [
        ROOT / "data/stenosis_arcade_singlelabel/train/labels",
        ROOT / "data/stenosis_arcade_singlelabel/val/labels",
        ROOT / "data/stenosis_arcade_singlelabel/test/labels",
    ],
    "CADICA": [ROOT / "data/cadica_50plus_new/labels"],
}

# We need image sizes to convert YOLO normalised -> pixel size. Use a canonical
# (assumed) image-size per dataset to compute pixel sizes. Both RIPCID and
# ARCADE images are square; we sample one image per dataset for the size.
IMG_DIRS = {
    "RIPCID (train+val+test)": [
        ROOT / "data/dataset2_split/train/images",
        ROOT / "data/dataset2_split/valid/images",
        ROOT / "data/dataset2_split/test/images",
    ],
    "RIPCID test": [ROOT / "data/dataset2_split/test/images"],
    "ARCADE (train+val+test)": [
        ROOT / "data/stenosis_arcade_singlelabel/train/images",
        ROOT / "data/stenosis_arcade_singlelabel/val/images",
        ROOT / "data/stenosis_arcade_singlelabel/test/images",
    ],
    "CADICA": [ROOT / "data/cadica_50plus_new/images"],
}


def load_image_size(img_dirs) -> tuple[int, int]:
    for d in img_dirs:
        for p in d.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img.shape[:2]  # (h, w)
    raise FileNotFoundError(f"no image found in {img_dirs}")


def collect(label_dirs, img_size_hw):
    h, w = img_size_hw
    rel_w, rel_h, rel_area_frac = [], [], []
    pix_w, pix_h, pix_area = [], [], []
    n_boxes = 0
    n_files = 0
    for d in label_dirs:
        for lbl in d.iterdir():
            if lbl.suffix != ".txt":
                continue
            n_files += 1
            if lbl.stat().st_size == 0:
                continue
            arr = np.loadtxt(lbl, dtype=np.float32).reshape(-1, 5)
            for row in arr:
                _, _, _, bw, bh = row
                rel_w.append(float(bw))
                rel_h.append(float(bh))
                rel_area_frac.append(float(bw * bh))
                pix_w.append(float(bw * w))
                pix_h.append(float(bh * h))
                pix_area.append(float(bw * w * bh * h))
                n_boxes += 1
    return {
        "img_h": h, "img_w": w,
        "n_files": n_files, "n_boxes": n_boxes,
        "rel_w": np.array(rel_w), "rel_h": np.array(rel_h),
        "rel_area_frac": np.array(rel_area_frac),
        "pix_w": np.array(pix_w), "pix_h": np.array(pix_h),
        "pix_area": np.array(pix_area),
    }


def summarise(name, s):
    a = s["pix_area"]
    rel = s["rel_area_frac"]
    side = np.sqrt(a)
    return {
        "dataset": name,
        "n_files": s["n_files"],
        "n_boxes": s["n_boxes"],
        "img_hxw": f"{s['img_h']}x{s['img_w']}",
        "median_side_px": float(np.median(side)),
        "mean_side_px": float(np.mean(side)),
        "p25_side_px": float(np.percentile(side, 25)),
        "p75_side_px": float(np.percentile(side, 75)),
        "median_rel_area_pct": float(np.median(rel) * 100),
        "mean_rel_area_pct": float(np.mean(rel) * 100),
    }


def main():
    summaries = []
    raw = {}
    for name, lbl_dirs in RUNS.items():
        img_size = load_image_size(IMG_DIRS[name])
        s = collect(lbl_dirs, img_size)
        summaries.append(summarise(name, s))
        raw[name] = s

    for r in summaries:
        print(f"\n{r['dataset']}  ({r['n_files']} files, {r['n_boxes']} boxes, img {r['img_hxw']})")
        print(f"  bbox side (px):  median={r['median_side_px']:.1f}  mean={r['mean_side_px']:.1f}  p25={r['p25_side_px']:.1f}  p75={r['p75_side_px']:.1f}")
        print(f"  bbox area (% of image):  median={r['median_rel_area_pct']:.3f}%  mean={r['mean_rel_area_pct']:.3f}%")

    with open("/tmp/bbox_stats.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("\nwrote /tmp/bbox_stats.json")

    # plot 
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.titlesize": 9, "figure.dpi": 200,
    })

    keep = ["ARCADE (train+val+test)", "RIPCID (train+val+test)", "CADICA"]
    display = {"ARCADE (train+val+test)": "ARCADE", "RIPCID (train+val+test)": "RIPCID", "CADICA": "CADICA"}
    colors = {"ARCADE": "#8ecae6", "RIPCID": "#fb8500", "CADICA": "#a3b18a"}

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    data = []
    labels = []
    for name in keep:
        s = raw[name]
        area_pct = s["rel_area_frac"] * 100.0  # % of image area
        data.append(area_pct)
        labels.append(display[name])
    bp = ax.boxplot(data, labels=labels, showfliers=False, widths=0.55, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.6))
    for patch, name in zip(bp["boxes"], keep):
        patch.set_facecolor(colors[display[name]])
        patch.set_alpha(0.75)
    ax.set_ylabel("Bbox area (% of image)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = ROOT / "thesis" / "fig_bbox_sizes.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
