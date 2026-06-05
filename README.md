# TempoRF-DETR: Video-Based Coronary Stenosis Detection with Image-to-Video Distillation

Bachelor thesis code submission — Dmitrii Sakharov, Maastricht University, 2026.

---

## 1. Overview

TempoRF-DETR is a video extension of [RF-DETR](https://github.com/roboflow/rf-detr) for coronary stenosis detection in X-ray angiography. It combines:

- **Early Temporal Fusion (ETF)** — a single position-wise multi-head attention block inserted between the 2D backbone and the DETR decoder, mixing context across all `T` frames of a clip in one forward pass.
- **Image-to-video distillation** — a frozen 2D RF-DETR teacher (trained on the union of single-frame and decoded video data) supervises the video student per frame via KD-DETR (specific + general query sampling) and CRRCD (cross-resolution relational contrastive distillation).

Results on the held-out test split: **AP$_{30}$ = 0.581 in-distribution (RIPCID-test), 0.416 out-of-distribution (CADICA)** — see Tables I–III and Discussion in the full thesis (`Dmitrii Sakharov Thesis.pdf`, at the repository root).

---

## 2. Repository Structure

```
stenosis_thesis_submission/
├── README.md
├── Dmitrii Sakharov Thesis.pdf              # Final thesis (full PDF)
├── requirements.txt
├── .gitignore
│
├── rf-detr/                                 # Forked RF-DETR (pip install -e .)
│   └── runs/                                # 2D RF-DETR runs (Table I rows 1–3)
│       ├── dataset2_augs/                          # Row 1: RF-DETR-S (RIPCID only)
│       ├── rfdetr_small_arcade2x_512_reg/          # Row 2: RF-DETR-S (R+A) — student init
│       └── rfdetr_large_arcade2x_704_reg/          # Row 3: RF-DETR-L (R+A) — teacher
│
├── rfdetr_video/                            # TempoRF-DETR (ETF + KD + CRRCD)
│   ├── train.py, evaluate.py, model.py, dataset.py, …
│   ├── consistency.py, prompt.py, postnet.py        # Table I/III modules
│   ├── sequence_dataset.py, sequence_eval.py        # Shared sequence utilities
│   ├── distill/                                     # KD + CRRCD: frozen_teacher.py, losses.py, crrcd.py
│   ├── tests/
│   └── runs/                                # 7 runs (Tables I rows 4–7, Table II main, Table III)
│       ├── video_5_etf_consistency_dataset2/       # Row 4: ETF (R-only init)
│       ├── video_5_etf_consistency/                # Row 5: ETF (R+A init)
│       ├── video_5_etf_consistency_distill/        # Row 6: +KD
│       ├── video_overfit_R1/  # Row 7: +KD+CRRCD = full TempoRF-DETR
│       ├── video_5_postnet_T5/                     # Table III: Post-Network Tuning
│       ├── video_5_prompt_T5/                      # Table III: Prompt Tuning
│       └── stqd_det_T5_dataset2/                   # Table II: STQD-Det (run dir)
│
├── detnet/                                  # Stenosis-DetNet reimpl. (Pang 2021) — Table II
│   └── runs/detnet_v1_T5/
├── psstt/                                   # PS-STT reimpl. (Han 2023) — Table II
│   └── runs/psstt_20ep/
├── stqd_det/                                # STQD-Det reimpl. (Li 2024) — Table II (weights in rfdetr_video/runs/)
│
├── scripts/                                 # Clearly named training / eval / stats / figure scripts
│   ├── train_2d_rfdetr.py                       # Train 2D RF-DETR (Table I rows 1–3)
│   ├── eval_video_run.py                        # Evaluate any TempoRF/STQD-Det/PET run dir (Tables I/II/III)
│   ├── eval_baseline_detnet.py                  # Evaluate Stenosis-DetNet (Table II)
│   ├── eval_baseline_psstt.py                   # Evaluate PS-STT (Table II)
│   ├── wilcoxon_table2_vs_baselines.py          # Wilcoxon vs each video baseline (Table II)
│   ├── wilcoxon_per_model_bootstrap.py          # Per-model mean AP30 + 95% bootstrap CI
│   ├── wilcoxon_table1_ablation_steps.py        # Wilcoxon for progressive ablation steps (Table I)
│   ├── compute_pr_f1_iou30.py                   # Micro-pooled P/R/F1 @ IoU=0.30
│   ├── make_fig3_bbox_sizes.py                  # Fig. 3
│   ├── dump_predictions_for_figs.py             # Dumps per-frame predictions → thesis/figdata/
│   └── make_figs_4_5_6.py                       # Fig. 4 (methods_bar), Fig. 5 (qualitative), Fig. 6 (ap_iou)
│
├── results/                                 # Pre-generated outputs + paper cross-check (see results/README.md)
│   ├── table1_module_ablation.txt           # Table I numbers + per-row paper vs repo verification
│   ├── table2_video_baselines.txt           # Table II numbers + verification
│   ├── table3_pet_alternatives.txt          # Table III numbers + verification
│   ├── wilcoxon_table1_ablation_steps.txt   # Wilcoxon p-values for progressive ablation
│   ├── wilcoxon_table2_vs_baselines.txt     # Wilcoxon p-values vs baselines
│   ├── wilcoxon_per_model_bootstrap.txt     # Per-model AP30 + bootstrap CI
│   └── README.md
│
├── thesis/                                  # LaTeX source + compiled paper PDF + figures
│   ├── conference.tex, conference.pdf
│   ├── TempoRF-DETR_final.{pdf,svg}, Distillation_final_v2.{pdf,svg}
│   ├── fig_bbox_sizes.pdf, fig_methods_bar.pdf, fig_qualitative.pdf, fig_ap_iou.pdf
│   ├── figdata/                             # Per-frame predictions for figure regeneration
│   └── IEEEtran.cls, UMlogo.jpg, stenosis.bib
```

---

## 3. Installation

```bash
# Conda environment (recommended)
conda create -n stenosis python=3.10 -y
conda activate stenosis

# Install RF-DETR fork (editable, used by all packages)
pip install -e ./rf-detr

# Project dependencies
pip install -r requirements.txt
```

CUDA 11.8 / 12.x compatible. Tested on RTX 3060 12GB (training) and CPU (eval, slower).

---

## 4. Datasets

Datasets are not bundled (size + medical-imaging licensing). After installation,
place each dataset under `data/` following the structure below.

### 4.1 RIPCID (called `dataset2_split` in code)

- Primary in-distribution dataset (videos from 64 patients, Philips Azurion 3 + Siemens Artis Zee).
- **License / access:** publicly available — open-access dataset released with the source paper; download from the figshare mirror listed below.
- **Where to put it:** `data/dataset2_split/`
- **Expected layout:** `train/`, `val/`, `test/` (patient-disjoint splits), `data.yaml`.
- Original source: Danilov et al., *Scientific Reports* 11:7582, 2021.
  Public mirror: [10.6084/m9.figshare.13643940](https://doi.org/10.6084/m9.figshare.13643940).

### 4.2 ARCADE (stenosis subset)

- Single-frame ARCADE dataset filtered to the stenosis class — used by the 2D teacher only.
- **License:** CC-BY 4.0.
- **Where to put it:** `data/stenosis_arcade/`
- Original source: Popov et al., *Scientific Data* 11:20, 2024 — [ARCADE on Zenodo](https://zenodo.org/records/10390295).

### 4.3 CADICA (≥50% stenosis subset)

- Out-of-distribution test set (Siemens Artis Zee).
- **License:** CC-BY 4.0.
- **Where to put it:** `data/cadica_50plus_new/`
- **How to obtain:** download the full CADICA release from [Mendeley Data DOI 10.17632/p9bpx9ctcv.1](https://data.mendeley.com/datasets/p9bpx9ctcv/1), keep only sequences with ground-truth lesion grade ≥50%, and extract centre-frames as documented in the original CADICA paper (Jiménez-Partinen et al., 2024 — [arXiv:2402.00570](https://arxiv.org/abs/2402.00570)).
- Sizes: 160 sequences, 1,614 centre frames, 2,284 boxes.

### 4.4 Expected `data/` layout

```
data/
├── dataset2_split/     # RIPCID  (train/, val/, test/, data.yaml)
├── stenosis_arcade/    # ARCADE  (train/, val/, test/, data.yaml)
└── cadica_50plus_new/  # CADICA  (sequences/, annotations/)
```

---

## 5. Pretrained Weights

**Only the final TempoRF-DETR model weight is bundled** (`rfdetr_video/runs/video_overfit_R1/best.pth`, ~124 MB).
This is the canonical model that produces the 0.581 / 0.416 headline numbers — sufficient for examiners to run inference on their own X-ray angiography data.

The final weights are also published on the HuggingFace Hub: **[`to505to505/TempoRF-DETR`](https://huggingface.co/to505to505/TempoRF-DETR)** (download with `hf download to505to505/TempoRF-DETR tempo_rf_detr_full.pth`).

| File | Used for | Bundled? |
|---|---|---|
| `rfdetr_video/runs/video_overfit_R1/best.pth` | **Final TempoRF-DETR (Table I row 7, Table II/III "ours")** — the model to test on new data | ✅ Yes |
| 2D RF-DETR teacher/student weights (Table I rows 1–3) | Reproducing the 2D ablation rows | ❌ No — retrain from configs in `rf-detr/runs/*/` |
| Video ablation weights (Table I rows 4–6) | Reproducing intermediate ablation steps | ❌ No — retrain from configs in `rfdetr_video/runs/*/config.json` |
| Baseline weights (Stenosis-DetNet / PS-STT / STQD-Det) | Reproducing Table II baselines | ❌ No — retrain from configs in `{detnet,psstt}/runs/*/config.json` and `rfdetr_video/runs/stqd_det_T5_dataset2/config.json` |
| PET weights (Post-Net / Prompt Tuning) | Reproducing Table III alternatives | ❌ No — retrain from configs in `rfdetr_video/runs/video_5_{postnet,prompt}_T5/config.json` |

The non-final weights are removed to keep the submission compact. **All experimental metadata is preserved**: each run directory still contains `config.json`, `best.txt`, `history.json`, `train.csv`, and `ablation_results.{json,txt}` — enough to retrain from scratch (see section 7) and to regenerate the Wilcoxon p-values in `results/` without needing the weights.

If the **DINOv2 backbone weights** (`rf-detr-{nano,small}.pth`) are missing at first run, the `rfdetr` package auto-downloads them from [HuggingFace `roboflow/rf-detr`](https://huggingface.co/roboflow/rf-detr) on first call.

---

## 6. Re-running the Experiments

> All commands assume `cwd = stenosis_thesis_submission/` (the project root). Always `conda activate stenosis` first.

### 6.1 Evaluation (regenerate the table numbers)

`eval_video_run.py` reads a run's `best.pth` and writes fresh `ablation_results.{json,txt}` into the run dir.

Only the **final TempoRF-DETR weight is bundled** (`rfdetr_video/runs/video_overfit_R1/best.pth`), so the only eval that runs out-of-the-box is:

```bash
python scripts/eval_video_run.py --run rfdetr_video/runs/video_overfit_R1
# Expected: AP30 ≈ 0.581 in-dist (RIPCID-test), 0.416 OOD (CADICA)
```

For the other rows of Tables I/II/III, **retrain the corresponding model first** (section 7) — each run directory already holds the exact `config.json` that was used.

The numbers themselves are also pre-recorded in `results/table1_module_ablation.txt`, `results/table2_video_baselines.txt`, `results/table3_pet_alternatives.txt` and read directly from each run's saved `ablation_results.json`.

### 6.2 Wilcoxon p-values (used in Tables I/II/III and Discussion)

```bash
python scripts/wilcoxon_table2_vs_baselines.py   > results/wilcoxon_table2_vs_baselines.txt
python scripts/wilcoxon_per_model_bootstrap.py  > results/wilcoxon_per_model_bootstrap.txt
python scripts/wilcoxon_table1_ablation_steps.py > results/wilcoxon_table1_ablation_steps.txt
```

### 6.3 Operating-point metrics

```bash
python scripts/compute_pr_f1_iou30.py
```

### 6.4 Figures

```bash
python scripts/make_fig3_bbox_sizes.py        # Fig. 3 (thesis/fig_bbox_sizes.pdf)
python scripts/dump_predictions_for_figs.py        # Optional: re-dump per-frame predictions → thesis/figdata/
python scripts/make_figs_4_5_6.py              # Fig. 4, 5, 6 (methods_bar, qualitative, ap_iou)
```

### 6.5 Compile the thesis

```bash
cd thesis
pdflatex conference.tex && pdflatex conference.tex   # second pass resolves cross-refs
```

---

## 7. Training from Scratch (re-running each experiment)

Every experiment has its `config.json` (or `training_config.json`) saved in the corresponding run dir. Re-training simply launches the appropriate trainer with that config; outputs land in a fresh sibling directory.

### 7.1 Table I rows 1–3 (2D RF-DETR)

```bash
# Row 1: 2D RF-DETR-S, RIPCID only
python scripts/train_2d_rfdetr.py --model-size small \
    --dataset-dir data/dataset2_split \
    --name rfdetr_small_RIPCIDonly --epochs 100 --batch-size 16

# Row 2: 2D RF-DETR-S, RIPCID + ARCADE (student init)
python scripts/train_2d_rfdetr.py --model-size small \
    --dataset-dir data/combined_arcade_dataset2 \
    --name rfdetr_small_arcade2x_512_reg --epochs 100 --batch-size 16

# Row 3: 2D RF-DETR-L, RIPCID + ARCADE (teacher)
python scripts/train_2d_rfdetr.py --model-size large \
    --dataset-dir data/combined_arcade_dataset2 \
    --resolution 704 \
    --name rfdetr_large_arcade2x_704_reg --epochs 100 --batch-size 4
```

The full training config of row 3 is saved at `rf-detr/runs/rfdetr_large_arcade2x_704_reg/training_config.json`.

### 7.2 Table I rows 4–7 and Table III (TempoRF-DETR variants)

```bash
# Each TempoRF-DETR variant launches from its saved config.json
python -m rfdetr_video.train --config rfdetr_video/runs/video_5_etf_consistency_dataset2/config.json     # Row 4
python -m rfdetr_video.train --config rfdetr_video/runs/video_5_etf_consistency/config.json              # Row 5
python -m rfdetr_video.train --config rfdetr_video/runs/video_5_etf_consistency_distill/config.json     # Row 6
python -m rfdetr_video.train --config rfdetr_video/runs/video_overfit_R1/config.json # Row 7 (full)

# Table III alternatives
python -m rfdetr_video.train --config rfdetr_video/runs/video_5_postnet_T5/config.json
python -m rfdetr_video.train --config rfdetr_video/runs/video_5_prompt_T5/config.json
```

Full TempoRF-DETR training takes ~24h on a single RTX 3060.

### 7.3 Table II baselines

```bash
# Stenosis-DetNet
python -m detnet.train --config detnet/runs/detnet_v1_T5/config.json

# PS-STT
python -m psstt.train --config psstt/runs/psstt_20ep/config.json

# STQD-Det
python -m stqd_det.train --config rfdetr_video/runs/stqd_det_T5_dataset2/config.json
```

---

## 8. Smoke Tests

```bash
conda activate stenosis
pytest rfdetr_video/tests/ detnet/tests/ psstt/tests/ stqd_det/tests/ -q
```

Expected: **125 passed, 26 skipped**.

---

## 9. References

The thesis re-implements three video-based stenosis detectors as baselines:

- **Stenosis-DetNet** — Pang et al., *Comput. Med. Imaging Graph.* 89, 101900 (2021)
- **PS-STT** — Han et al., *Comput. Biol. Med.* 153, 106546 (2023)
- **STQD-Det** — Li et al., *IEEE TPAMI* 46(12), 9908–9920 (2024)

Full bibliography in [`thesis/stenosis.bib`](thesis/stenosis.bib) and inline in `thesis/conference.tex`.

---

## 10. Contact

Dmitrii Sakharov — `dmitrii.sakharov04@gmail.com`
Supervisor(s): Yusuf Can Semeri, Mark Punt (Maastricht University, Faculty of Science and Engineering).
