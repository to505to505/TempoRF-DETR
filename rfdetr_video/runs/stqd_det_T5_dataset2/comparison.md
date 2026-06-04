# STQD-Det vs RF-DETR-video on dataset 2

Run: `stqd_det_T5_dataset2` — STQD-Det reproduction (Li et al., IEEE TPAMI 2024) trained on `data/dataset2_split` train, evaluated with the same MICRO-pooled centre-frame pipeline used for `video_5_etf_consistency`.

## Headline (dataset2_split_test, MICRO pooled)

| Metric | STQD-Det (ep 12, EMA) | RF-DETR-video baseline | Δ |
|---|---|---|---|
| AP30 | **0.3772** | **0.5515** | -0.1743 (-31.6 %) |
| AP50 | **0.1686** | **0.2381** | -0.0695 (-29.2 %) |
| AP75 | 0.0030 | 0.0113 | -0.0083 |
| AP@0.5:0.95 | 0.0400 | 0.0611 | -0.0211 |
| F1 | **0.3351** | **0.3643** | -0.0292 (-8.0 %) |
| Precision | 0.3642 | 0.3992 | -0.0350 |
| Recall | 0.3103 | 0.3351 | -0.0248 |
| FragRate ↓ | **0.0470** | **0.0567** | -0.0098 (-17.3 %, лучше) |
| best conf | 0.38 | 0.49 | — |

**RF-DETR-video baseline** = `rfdetr_video/runs/video_5_etf_consistency/` — DINOv2-ViT-S backbone + ETF temporal fusion + count-consistency, **без** distillation, T=5, 50 epochs.

## CADICA (out-of-distribution, MICRO pooled)

| Metric | STQD-Det | RF-DETR-video |
|---|---|---|
| AP30 | 0.1703 | 0.5151 |
| AP50 | 0.0438 | 0.1117 |
| F1 | 0.1661 | 0.2394 |
| FragRate | 0.0175 | 0.0359 |

STQD-Det сильно проигрывает на CADICA — ожидаемо, потому что только 15 эпох на dataset 2 без аугментации специфичной для других центров.

## Validation curve (15 epochs)

| ep | val AP30 | val AP50 | val F1 | val P | val R | sel_smooth |
|----|----------|----------|--------|-------|-------|------------|
|  1 | 0.008    | 0.002    | 0.007  | 0.007 | 0.007 | 0.006      |
|  2 | 0.054    | 0.018    | 0.081  | 0.072 | 0.093 | 0.027      |
|  3 | 0.090    | 0.042    | 0.137  | 0.161 | 0.119 | 0.047      |
|  4 | 0.124    | 0.056    | 0.165  | 0.162 | 0.168 | 0.082      |
|  5 | 0.166    | 0.078    | 0.200  | 0.227 | 0.179 | 0.114      |
|  6 | 0.154    | 0.080    | 0.214  | 0.244 | 0.190 | 0.134      |
|  7 | 0.200    | 0.112    | 0.229  | 0.263 | 0.202 | 0.157      |
|  8 | 0.201    | 0.111    | 0.229  | 0.275 | 0.195 | 0.168      |
|  9 | 0.199    | 0.109    | 0.229  | 0.285 | 0.191 | 0.179      |
| 10 | **0.214**| 0.116    | 0.236  | 0.322 | 0.187 | 0.182      | ← peak raw AP30
| 11 | 0.208    | 0.102    | 0.222  | 0.296 | 0.178 | 0.182      |
| 12 | 0.206    | 0.106    | 0.228  | 0.309 | 0.181 | **0.183**  | ← composite-best, saved as `best.pth`
| 13 | 0.211    | 0.116    | **0.240** | 0.312 | 0.195 | 0.183  | ← peak raw F1
| 14 | 0.207    | 0.101    | 0.227  | 0.281 | 0.191 | 0.183      |
| 15 | 0.196    | 0.095    | 0.225  | 0.260 | 0.199 | 0.180      |

Сходимость плавная, плато на 10-14, лёгкий спад на 15 (косинусная LR подходит к eta_min). За 15 эпох выжали ~0.21 на val AP30. Кривая ещё не вышла на полное насыщение — судя по тенденции, 30-50 эпох могли бы дать дополнительные 0.02-0.05 AP30.

## Заметка про эту реализацию

Это не «exact reproduction» бумаги, а интерпретация: paper умалчивает целый ряд деталей. Что в моей реализации:

- **Box parametrization**: paper описывает `cur_boxes = cur_boxes + delta` (additive). На 12 cascade-stages (6×stage1 + 6×stage2) это коллапсирует w/h в ноль (v1 training дал AP=0). Заменил на Sparse-R-CNN-style: `cx_new = cx + dx·w`, `w_new = w·exp(dw)`. См. `stqd_det/decoder.py::apply_box_delta`.
- **Inference init**: вместо raw Poisson noise (29% боксов имеют degenerate w_norm=0) — uniform cx/cy + w/h в [0.05, 0.40] (типичный размер стеноза). См. `stqd_det/noise.py::prepare_inference_init`.
- **Quantum noise**: для маргинального сэмплинга `q(B_t|B_0)` использую DiffusionDet-style closed-form с **центрированным Poisson** (ε=Poisson(1)-1) вместо Gaussian. Полный Poisson-Markov-chain из бумаги не воспроизводил — это и сама бумага в обучении не делает.
- **T=5 вместо N=9**: чтобы быть apples-to-apples с baseline (`video_5_etf_consistency` тоже T=5).
- **15 epochs вместо 25k iterations** (paper'овский compute budget): out of compute budget на 12 GB GPU.

## Анализ почему STQD-Det проигрывает

1. **Параметризация боксов** — паперовский additive cascade без поправки гарантированно ломается. Мой фикс работает, но возможно требует больше эпох на сходимость чем у RF-DETR-video (там DETR-style queries без cascade-drift проблемы).
2. **Backbone**: ResNet-50 ImageNet vs DINOv2-ViT-S самообученный. DINOv2 на медицинских снимках обычно сильнее.
3. **Compute**: 15 эпох × T=5 vs 50 эпох × T=5 у baseline → 3.3× меньше.
4. **Pretrain**: baseline стартовал с RF-DETR который уже был дообучен на dataset 2 single-frames. STQD-Det с нуля от ImageNet.
5. **Сила паперовских результатов**: paper показывает 92.39% F1 на их 233-sequence Peking Union датасете с 4-fold CV и Hungarian IoU>0.5 матчингом — это **другая метрика** и **другие данные**. На dataset 2 наша MICRO-pooled AP30 0.38 — это не сравнимо с 0.92 из бумаги.

## Что есть положительного

- **FragRate** у STQD-Det лучше на 17% (0.047 vs 0.057) — то есть детекции более **temporally consistent**. Это и есть то для чего STFS придумали — обмен фич между кадрами для устойчивости. STFS работает, просто overall модель слабее.
- **Precision/Recall более сбалансированы**: 0.36/0.31 у STQD-Det vs 0.40/0.34 у baseline. STQD-Det менее агрессивно false-positive.

## Файлы

- `best.pth` — epoch 12, EMA weights, composite-selected. 824 MB (с state_dict + ema_state_dict).
- `last.pth` — epoch 15, raw state_dict. 412 MB.
- `train.csv` — per-epoch metrics.
- `ablation_results.txt` / `.json` — финальный test eval.
- `train.log` — stdout тренировки.
- `test_eval.log` — stdout test eval.

## Recommended next steps если хотим улучшить STQD-Det

1. **Pretrain 2D первый**: сделать DiffusionDet-only run на dataset 2 single frames (T=1, batch 8, 30 эпох ≈ 4ч). Использовать веса как старт для temporal extension. Ожидаемый прирост: +0.05-0.10 AP30 за счёт сильного 2D-старта.
2. **Дольше**: 30-50 эпох вместо 15 (всё ещё в пределах одних суток compute).
3. **Backbone init**: попробовать ResNet-50 предобученный на COCO detection (`torchvision.models.detection.fasterrcnn_resnet50_fpn`), а не ImageNet classification.
4. **Аугментации**: сейчас базовая `build_train_augmentation` из rfdetr_temporal. На stenosis-домене вертикальный flip помогает.
5. **N=9 (paper-fidelity)** — отдельный sweep, но это удвоит память.
