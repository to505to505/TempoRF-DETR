#!/usr/bin/env bash
# Launch the full STQD-Det run on dataset 2 (matches plan Task 14).
#
# Output: rfdetr_video/runs/stqd_det_T5_dataset2/
#   best.pth, last.pth, train.csv, history.json, best.txt, config.json
#
# Eval after training:
#   python _eval_stfs_ablations.py stqd_det_T5_dataset2 --model-type stqd_det
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${1:-}" ]] || [[ "${1:-}" == -* ]]; then
    RUN_NAME="stqd_det_T5_dataset2"
else
    RUN_NAME="$1"
    shift
fi

exec python -m stqd_det.train \
    --run-name "$RUN_NAME" \
    --output-dir rfdetr_video/runs \
    --data-root data/dataset2_split \
    --T 5 \
    --img-size 512 \
    --num-proposals 300 \
    --batch-size 1 \
    --grad-accum-steps 4 \
    --num-workers 4 \
    --epochs 50 \
    --eval-interval 2 \
    --lr 2.5e-5 \
    --weight-decay 1e-4 \
    --lr-schedule cosine \
    --warmup-iters 500 \
    --diffusion-T-steps 1000 \
    --consistency-weight 1.0 \
    "$@"
