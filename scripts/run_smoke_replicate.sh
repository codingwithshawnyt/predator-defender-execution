#!/usr/bin/env bash
# Run Lillo-Macri smoke replicate on Lambda.
# Usage: bash scripts/run_smoke_replicate.sh
#
# Prerequisites:
#   - uv installed and on PATH
#   - vendor/pymarketsim fixes applied (see docs/decisions.md Fix 7-11)
#   - uv sync --extra dev completed

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

uv run python -m pde.training.smoke_replicate \
    --n-steps 100 \
    --q-0 100.0 \
    --n-bg 50 \
    --alpha 0.002 \
    --n-volume-bins 21 \
    --episodes 5000 \
    --lr 1e-4 \
    --batch-size 64 \
    --buffer-size 15000 \
    --gamma 1.0 \
    --seed 0 \
    --device cuda \
    --log-dir artifacts/logs/smoke_replicate

uv run python -c "from pde.figures.fig01_smoke_replicate import plot; plot()"
