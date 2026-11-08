#!/usr/bin/env bash
# One task:   bash bicoord_metrics/run_bicoord.sh pickObject
# All tasks:  bash bicoord_metrics/run_bicoord.sh
set -euo pipefail

DATA_ROOT=/home/qianhui/trajectories_symdex
ALL_TASKS=insertDrawer,pickObject,pouring,threading
TASKS="${1:-$ALL_TASKS}"

python bicoord_metrics/bicoord_metrics.py \
  --data_dir "$DATA_ROOT" \
  --task "$TASKS" \
  --plot_dir ./bicoord_metrics/plots
