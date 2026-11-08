#!/bin/bash
set -e  # stop at the first failed build

TRAJ_ROOT="/home/qianhui/trajectories_symdex"
OUTPUT_DIR="/home/qianhui/deminf/symdex_rlds"

source /home/qianhui/miniconda3/bin/activate
conda activate openx

cd /home/qianhui/demonstration-information/rlds/symdex
TASKS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('tasks.yaml')).keys()))")

for TASK in $TASKS; do
    echo "Starting ${TASK}"
    tfds build --config=${TASK} --manual_dir=${TRAJ_ROOT}/${TASK} --data_dir=${OUTPUT_DIR}
done

echo "Processing complete."
