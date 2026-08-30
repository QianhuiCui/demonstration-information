#!/bin/bash

TRAJ_ROOT="/home/qianhui/Downloads/trajectories_symdex"
OUTPUT_DIR="path/to/output/dir"

source /path/to/miniconda3/bin/activate
conda activate openx

cd path/to/rlds/symdex
mkdir -p "$OUTPUT_DIR"

# Task list comes straight from tasks.yaml -- add a task there, not here.
TASKS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('tasks.yaml')).keys()))")

for TASK in $TASKS; do
    echo "Starting ${TASK}"
    tfds build --config=${TASK} --manual_dir ${TRAJ_ROOT}/${TASK} --data_dir ${OUTPUT_DIR}/${TASK}

    # Now move the files to the correct location.
    # NOTE: "sym_dex" is tfds's snake_case of the `SymDex` builder class name (matches the
    # RoboMimic -> robo_mimic pattern below in git history) -- verify this against the actual
    # folder tfds creates on your first real run before trusting this in a batch script.
    mv ${OUTPUT_DIR}/${TASK}/sym_dex/${TASK}/1.0.0 ${OUTPUT_DIR}/${TASK}
    rm -r ${OUTPUT_DIR}/${TASK}/sym_dex
    rm -r ${OUTPUT_DIR}/${TASK}/downloads
done

echo "Processing complete."
