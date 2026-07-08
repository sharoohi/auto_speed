#!/bin/bash
#
# Run five different variants of the model training to compare training
# throughput under different optimizations and output results to subdirectories.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Base dir of repo
REPO_DIR=$(readlink -e "$SCRIPT_DIR/../../..")

#################################################################
# Paths (edit as needed)
# This is run inside a Docker container and these are the mountpoints we use for various directories
RAW_DATASET_PATH="/app/auto_speed/raw_data"
DATASET_PATH="/app/auto_speed/data"
DEFAULT_CONFIG="$REPO_DIR/Models/config/auto_speed.yaml"
# Output will be stored in subdirectories of this dir
BASE_RUNS_DIR="$REPO_DIR/Models/training/runs"
#################################################################

# Location of data after unpacking and converting
DATA_DIR="$DATASET_PATH/images/train"
# Location of data after preprocessing
PREPROCESSED_DIR=${DATA_DIR}_preprocessed

# If the preprocessed data is already available, skip all data prep
if [[ -d "$PREPROCESSED_DIR" ]]; then
    echo "Skipping unpacking and pre-processing, as $PREPROCESSED_DIR already exists"
else
    # Unpack data and preprocess
    # This produces both train/val, train_hpo/val_hpo, train_preprocessed/train_hpo_preprocessed, which we will use
    "$REPO_DIR/Models/tutorials/auto_speed_hpo_amd/unpack_data.sh" "$RAW_DATASET_PATH" "$DATASET_PATH" 1
fi

TRAINER="\
    python3 $REPO_DIR/Models/training/auto_speed_trainer.py \
        --dataset $DATASET_PATH \
        --config $DEFAULT_CONFIG \
        --epochs 5 \
        --runs_dir $BASE_RUNS_DIR"

mkdir -p "$BASE_RUNS_DIR"

# ---------------------------------------------------------------------------
# Helper: run one training variant
#   $1 = output_subdir name (used to locate training.log afterwards)
#   $2 = extra trainer flags
# ---------------------------------------------------------------------------
run_variant() {
    local subdir="$1"
    local extra_flags="$2"
    echo "--- Training: $subdir ---"
    $TRAINER $extra_flags --output_subdir "$subdir"
}

# ---------------------------------------------------------------------------
# Run each optimisation variant
# ---------------------------------------------------------------------------
echo "=== Run 1 - Original (no vectorized loss) ==="
run_variant "1_original" "--disable_vectorized_loss"

echo "=== Run 2 - Vectorized loss ==="
run_variant "2_vectorized_loss" ""

echo "=== Run 3 - Pre-processing (no vectorized loss) ==="
run_variant "3_preprocessing" "--disable_vectorized_loss --use_preprocessed"

echo "=== Run 4 - torch.compile (no vectorized loss) ==="
run_variant "4_torch_compile" "--disable_vectorized_loss --do_compile"

echo "=== Run 5 - All optimisations ==="
run_variant "5_all" "--do_compile --use_preprocessed"

# ---------------------------------------------------------------------------
# Parse results and print the comparison table
# ---------------------------------------------------------------------------
echo ""
echo "=== Results ==="

# extract_fps <log_file> <metric>
#   metric: "compute-only" or "full workload"
extract_fps() {
    local log="$1"
    local metric="$2"
    grep -oP "${metric} FPS\)[^:]*: \K\d+\.\d+" "$log" 2>/dev/null | tail -1 || echo "N/A"
}

declare -A COMPUTE_FPS
declare -A WORKLOAD_FPS

VARIANTS=(
    "1_original:Original"
    "2_vectorized_loss:Vectorized loss"
    "3_preprocessing:Pre-processing"
    "4_torch_compile:torch.compile"
    "5_all:All"
)

for entry in "${VARIANTS[@]}"; do
    subdir="${entry%%:*}"
    label="${entry#*:}"
    log="$BASE_RUNS_DIR/$subdir/training.log"
    if [[ -f "$log" ]]; then
        COMPUTE_FPS["$label"]=$(extract_fps "$log" "compute-only")
        WORKLOAD_FPS["$label"]=$(extract_fps "$log" "full workload")
    else
        COMPUTE_FPS["$label"]="N/A"
        WORKLOAD_FPS["$label"]="N/A"
    fi
done

# Print Markdown table
printf "\n| %-20s | %-30s | %-25s |\n" "**Run**" "**Compute throughput (FPS)**" "**Full throughput (FPS)**"
printf "| %-20s | %-30s | %-25s |\n" "$(printf '%0.s-' {1..20})" "$(printf '%0.s-' {1..30})" "$(printf '%0.s-' {1..25})"
for entry in "${VARIANTS[@]}"; do
    label="${entry#*:}"
    printf "| %-20s | %-30s | %-25s |\n" "$label" "${COMPUTE_FPS[$label]}" "${WORKLOAD_FPS[$label]}"
done
echo ""
