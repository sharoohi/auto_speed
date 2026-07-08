#!/bin/bash
# Run an experiment to measure the throughput of parallel training
# when using GPU partitioning. Partitioning mode should already have
# been set before running this
set -euo pipefail

usage() {
  echo "Usage: $0 [trainer-args ...]"
  echo "  Optional trailing args are passed through to auto_speed_trainer.py"
}

if [[ "${1:-}" = "-h" || "${1:-}" = "--help" ]]; then
  usage
  exit 0
fi

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
PREPROCESSED_DIR="$DATASET_PATH/images/train_preprocessed"
#################################################################

mkdir -p "$BASE_RUNS_DIR"

# Avoid CPU oversubscription: limit CPU threads used by math runtimes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -n "${AUTOSPEED_BS:-}" ]];
then
  BATCH_SIZES="$AUTOSPEED_BS"
  echo "Running for batch sizes: $BATCH_SIZES"
else
  BATCH_SIZES="8 16 32 64 128 256 512 1024"
  echo "Running all batch sizes ($BATCH_SIZES)"
fi

# If the preprocessed data is already available, skip all data prep
if [[ -d "$PREPROCESSED_DIR" ]]; then
  echo "Skipping unpacking and pre-processing, as $PREPROCESSED_DIR already exists"
else
  # Unpack data and preprocess
  # This produces both train/val, train_hpo/val_hpo, train_preprocessed/train_hpo_preprocessed, which we will use
  "$REPO_DIR/Models/tutorials/auto_speed_hpo_amd/unpack_data.sh" "$RAW_DATASET_PATH" "$DATASET_PATH" 1
fi

# Check how many GPUs are available and use this to work out the partitioning mode, assuming a node with 8 GPUs total
num_gpus=$(python3 -c 'import torch; print(torch.cuda.device_count())')
if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || (( num_gpus < 1 )); then
  echo "[ERROR] Invalid GPU count from torch: $num_gpus" >&2
  exit 1
fi

# Decide whether to use fp32/AMP based on partitioning mode
# For QPX (32) & CPX (63), use AMP
if (( num_gpus == 8 )) || (( num_gpus == 16 )); then
  FP_PARAM=()
  echo "Partitioning mode SPX|DPX -> default AMP (no --fp32)"
elif (( num_gpus == 32 )) || (( num_gpus == 63 )) || (( num_gpus == 64 )); then
  FP_PARAM=("--fp32")
  echo "Partitioning mode QPX|CPX -> use --fp32"
elif (( num_gpus == 1 )); then
  # Allow single-GPU running just for testing
  FP_PARAM=()
  echo "[WARNING] TESTING: only 1 GPU is available, this is not a real experiment run!"
else
  echo "[ERROR] Got unexpected number of GPUs: $num_gpus. Expected 8, 16, 32 or 63"
  exit 1
fi

LOGFILE="$BASE_RUNS_DIR/job_throughput_${num_gpus}.log"
echo "Logging to $LOGFILE"
touch "$LOGFILE"
# This first write with overwrite an existing file
echo "[INFO] Job started at: $(date)" > "$LOGFILE"

for batch_size in $BATCH_SIZES; do
  echo "=== Batch size $batch_size ==="
  echo "[INFO] Starting batch_size=$batch_size at $(date)" >> "$LOGFILE"
  echo "[INFO] Detected num_gpus=$num_gpus" >> "$LOGFILE"

  pids=()
  for ((gpu_id=0; gpu_id<num_gpus; gpu_id++)); do
    # Directory to output all run output and logfile to
    run_output_dir="$BASE_RUNS_DIR/partition_${num_gpus}/bs_${batch_size}/gpu_${gpu_id}"
    mkdir -p "$run_output_dir"
    job_log_file="${run_output_dir}/console.log"
    
    echo "[INFO] Launching trainer: batch_size=$batch_size gpu_id=$gpu_id" >> "$LOGFILE"
    echo "[INFO] Training logs: $job_log_file" >> "$LOGFILE"

    # Run the job for this GPU
    AMD_VISIBLE_DEVICES="$gpu_id" CUDA_VISIBLE_DEVICES="$gpu_id" \
      python3 "$REPO_DIR/Models/training/auto_speed_trainer.py" \
      --dataset "$DATASET_PATH" \
      --epochs 5 \
      --val_batch_size 64 \
      --batch-size "$batch_size" \
      --runs_dir "${run_output_dir}" \
      --use_preprocessed \
      --do_compile \
      "${FP_PARAM[@]}" "$@" \
      --config "$DEFAULT_CONFIG" >> "${job_log_file}" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "[ERROR] One or more trainers failed for batch_size=$batch_size" >> "$LOGFILE"
    exit 1
  fi

  # Don't sum over all runs, but only the *latest* run for each GPU
  logs=()
  for gpu_dir in "$BASE_RUNS_DIR/partition_${num_gpus}/bs_${batch_size}"/gpu_*/ ; do
    [[ -d "$gpu_dir" ]] || continue
    run_dir=$(ls -1 -d "${gpu_dir}"run-*/ 2>/dev/null | tail -n 1 || true)
    if [[ -n "$run_dir" ]]; then
      logs+=("${run_dir}training.log")
    fi
  done
  if (( ${#logs[@]} == 0 )); then
    compute_sum="NA"
    workload_sum="NA"
  else
    compute_sum=$(awk -F': ' '/Overall Average Training Throughput \(compute-only FPS\)/ {sum+=$NF} END {if (sum>0) printf "%.2f", sum; else print "NA"}' "${logs[@]}")
    workload_sum=$(awk -F': ' '/Overall Average Training Throughput \(full workload FPS\)/ {sum+=$NF} END {if (sum>0) printf "%.2f", sum; else print "NA"}' "${logs[@]}")
  fi

  echo "[RESULT] batch_size=$batch_size sum_compute_fps=$compute_sum sum_full_workload_fps=$workload_sum" | tee -a "$LOGFILE"
done

echo "[INFO] Job completed at: $(date)" >> "$LOGFILE"