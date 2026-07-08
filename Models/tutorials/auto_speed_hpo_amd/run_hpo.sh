#!/bin/bash
set -euo pipefail

usage() {
  echo
  echo "Usage: $0 [cpx|spx] <config-name> [tune-parameters]"
  echo "  cpx or spx is the partitioning mode you are using"
  echo "  <config-name> points to a YAML config in tutorials/auto_speed_hpo_amd/hpo_config/"
  echo "  Additional parameters (e.g. --epochs and --trials) are passed through to tune_autospeed_optuna.py"
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Base dir of repo
REPO_DIR=$(readlink -e "$SCRIPT_DIR/../../..")

#################################################################
# Paths (edit as needed)
# This is run inside a Docker container and these are the mountpoints we use for various directories
RAW_DATASET_PATH="/app/auto_speed/raw_data"
DATASET_PATH="/app/auto_speed/data"
HPO_CONFIG_DIR="$REPO_DIR/Models/tutorials/auto_speed_hpo_amd/hpo_config"
DEFAULT_CONFIG="$REPO_DIR/Models/config/auto_speed.yaml"
# Output will be stored in subdirectories of this dir
BASE_RUNS_DIR="$REPO_DIR/Models/training/runs"
#################################################################

LOGFILE="/app/auto_speed/Models/training/runs/hpo_run.log"
mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

if [[ "${1:-}" = "-h" || "${1:-}" = "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

MODE_ARG=$(echo "$1" | tr '[:lower:]' '[:upper:]')
if [ "$MODE_ARG" != "CPX" ] && [ "$MODE_ARG" != "SPX" ]; then
  echo "Invalid mode: $1. Use 'cpx' or 'spx'."
  exit 1
fi
PARTITION_MODE="$MODE_ARG"

EXP_NAME=$2
HPO_RANGES_PATH="$HPO_CONFIG_DIR/$EXP_NAME.yaml"
RUNS_DIR="$BASE_RUNS_DIR/$EXP_NAME"
# Remaining parameters will be passed to tuning script
shift 2

if [[ ! -f "$HPO_RANGES_PATH" ]]; then
  echo "[ERROR] HPO config not found: $HPO_RANGES_PATH" >&2
  exit 1
fi

# Avoid CPU oversubscription: limit CPU threads used by math runtimes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Location of data after unpacking and converting
HPO_DATA_DIR="$DATASET_PATH/images/train_hpo"
# Location of data after preprocessing
PREPROCESSED_DIR=${HPO_DATA_DIR}_preprocessed

# If the preprocessed data is already available, skip all data prep
if [[ -d "$PREPROCESSED_DIR" ]]; then
  echo "Skipping unpacking and pre-processing, as $PREPROCESSED_DIR already exists"
else
  # Unpack data and preprocess
  # This produces both train/val, train_hpo/val_hpo, train_preprocessed/train_hpo_preprocessed, which we will use
  "$REPO_DIR/Models/tutorials/auto_speed_hpo_amd/unpack_data.sh" "$RAW_DATASET_PATH" "$DATASET_PATH" 1
fi

echo "Running Optuna HPO for batch size 32 on $PARTITION_MODE mode"
echo "Pre-processed data: $PREPROCESSED_DIR"
echo "Additional parameters for tune_autospeed_optuna.py: $*"

if [[ "$PARTITION_MODE" = "CPX" ]]; then
  # If in CPX mode, always use --fp32
  FP_PARAM=("--fp32")
else
  FP_PARAM=()
fi

set -x
PYTHONPATH="$REPO_DIR/Models/training${PYTHONPATH:+:$PYTHONPATH}" \
python3 "$REPO_DIR/Models/hyperparam_optim/tune_autospeed_optuna.py" \
  --dataset "$DATASET_PATH" \
  --hpo_ranges "$HPO_RANGES_PATH" \
  --runs_dir "$RUNS_DIR/optuna_${PARTITION_MODE}_bs32" \
  --best_params_output "$RUNS_DIR/optuna_${PARTITION_MODE}_bs32/best_params.yaml" \
  --do_compile \
  --default_config "$DEFAULT_CONFIG" \
  "${FP_PARAM[@]}" "$@"
set +x
