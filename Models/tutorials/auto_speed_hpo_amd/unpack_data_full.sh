#!/bin/bash
# Version for the *full* OpenLane dataset (not just a single split)
set -euo pipefail

usage() {
    echo "Usage: $0 <data-source-dir> <data-store-dir> [save_numpy_arrays:0|1]"
}

if [[ "${1:-}" = "-h" || "${1:-}" = "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage
    exit 1
fi

DATA_SOURCE="$1"
DATA_STORE="$2"
SAVE_NUMPY_ARRAYS="${3:-0}"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ ! -d "$DATA_SOURCE" ]]; then
    echo "[ERROR] Data source directory not found: $DATA_SOURCE" >&2
    exit 1
fi

mkdir -p "$DATA_STORE"

if [[ -d "$DATA_STORE/images/train" && -d "$DATA_STORE/images/val" ]]; then
    echo "Converted data already exists: skipping extraction and conversion"
else
    echo "Extracting data..."
    # Put the OpenLane images and labels into the directories expected by the converter tool
    if [[ ! -d "$DATA_STORE/cipo" ]]; then
        echo "  cipo.zip"
        unzip -q "$DATA_SOURCE/cipo.zip" -d "$DATA_STORE"
    fi
    if [[ ! -d "$DATA_STORE/scene" ]]; then
        echo "  scene.zip"
        unzip -q "$DATA_SOURCE/scene.zip" -d "$DATA_STORE"
    fi
    if [[ ! -d "$DATA_STORE/images/training" ]]; then
        mkdir -p "$DATA_STORE/raw_training"
        mkdir -p "$DATA_STORE/images/training"
        shopt -s nullglob
        training_tars=("$DATA_SOURCE"/images_training_*.tar)
        if (( ${#training_tars[@]} == 0 )); then
            echo "[ERROR] No images_training_*.tar found in $DATA_SOURCE" >&2
            exit 1
        fi
        for image_tar in "${training_tars[@]}"; do
            image_tar_name="$(basename "$image_tar")"
            echo "  $image_tar_name"
            tar -xf "$image_tar" -C "$DATA_STORE/raw_training/"
            # The tar contains a single dir with the same name as the archive
            subdir="${image_tar_name%.tar}"
            mv "$DATA_STORE/raw_training/$subdir/"* "$DATA_STORE/images/training/"
            rmdir "$DATA_STORE/raw_training/$subdir"
        done
        shopt -u nullglob
        rmdir "$DATA_STORE/raw_training"
    fi
    if [[ ! -d "$DATA_STORE/images/validation" ]]; then
        mkdir -p "$DATA_STORE/raw_validation"
        mkdir -p "$DATA_STORE/images/validation"
        shopt -s nullglob
        validation_tars=("$DATA_SOURCE"/images_validation_*.tar)
        if (( ${#validation_tars[@]} == 0 )); then
            echo "[ERROR] No images_validation_*.tar found in $DATA_SOURCE" >&2
            exit 1
        fi
        for image_tar in "${validation_tars[@]}"; do
            image_tar_name="$(basename "$image_tar")"
            echo "  $image_tar_name"
            tar -xf "$image_tar" -C "$DATA_STORE/raw_validation/"
            # The tar contains a single dir with the same name as the archive
            subdir="${image_tar_name%.tar}"
            mv "$DATA_STORE/raw_validation/$subdir/"* "$DATA_STORE/images/validation/"
            rmdir "$DATA_STORE/raw_validation/$subdir"
        done
        shopt -u nullglob
        rmdir "$DATA_STORE/raw_validation"
    fi
    if [[ ! -d "$DATA_STORE/labels/training" ]]; then
        mkdir -p "$DATA_STORE/labels/training"
        cp -r "$DATA_STORE/cipo/"* "$DATA_STORE/labels/training"
    fi
    if [[ ! -d "$DATA_STORE/labels/validation" ]]; then
        mkdir -p "$DATA_STORE/labels/validation"
        cp -r "$DATA_STORE/cipo/"* "$DATA_STORE/labels/validation"
    fi

    # Reorganise the data into the form the training script requires
    echo "Converting data"
    CONVERTER_ARGS=(-i "$DATA_STORE" -o "$DATA_STORE")
    if [ "$SAVE_NUMPY_ARRAYS" = "1" ]; then
        CONVERTER_ARGS+=(--save-numpy-arrays)
    fi
    python3 "$SCRIPT_DIR/../../data_parsing/OpenLane/converter.py" "${CONVERTER_ARGS[@]}"
fi

# Make sure there are no old caches left
rm -f "$DATA_STORE/images/train.cache"
rm -f "$DATA_STORE/images/val.cache"
rm -f "$DATA_STORE/images/training_preprocessed.cache"
rm -f "$DATA_STORE/images/train_hpo.cache"
rm -f "$DATA_STORE/images/train_hpo_preprocessed.cache"
rm -f "$DATA_STORE/images/val_hpo.cache"
