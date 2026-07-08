#!/bin/bash
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
    if [[ ! -d "$DATA_STORE/images_training_0" ]]; then
        echo "  images_training_0.tar"
        tar -xf "$DATA_SOURCE/images_training_0.tar" -C "$DATA_STORE/"
    fi
    if [[ ! -d "$DATA_STORE/images_validation_0" ]]; then
        echo "  images_validation_0.tar"
        tar -xf "$DATA_SOURCE/images_validation_0.tar" -C "$DATA_STORE/"
    fi
    if [[ ! -d "$DATA_STORE/images/training" ]]; then
        echo "  Use training_0 as training data"
        mkdir -p "$DATA_STORE/images/training"
        cp -r "$DATA_STORE/images_training_0/"* "$DATA_STORE/images/training/"
    fi
    if [[ ! -d "$DATA_STORE/labels/training" ]]; then
        echo "  Use training_0 as training labels"
        mkdir -p "$DATA_STORE/labels/training"
        cp -r "$DATA_STORE/cipo/"* "$DATA_STORE/labels/training"
    fi
    if [[ ! -d "$DATA_STORE/images/validation" ]]; then
        echo "  Use validation_0 as eval data"
        mkdir -p "$DATA_STORE/images/validation"
        cp -r "$DATA_STORE/images_validation_0/"* "$DATA_STORE/images/validation/"
    fi
    if [[ ! -d "$DATA_STORE/labels/validation" ]]; then
        echo "  Use validation_0 as eval labels"
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
