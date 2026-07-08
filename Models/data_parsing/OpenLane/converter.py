from functools import partial
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm  # pip install tqdm
import json
import random
import numpy as np
from PIL import Image
import cv2
from multiprocessing import Pool

orig_image_width = 1920
orig_image_height = 1280
new_image_width = 1024
new_image_height = 512


def resample():
    choices = (
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LINEAR,
        cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4,
    )
    return random.choice(seq=choices)


def decode_and_resize(filename: Path, input_width: int, input_height: int, augment: bool = True):
    data = cv2.imread(filename.as_posix())
    if data is None:
        raise FileNotFoundError(f"Failed to decode image: {filename}")

    h, w = data.shape[:2]
    ratio = min(input_height / h, input_width / w)
    if ratio != 1:
        data = cv2.resize(
            data,
            dsize=(int(w * ratio), int(h * ratio)),
            interpolation=resample() if augment else cv2.INTER_LINEAR,
        )
    return data


def decode_and_save_array(
    filename: Path,
    save_dir: Path,
    input_width: int,
    input_height: int,
    augment: bool = True,
):
    data = decode_and_resize(filename, input_width=input_width, input_height=input_height, augment=augment)
    np.save(save_dir / f"{filename.stem}.npy", data)
    
def save_split_as_numpy_arrays(
    images_dir: Path,
    input_width: int,
    input_height: int,
    workers: int = 16,
    augment: bool = True,
):
    files = [f for f in images_dir.iterdir() if f.is_file()]
    if not files:
        print(f"Skipping numpy export for {images_dir}: no files found")
        return

    np_datadir = images_dir.parent / f"{images_dir.name}_preprocessed"
    if np_datadir.exists():
        shutil.rmtree(np_datadir)
    np_datadir.mkdir(parents=True, exist_ok=True)
    print(f"Saving preprocessed data to: {np_datadir}")

    work_func = partial(
        decode_and_save_array,
        save_dir=np_datadir,
        input_width=input_width,
        input_height=input_height,
        augment=augment,
    )
    with Pool(processes=workers) as pool:
        list(tqdm(pool.imap_unordered(work_func, files), total=len(files), desc=f"Preprocess {images_dir.name}"))

def move_images(input_dir, output_dir):
    input_dir = Path(input_dir)

    files = [f for f in input_dir.rglob("*") if f.is_file()]

    for file in tqdm(files, desc="Moving files", unit="file"):
        if file.is_file():
            target = Path(output_dir) / file.name
            target.parent.mkdir(parents=True, exist_ok=True)

            # If file with same name exists, rename
            if target.exists():
                print(f"File with same name already exists: {target}")

            # Move file
            shutil.move(str(file), str(target))

    if input_dir.exists() and input_dir.is_dir():
        shutil.rmtree(input_dir)


def process_images(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = [f for f in input_dir.rglob("*") if f.is_file()]

    for file in tqdm(files, desc="Cropping files", unit="file"):
        try:
            with Image.open(file) as img:
                width, height = img.size
                # Crop: (left, upper, right, lower)
                cropped = img.crop((0, 320, width, height))
                # Resize
                resized = cropped.resize((new_image_width, new_image_height), Image.LANCZOS)

                # Save to output directory
                target = output_dir / file.name
                target.parent.mkdir(parents=True, exist_ok=True)

                # If file with same name exists, rename
                if target.exists():
                    print(f"File with same name already exists: {target}")
                    # optional: add suffix
                    stem, ext = file.stem, file.suffix
                    target = output_dir / f"{stem}_cropped{ext}"

                resized.save(target)

        except Exception as e:
            print(f"Failed to process {file}: {e}")

    # Optionally, delete the original input directory
    if input_dir.exists() and input_dir.is_dir():
        shutil.rmtree(input_dir)


def convert_labels(input_dir, output_dir):
    crop_top = 320

    input_dir = Path(input_dir)
    files = [f for f in input_dir.rglob("*") if f.is_file()]

    new_height = orig_image_height - crop_top

    for file in tqdm(files, desc="Convert labels", unit="file"):
        base_name = file.name.split(".", 1)[0]
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            labels = []

            for box in data["result"]:
                id = box.get("id", box.get("attribute"))
                id = 3 if str(id) == "4" else id

                # Adjust y because top was cropped
                y_top = float(box["y"]) - crop_top

                # Skip boxes fully removed by crop
                if y_top + float(box["height"]) <= 0:
                    continue

                # Clamp to image
                y_top = max(0, y_top)

                # Normalize
                width = float(box["width"]) / orig_image_width
                height = float(box["height"]) / new_height
                x = (float(box["x"]) + float(box["width"]) / 2) / orig_image_width
                y = (y_top + float(box["height"]) / 2) / new_height

                labels.append([id, x, y, width, height])

        target = Path(output_dir) / f"{base_name}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("w", encoding="utf-8") as f:
            for item in labels:
                f.write(" ".join(map(str, item)) + "\n")

        file.unlink()

    shutil.rmtree(input_dir)


def convert_lane3d_labels(input_dir, output_dir):
    input_dir = Path(input_dir)
    files = [f for f in input_dir.rglob("*") if f.is_file()]

    for file in tqdm(files, desc="Convert 3dlane labels", unit="file"):
        base_name = file.name.split(".", 1)[0]
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

            for lane_line in data['lane_lines']:
                # Convert to Nx2 points
                pts = np.array(lane_line['uv']).T.astype(np.float32)  # shape: (N, 2)
                # Shift y coordinates (v axis)
                pts[:, 1] -= orig_image_height - (orig_image_width / 2)
                # Scale x and y
                pts *= new_image_width / orig_image_width
                # Convert back to original format (2, N)
                lane_line['uv'] = pts.T.tolist()

            target = Path(output_dir) / f"{base_name}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)

            with target.open("w", encoding="utf-8") as f:
                json.dump(data, f)

        file.unlink()

    shutil.rmtree(input_dir)


def expand_training_set_and_split_for_hpo(dataset_dir, fract=0.25, val_fraction=0.2):
    """
    Expand the training set by moving 75% of validation samples into training,
    keeping 25% in val as a held-out test set, then create HPO-specific train/val
    splits (train_hpo / val_hpo) by reserving 20% of the expanded training set
    for val_hpo as HPO-specific validation set.
    """
    dataset_dir = Path(dataset_dir)
    val_images_dir = dataset_dir / "images" / "val"
    val_labels_dir = dataset_dir / "labels" / "val"
    train_images_dir = dataset_dir / "images" / "train"
    train_labels_dir = dataset_dir / "labels" / "train"
    train_hpo_images_dir = dataset_dir / "images" / "train_hpo"
    train_hpo_labels_dir = dataset_dir / "labels" / "train_hpo"
    val_hpo_images_dir = dataset_dir / "images" / "val_hpo"
    val_hpo_labels_dir = dataset_dir / "labels" / "val_hpo"

    val_images = [f for f in val_images_dir.rglob("*") if f.is_file()]
    random.shuffle(val_images)

    split_idx = int(len(val_images) * fract)
    # val_files = files[:split_idx]
    train_images = val_images[split_idx:]

    for image in tqdm(train_images, desc="Expand training dataset", unit="file"):
        if image.is_file():
            target_image = train_images_dir / image.name
            label = val_labels_dir / f"{image.stem}.txt"
            target_label = train_labels_dir / f"{image.stem}.txt"

            # Move file
            try:
                shutil.move(str(image), str(target_image))
                shutil.move(str(label), str(target_label))
            except Exception as e:
                print(f"Failed to move {image.name}: {e}")

    print("Creating HPO train/val split")

    if train_hpo_images_dir.exists():
        shutil.rmtree(train_hpo_images_dir)
    if train_hpo_labels_dir.exists():
        shutil.rmtree(train_hpo_labels_dir)
    train_hpo_images_dir.mkdir(parents=True, exist_ok=True)
    train_hpo_labels_dir.mkdir(parents=True, exist_ok=True)
    for image in train_images_dir.glob("*"):
        if image.is_file():
            shutil.copy(str(image), str(train_hpo_images_dir / image.name))
    for label in train_labels_dir.glob("*"):
        if label.is_file():
            shutil.copy(str(label), str(train_hpo_labels_dir / label.name))

    print(f"HPO train set in\n- {train_hpo_images_dir}\n- {train_hpo_labels_dir}")

    all_train_hpo_images = [f for f in train_hpo_images_dir.glob("*") if f.is_file()]
    random.shuffle(all_train_hpo_images)
    split_idx = int(len(all_train_hpo_images) * (1 - val_fraction))
    new_val_hpo_images = all_train_hpo_images[split_idx:]

    if val_hpo_images_dir.exists():
        shutil.rmtree(val_hpo_images_dir)
    if val_hpo_labels_dir.exists():
        shutil.rmtree(val_hpo_labels_dir)
    val_hpo_images_dir.mkdir(parents=True, exist_ok=True)
    val_hpo_labels_dir.mkdir(parents=True, exist_ok=True)

    for image in new_val_hpo_images:
        label = train_hpo_labels_dir / f"{image.stem}.txt"
        shutil.move(str(image), str(val_hpo_images_dir / image.name))
        if label.exists():
            shutil.move(str(label), str(val_hpo_labels_dir / label.name))

    print(f"HPO validation set in\n- {val_hpo_images_dir}\n- {val_hpo_labels_dir}")


def convert(input_ds_dir, output_ds_dir):
    # convert training data
    input_training_dir = input_ds_dir + "/images/training"
    output_training_dir = output_ds_dir + "/images/train"
    process_images(input_training_dir, output_training_dir)

    input_dir = input_ds_dir + "/labels/training"
    output_dir = output_ds_dir + "/labels/train"
    convert_labels(input_dir, output_dir)

    # input_dir = dataset_dir + "/labels_lane3d/training"
    # output_dir = dataset_dir + "/labels_lane3d/train"
    # convert_lane3d_labels(input_dir, output_dir)

    # convert validation data
    input_dir = input_ds_dir + "/images/validation"
    output_dir = output_ds_dir + "/images/val"
    process_images(input_dir, output_dir)

    input_dir = input_ds_dir + "/labels/validation"
    output_dir = output_ds_dir + "/labels/val"
    convert_labels(input_dir, output_dir)

    # input_dir = dataset_dir + "/labels_lane3d/validation"
    # output_dir = dataset_dir + "/labels_lane3d/val"
    # convert_lane3d_labels(input_dir, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_ds_dir", help="Input dataset directory")
    parser.add_argument("-o", "--output_ds_dir", help="Output dataset directory")
    parser.add_argument(
        "--training-input-width",
        type=int,
        default=1024,
        help="Target input width used by AutoSpeedTrainingArgs (default: 1024)",
    )
    parser.add_argument(
        "--training-input-height",
        type=int,
        default=512,
        help="Target input height used by AutoSpeedTrainingArgs (default: 512)",
    )
    parser.add_argument(
        "--save-numpy-arrays",
        action="store_true",
        help="Save train and train_hpo images as .npy arrays in *_preprocessed directories",
    )
    parser.add_argument(
        "--npy-workers",
        type=int,
        default=16,
        help="Number of worker processes to use when exporting .npy arrays",
    )
    args = parser.parse_args()

    input_ds_dir = args.input_ds_dir
    output_ds_dir = args.output_ds_dir

    convert(input_ds_dir, output_ds_dir)
    
    expand_training_set_and_split_for_hpo(output_ds_dir)

    # Decode images, resize them so each sample is roughly within trainin input bounds
    # and store them as arrays for faster loading during training
    if args.save_numpy_arrays:
        save_split_as_numpy_arrays(
            Path(output_ds_dir) / "images" / "train",
            input_width=args.training_input_width,
            input_height=args.training_input_height,
            workers=args.npy_workers,
            augment=True,
        )
        save_split_as_numpy_arrays(
            Path(output_ds_dir) / "images" / "train_hpo",
            input_width=args.training_input_width,
            input_height=args.training_input_height,
            workers=args.npy_workers,
            augment=True,
        )
