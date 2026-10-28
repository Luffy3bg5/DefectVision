import os
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_manifest_split(
    manifest_path: str,
    split: str = "train",
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """
    Loads file paths and integer class labels for a specific split
    from an existing splits_manifest.json.

    Returns:
        file_paths: List of absolute file paths to images.
        labels: List of integer class labels (0 to num_classes - 1).
        metadata: Class mapping and dataset summary dictionary.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest not found: '{manifest_path}'")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    if split not in manifest_data["splits"]:
        raise ValueError(
            f"Split '{split}' not found in manifest. Available splits: {list(manifest_data['splits'].keys())}"
        )

    split_items = manifest_data["splits"][split]
    file_paths = [item["path"] for item in split_items]
    labels = [int(item["label"]) for item in split_items]
    metadata = {
        "class_mapping": manifest_data.get("class_mapping", {}),
        "seed": manifest_data.get("seed", 42),
        "total_count": len(split_items),
    }

    return file_paths, labels, metadata


def parse_and_process_image(
    file_path: tf.Tensor,
    label: tf.Tensor,
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow graph-compatible parser and preprocessor for MobileNetV2:
      1. Reads raw bytes and decodes image (BMP/PNG/JPEG).
      2. Ensures 3-channel RGB output (converts single-channel grayscale if needed).
      3. Resizes to target dimensions (224 x 224).
      4. Applies data augmentations (TRAINING ONLY: flips, subtle contrast/brightness).
      5. Applies MobileNetV2 preprocessing (scales pixel values from [0, 255] to [-1, 1]).
      6. Enforces static shape [224, 224, 3].
    """
    # 1. Read file bytes and decode
    raw_bytes = tf.io.read_file(file_path)
    # decode_image handles BMP, PNG, JPEG; channels=3 converts grayscale to 3 identical RGB channels
    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False, dtype=tf.uint8)

    # 2. Defensive check: Convert grayscale (H, W, 1) to RGB (H, W, 3) if channels was 1
    image_shape = tf.shape(image)
    image = tf.cond(
        tf.equal(image_shape[-1], 1),
        lambda: tf.image.grayscale_to_rgb(image),
        lambda: image
    )

    # 3. Cast to float32 and resize to (224, 224)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)

    # 4. Data Augmentation (APPLIED ONLY TO TRAINING SPLIT)
    if is_training:
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        # Random vertical flip (defects on steel sheets have no preferential orientation)
        image = tf.image.random_flip_up_down(image)
        # Random subtle brightness perturbation
        image = tf.image.random_brightness(image, max_delta=0.08)
        # Random subtle contrast perturbation
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        # Keep values clamped in valid pixel range before normalization
        image = tf.clip_by_value(image, 0.0, 255.0)

    # 5. MobileNetV2 Specific Preprocessing: scales values to [-1.0, 1.0]
    # Functionally equivalent to: (image / 127.5) - 1.0
    image = preprocess_input(image)

    # 6. Ensure static shape for graph execution
    image = tf.ensure_shape(image, [image_size[0], image_size[1], 3])
    label = tf.ensure_shape(label, [])

    return image, label


def build_tf_dataset(
    file_paths: List[str],
    labels: List[int],
    batch_size: int = 32,
    is_training: bool = False,
    image_size: Tuple[int, int] = (224, 224),
    shuffle_buffer: Optional[int] = None,
    cache: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Constructs an optimized tf.data.Dataset pipeline:
    from_tensor_slices -> [shuffle] -> map(preprocess) -> [cache] -> batch -> prefetch
    """
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        buf_size = shuffle_buffer or max(len(file_paths), 1024)
        dataset = dataset.shuffle(buffer_size=buf_size, seed=seed, reshuffle_each_iteration=True)

    # Map preprocessing in parallel across available CPU cores
    dataset = dataset.map(
        lambda p, l: parse_and_process_image(p, l, image_size=image_size, is_training=is_training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Cache dataset in memory to avoid repeated disk reads after epoch 1
    if cache:
        dataset = dataset.cache()

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def create_mobilenet_dataloaders(
    manifest_path: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    cache: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, tf.data.Dataset], Dict[str, Any]]:
    """
    Creates train, val, and test tf.data.Dataset loaders from a split manifest.
    
    Augmentations are applied ONLY to the 'train' loader.
    'val' and 'test' loaders use deterministic evaluation preprocessing.

    Returns:
        loaders: {"train": train_ds, "val": val_ds, "test": test_ds}
        class_mapping: Class dictionary and metadata from manifest.
    """
    loaders = {}
    metadata = {}

    for split in ["train", "val", "test"]:
        is_training = (split == "train")
        paths, labels, meta = load_manifest_split(manifest_path, split=split)
        if not metadata:
            metadata = meta

        ds = build_tf_dataset(
            file_paths=paths,
            labels=labels,
            batch_size=batch_size,
            is_training=is_training,
            image_size=image_size,
            cache=cache,
            seed=seed,
        )
        loaders[split] = ds

    return loaders, metadata


def verify_pipeline(manifest_path: str, batch_size: int = 32):
    """
    Verifies batch output shapes, data types, pixel value ranges, and class alignment.
    """
    print("\n" + "=" * 78)
    print("       MOBILENETV2 TENSORFLOW / KERAS DATA PIPELINE VERIFICATION")
    print("=" * 78)

    loaders, meta = create_mobilenet_dataloaders(manifest_path=manifest_path, batch_size=batch_size)
    class_names = meta.get("class_mapping", {}).get("classes", [])
    print(f"Manifest File  : {os.path.abspath(manifest_path)}")
    print(f"Target Classes : {class_names}")
    print("-" * 78)

    for split_name, ds in loaders.items():
        # Fetch first batch
        for images, labels in ds.take(1):
            img_shape = images.shape
            lbl_shape = labels.shape
            min_val = float(tf.reduce_min(images).numpy())
            max_val = float(tf.reduce_max(images).numpy())
            mean_val = float(tf.reduce_mean(images).numpy())

            print(f"[{split_name.upper():<5}] Batch Image Shape : {img_shape} (dtype: {images.dtype.name})")
            print(f"        Batch Label Shape : {lbl_shape} (dtype: {labels.dtype.name})")
            print(f"        Pixel Value Range : [{min_val:.4f}, {max_val:.4f}] | Mean: {mean_val:.4f}")
            print(f"        Sample Labels     : {labels[:8].numpy().tolist()}")
            
            # Sanity assertions
            assert img_shape[-3:] == (224, 224, 3), f"Expected (224, 224, 3), got {img_shape[-3:]}"
            assert min_val >= -1.01 and max_val <= 1.01, f"Expected [-1, 1], got [{min_val}, {max_val}]"

    print("-" * 78)
    print("[OK] All pipeline assertions passed. Loaders are ready for MobileNetV2.")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify TensorFlow/Keras input pipeline for MobileNetV2 defect classification."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="ml/artifacts/splits/splits_manifest.json",
        help="Path to splits_manifest.json",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for verification")
    args = parser.parse_args()

    if os.path.exists(args.manifest):
        verify_pipeline(args.manifest, batch_size=args.batch_size)
    else:
        print(f"[Notice] Manifest file '{args.manifest}' does not exist yet.")
        print("Run 'python -m ml.data.split_dataset --data-dir <path>' first to generate it.")
