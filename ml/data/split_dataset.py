import os
import json
import shutil
import random
import argparse
from typing import Dict, List, Any, Tuple
from collections import Counter


def scan_class_images(data_dir: str, extensions: Tuple[str, ...] = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> Dict[str, List[str]]:
    """
    Scans dataset directory and returns a dictionary of {class_name: [list_of_image_paths]}.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

    class_images = {}
    class_folders = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".") and d not in ("train", "val", "test")
    ])

    if not class_folders:
        raise ValueError(f"No class folders found directly inside '{data_dir}'.")

    for cls in class_folders:
        cls_path = os.path.join(data_dir, cls)
        image_files = []
        for root, _, files in os.walk(cls_path):
            for f in files:
                if f.lower().endswith(extensions):
                    image_files.append(os.path.abspath(os.path.join(root, f)))
        class_images[cls] = sorted(image_files)

    return class_images


def stratified_split(
    class_images: Dict[str, List[str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Performs reproducible stratified splitting per class.
    
    Guarantees:
      - Class distributions are preserved identically across splits.
      - Exact reproducibility via fixed random seed.
      - Disjoint partition (0 data leakage between splits).
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    train_pct = train_ratio / ratio_sum
    val_pct = val_ratio / ratio_sum

    rng = random.Random(seed)

    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    class_names = sorted(class_images.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for cls_name in class_names:
        images = list(class_images[cls_name])
        rng.shuffle(images)

        n_total = len(images)
        n_train = int(round(n_total * train_pct))
        n_val = int(round(n_total * val_pct))
        # Ensure remaining go to test so no sample is lost
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        cls_idx = class_to_idx[cls_name]

        for p in train_imgs:
            splits["train"].append({"path": p, "class": cls_name, "label": cls_idx})
        for p in val_imgs:
            splits["val"].append({"path": p, "class": cls_name, "label": cls_idx})
        for p in test_imgs:
            splits["test"].append({"path": p, "class": cls_name, "label": cls_idx})

    return splits


def save_manifests_and_mappings(
    splits: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
    class_images: Dict[str, List[str]],
    seed: int,
) -> Tuple[str, str]:
    """
    Saves class mappings and complete split manifests to disk as structured JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    class_names = sorted(class_images.keys())
    class_mapping = {
        "classes": class_names,
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "idx_to_class": {i: name for i, name in enumerate(class_names)},
        "num_classes": len(class_names),
    }

    manifest_data = {
        "seed": seed,
        "class_mapping": class_mapping,
        "counts": {
            "total": sum(len(items) for items in splits.values()),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "splits": splits,
    }

    mapping_file = os.path.join(output_dir, "class_mapping.json")
    manifest_file = os.path.join(output_dir, "splits_manifest.json")

    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_file, mapping_file


def export_split_folders(
    splits: Dict[str, List[Dict[str, Any]]],
    destination_root: str,
    copy_files: bool = False,
):
    """
    Materializes a structured split directory: destination_root/{train,val,test}/{class_name}/*.
    Uses symlinks by default or hard copies if copy_files=True.
    """
    print(f"\nMaterializing split folders at '{destination_root}'...")
    for split_name, items in splits.items():
        for item in items:
            src = item["path"]
            cls_name = item["class"]
            dest_dir = os.path.join(destination_root, split_name, cls_name)
            os.makedirs(dest_dir, exist_ok=True)
            dest_file = os.path.join(dest_dir, os.path.basename(src))

            if not os.path.exists(dest_file):
                if copy_files:
                    shutil.copy2(src, dest_file)
                else:
                    try:
                        os.symlink(src, dest_file)
                    except (OSError, NotImplementedError):
                        shutil.copy2(src, dest_file)

    print(f"[Done] Exported train/val/test folders into '{destination_root}'")


def verify_split_manifest(manifest_path: str):
    """
    Validates split manifest for data integrity, class balance, and data leakage.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file '{manifest_path}' not found.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = data["splits"]
    train_paths = {item["path"] for item in splits["train"]}
    val_paths = {item["path"] for item in splits["val"]}
    test_paths = {item["path"] for item in splits["test"]}

    # 1. Leakage Verification (Disjoint Sets Check)
    train_val_overlap = train_paths.intersection(val_paths)
    train_test_overlap = train_paths.intersection(test_paths)
    val_test_overlap = val_paths.intersection(test_paths)

    total_images = len(train_paths) + len(val_paths) + len(test_paths)

    print("\n" + "=" * 78)
    print("                  DATASET SPLIT INTEGRITY VERIFICATION")
    print("=" * 78)
    print(f"Manifest File       : {os.path.abspath(manifest_path)}")
    print(f"Random Seed Used    : {data.get('seed')}")
    print(f"Total Unique Images : {total_images}")
    print("-" * 78)

    # Overlap Status
    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)
    if not has_leakage:
        print("[OK] DATA LEAKAGE CHECK PASSED: Sets are 100% disjoint (0 overlapping samples).")
    else:
        print("[FAIL] DATA LEAKAGE DETECTED:")
        if train_val_overlap:
            print(f"  - Train <-> Val Overlap: {len(train_val_overlap)} images")
        if train_test_overlap:
            print(f"  - Train <-> Test Overlap: {len(train_test_overlap)} images")
        if val_test_overlap:
            print(f"  - Val <-> Test Overlap: {len(val_test_overlap)} images")

    # 2. Per-class distribution across splits
    classes = data["class_mapping"]["classes"]
    train_counts = Counter(item["class"] for item in splits["train"])
    val_counts = Counter(item["class"] for item in splits["val"])
    test_counts = Counter(item["class"] for item in splits["test"])

    print("-" * 78)
    print(f"{'Class Name':<22} | {'Train (70%)':<12} | {'Val (15%)':<10} | {'Test (15%)':<10} | {'Total':<8}")
    print("-" * 78)

    for cls in classes:
        tr = train_counts[cls]
        va = val_counts[cls]
        te = test_counts[cls]
        tot = tr + va + te
        print(f"{cls:<22} | {tr:<12} | {va:<10} | {te:<10} | {tot:<8}")

    print("-" * 78)
    print(
        f"{'TOTAL':<22} | "
        f"{len(train_paths):<12} | "
        f"{len(val_paths):<10} | "
        f"{len(test_paths):<10} | "
        f"{total_images:<8}"
    )
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Create and verify stratified Train / Val / Test splits for NEU steel defect dataset."
    )
    parser.add_argument("--data-dir", type=str, default="dataset", help="Input dataset directory with class folders")
    parser.add_argument("--output-dir", type=str, default="ml/artifacts/splits", help="Directory to save split manifests")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Fraction for training set (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction for validation set (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Fraction for test set (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--export-folders", type=str, default=None, help="Optional destination path to export physical train/val/test folders")
    parser.add_argument("--copy-files", action="store_true", help="Copy files instead of symlinks when exporting folders")
    parser.add_argument("--verify", action="store_true", help="Run verification on existing manifest")
    parser.add_argument("--manifest-path", type=str, default=None, help="Manifest file path to verify (used with --verify)")

    args = parser.parse_args()

    manifest_target = args.manifest_path or os.path.join(args.output_dir, "splits_manifest.json")

    if args.verify:
        verify_split_manifest(manifest_target)
        return

    print(f"Scanning dataset from: {args.data_dir}")
    class_images = scan_class_images(args.data_dir)
    print(f"Found {len(class_images)} classes with {sum(len(v) for v in class_images.values())} total images.")

    splits = stratified_split(
        class_images=class_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    manifest_path, mapping_path = save_manifests_and_mappings(
        splits=splits,
        output_dir=args.output_dir,
        class_images=class_images,
        seed=args.seed,
    )

    print(f"\n[Saved] Split manifest saved to: {manifest_path}")
    print(f"[Saved] Class mapping saved to : {mapping_path}")

    if args.export_folders:
        export_split_folders(splits, args.export_folders, copy_files=args.copy_files)

    # Immediately verify generated split
    verify_split_manifest(manifest_path)


if __name__ == "__main__":
    main()
