import os
import json
import shutil
import random
import argparse
from typing import Dict, List, Any, Tuple, Union, Optional
from collections import Counter


def normalize_to_project_root(p: str, root: Optional[str] = None) -> str:
    """
    Normalizes a path relative to the project root with forward slashes.
    Falls back to normalized absolute path if outside the project root.
    """
    proj_root = os.path.abspath(root or os.getcwd())
    abs_p = os.path.abspath(p)
    try:
        rel = os.path.relpath(abs_p, start=proj_root)
        if rel.startswith(".."):
            return abs_p.replace("\\", "/")
        return rel.replace("\\", "/")
    except ValueError:
        return abs_p.replace("\\", "/")


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
    project_root: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Performs reproducible stratified splitting per class from a single image pool.
    
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
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        cls_idx = class_to_idx[cls_name]

        for p in train_imgs:
            splits["train"].append({
                "path": normalize_to_project_root(p, root=project_root),
                "class": cls_name,
                "label": cls_idx,
            })
        for p in val_imgs:
            splits["val"].append({
                "path": normalize_to_project_root(p, root=project_root),
                "class": cls_name,
                "label": cls_idx,
            })
        for p in test_imgs:
            splits["test"].append({
                "path": normalize_to_project_root(p, root=project_root),
                "class": cls_name,
                "label": cls_idx,
            })

    return splits


def split_train_val_sources(
    train_dir: str,
    val_dir: str,
    seed: int = 42,
    project_root: Optional[str] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """
    Constructs train, validation, and held-out test splits from existing train and val directories:
      - 100% of images from train_dir are preserved in 'train' split (1,440 images total, 240 per class).
      - Images from val_dir are split 50/50 per class into 'val' (180 images) and 'test' (180 images)
        stratified by class using fixed seed (30 val + 30 test per class).
    """
    train_class_images = scan_class_images(train_dir)
    val_class_images = scan_class_images(val_dir)

    train_classes = sorted(train_class_images.keys())
    val_classes = sorted(val_class_images.keys())

    if train_classes != val_classes:
        raise ValueError(
            f"Class mismatch between train_dir and val_dir: {train_classes} vs {val_classes}"
        )

    expected_neu_classes = [
        "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"
    ]
    if train_classes != expected_neu_classes:
        print(f"[Notice] Classes detected: {train_classes} (Standard NEU: {expected_neu_classes})")

    class_to_idx = {name: idx for idx, name in enumerate(train_classes)}

    splits: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": []
    }

    # 1. 100% of train images to 'train' split (preserving all original training images)
    for cls_name in train_classes:
        for p in sorted(train_class_images[cls_name]):
            splits["train"].append({
                "path": normalize_to_project_root(p, root=project_root),
                "class": cls_name,
                "label": class_to_idx[cls_name],
            })

    # 2. Stratified 50/50 split of validation images into 'val' (180) and held-out 'test' (180)
    rng = random.Random(seed)
    for cls_name in val_classes:
        images = list(sorted(val_class_images[cls_name]))
        rng.shuffle(images)
        n_val = len(images) // 2
        val_imgs = images[:n_val]
        test_imgs = images[n_val:]

        for p in val_imgs:
            splits["val"].append({
                "path": normalize_to_project_root(p, root=project_root),
                "class": cls_name,
                "label": class_to_idx[cls_name],
            })
        for p in test_imgs:
            splits["test"].append({
                "path": normalize_to_project_root(p, root=project_root),
                "class": cls_name,
                "label": class_to_idx[cls_name],
            })

    return splits, train_classes


def save_manifests_and_mappings(
    splits: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
    class_info: Union[Dict[str, List[str]], List[str]],
    seed: int,
) -> Tuple[str, str]:
    """
    Saves class mappings and complete split manifests to disk as structured JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(class_info, dict):
        class_names = sorted(class_info.keys())
    else:
        class_names = sorted(class_info)

    class_mapping = {
        "classes": class_names,
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "idx_to_class": {str(i): name for i, name in enumerate(class_names)},
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


def verify_split_manifest(manifest_path: str, project_root: Optional[str] = None) -> bool:
    """
    Validates split manifest for data integrity, class balance, path readability,
    split counts, continuous indices, and zero data leakage.
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
    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    # 2. Six Class Names & Indices Validation
    expected_classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    class_mapping = data.get("class_mapping", {})
    manifest_classes = class_mapping.get("classes", [])
    classes_match = (manifest_classes == expected_classes)
    class_to_idx = class_mapping.get("class_to_idx", {})
    expected_class_to_idx = {name: i for i, name in enumerate(expected_classes)}
    indices_match = (class_to_idx == expected_class_to_idx)

    # 3. Path Existence and Readability Check
    root = os.path.abspath(project_root or os.getcwd())
    unreadable_paths = []
    invalid_labels = []

    for split_name in ("train", "val", "test"):
        for item in splits[split_name]:
            p = item["path"]
            resolved_p = p if os.path.isabs(p) else os.path.join(root, p)
            if not os.path.exists(resolved_p) or not os.path.isfile(resolved_p) or os.path.getsize(resolved_p) == 0:
                unreadable_paths.append((split_name, p))
            if item.get("label") != class_to_idx.get(item.get("class")):
                invalid_labels.append((item.get("class"), item.get("label")))

    # 4. Per-Class Counts and Total Counts Validation
    train_counts = Counter(item["class"] for item in splits["train"])
    val_counts = Counter(item["class"] for item in splits["val"])
    test_counts = Counter(item["class"] for item in splits["test"])

    split_counts_valid = True
    if len(splits["train"]) != 1440 or len(splits["val"]) != 180 or len(splits["test"]) != 180:
        split_counts_valid = False
    for cls in expected_classes:
        if train_counts[cls] != 240 or val_counts[cls] != 30 or test_counts[cls] != 30:
            split_counts_valid = False

    total_images = len(train_paths) + len(val_paths) + len(test_paths)

    print("\n" + "=" * 78)
    print("                  DATASET SPLIT INTEGRITY VERIFICATION")
    print("=" * 78)
    print(f"Manifest File       : {os.path.abspath(manifest_path)}")
    print(f"Random Seed Used    : {data.get('seed')}")
    print(f"Total Unique Images : {total_images}")
    print("-" * 78)

    # Status Reports
    if not has_leakage:
        print("[PASS] DATA LEAKAGE: Sets are 100% disjoint (0 overlapping samples).")
    else:
        print("[FAIL] DATA LEAKAGE DETECTED:")
        if train_val_overlap:
            print(f"  - Train <-> Val Overlap: {len(train_val_overlap)} images")
        if train_test_overlap:
            print(f"  - Train <-> Test Overlap: {len(train_test_overlap)} images")
        if val_test_overlap:
            print(f"  - Val <-> Test Overlap: {len(val_test_overlap)} images")

    if classes_match and indices_match:
        print(f"[PASS] CLASS MAPPING: All 6 classes verified with continuous indices 0..5.")
    else:
        print(f"[FAIL] CLASS MAPPING MISMATCH: Found {manifest_classes}, expected {expected_classes}.")

    if not unreadable_paths:
        print(f"[PASS] PATH RESOLUTION: All {total_images} image paths exist and are readable from root.")
    else:
        print(f"[FAIL] UNREADABLE PATHS: {len(unreadable_paths)} image paths failed to resolve.")
        for s, p in unreadable_paths[:5]:
            print(f"  - [{s}] {p}")

    if not invalid_labels:
        print(f"[PASS] LABEL INTEGRITY: All {total_images} item labels match class_to_idx mapping.")
    else:
        print(f"[FAIL] LABEL MISMATCHES: {len(invalid_labels)} items have incorrect labels.")

    if split_counts_valid:
        print(f"[PASS] SPLIT BALANCING: Exactly 1440 Train (240/cls), 180 Val (30/cls), 180 Test (30/cls).")
    else:
        print(f"[FAIL] SPLIT COUNTS MISMATCH: Unexpected sample distribution.")

    print("-" * 78)
    print(f"{'Class Name':<22} | {'Train':<12} | {'Val':<10} | {'Test':<10} | {'Total':<8}")
    print("-" * 78)
    for cls in manifest_classes:
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

    all_passed = (
        not has_leakage and classes_match and indices_match and
        not unreadable_paths and not invalid_labels and split_counts_valid
    )
    if all_passed:
        print("[SUCCESS] All manifest integrity and validation checks passed successfully.")
    else:
        print("[ERROR] One or more manifest verification checks failed.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Create and verify stratified Train / Val / Test splits for NEU steel defect dataset."
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Input dataset directory with class folders")
    parser.add_argument("--train-dir", type=str, default=None, help="Path to training images directory (e.g., NEU-DET/train/images)")
    parser.add_argument("--val-dir", type=str, default=None, help="Path to validation images directory (e.g., NEU-DET/validation/images)")
    parser.add_argument("--output-dir", type=str, default="ml/artifacts/splits", help="Directory to save split manifests")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Fraction for training set (used with --data-dir)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction for validation set (used with --data-dir)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Fraction for test set (used with --data-dir)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--export-folders", type=str, default=None, help="Optional destination path to export physical train/val/test folders")
    parser.add_argument("--copy-files", action="store_true", help="Copy files instead of symlinks when exporting folders")
    parser.add_argument("--verify", action="store_true", help="Run verification on existing manifest")
    parser.add_argument("--manifest-path", type=str, default=None, help="Manifest file path to verify (used with --verify)")

    args = parser.parse_args()

    if args.verify:
        manifest_target = args.manifest_path or os.path.join(args.output_dir, "splits_manifest.json")
        verify_split_manifest(manifest_target)
        return

    if args.train_dir and args.val_dir:
        print(f"Generating splits from separate directories:")
        print(f"  Train Source: {args.train_dir}")
        print(f"  Val Source  : {args.val_dir}")
        splits, class_names = split_train_val_sources(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            seed=args.seed,
        )
    elif args.data_dir:
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
        class_names = sorted(class_images.keys())
    else:
        raise ValueError("Must provide either (--train-dir and --val-dir) or --data-dir.")

    manifest_path, mapping_path = save_manifests_and_mappings(
        splits=splits,
        output_dir=args.output_dir,
        class_info=class_names,
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
