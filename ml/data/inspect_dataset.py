import os
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import matplotlib.pyplot as plt


# Standard NEU 6-class defect mappings
NEU_CLASSES = {
    "crazing": "Cr",
    "inclusion": "In",
    "patches": "Pa",
    "pitted_surface": "PS",
    "rolled-in_scale": "RS",
    "scratches": "Sc",
}


def verify_image(image_path: str) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str], Optional[str]]:
    """
    Attempts to read and verify an image file integrity.
    
    Returns:
        (is_valid, size_tuple, mode_str, error_message)
    """
    if os.path.getsize(image_path) == 0:
        return False, None, None, "Zero-byte file"

    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifies file header and stream
        # Re-open to read dimensions and verify complete decoding
        with Image.open(image_path) as img:
            size = img.size  # (width, height)
            mode = img.mode
            img.load()  # Forces full raster load into memory
        return True, size, mode, None
    except Exception as e:
        return False, None, None, str(e)


def scan_dataset(
    dataset_dir: str,
    extensions: Tuple[str, ...] = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> Dict[str, Any]:
    """
    Scans dataset directory, identifies class subdirectories, counts samples,
    and detects corrupt/unreadable files.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: '{dataset_dir}'")

    entries = os.listdir(dataset_dir)
    class_dirs = [
        d for d in entries
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ]

    # Handle nested split directories (e.g., dataset/train/...)
    sub_splits = [d for d in class_dirs if d.lower() in ("train", "val", "test", "valid")]
    if sub_splits and len(class_dirs) == len(sub_splits):
        print(f"Detected split root containing {sub_splits}. Scanning across splits...")
        combined_classes = {}
        for split in sub_splits:
            split_path = os.path.join(dataset_dir, split)
            for d in os.listdir(split_path):
                if os.path.isdir(os.path.join(split_path, d)):
                    combined_classes.setdefault(d, []).append(os.path.join(split_path, d))
        classes_to_scan = combined_classes
    else:
        classes_to_scan = {d: [os.path.join(dataset_dir, d)] for d in class_dirs}

    if not classes_to_scan:
        raise ValueError(
            f"No class folders found in '{dataset_dir}'. Ensure images are grouped in class folders."
        )

    results = {
        "dataset_dir": os.path.abspath(dataset_dir),
        "total_classes": len(classes_to_scan),
        "classes": {},
        "corrupted_files": [],
        "sample_images": {},
    }

    for class_name, folder_list in sorted(classes_to_scan.items()):
        valid_files = []
        corrupt_files = []
        resolutions = set()
        channels = set()

        for folder in folder_list:
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(extensions):
                        full_path = os.path.join(root, f)
                        is_valid, size, mode, err = verify_image(full_path)
                        if is_valid:
                            valid_files.append(full_path)
                            if size:
                                resolutions.add(size)
                            if mode:
                                channels.add(mode)
                        else:
                            corrupt_files.append((full_path, err))
                            results["corrupted_files"].append((full_path, err))

        results["classes"][class_name] = {
            "valid_count": len(valid_files),
            "corrupt_count": len(corrupt_files),
            "total_files": len(valid_files) + len(corrupt_files),
            "resolutions": list(resolutions),
            "color_modes": list(channels),
        }

        # Keep up to 3 valid sample paths for visualization
        results["sample_images"][class_name] = valid_files[:3]

    return results


def print_summary_report(results: Dict[str, Any]):
    """Prints a structured ASCII report of the dataset scan."""
    print("\n" + "=" * 78)
    print("        NEU STEEL SURFACE DEFECT DATASET - INSPECTION REPORT")
    print("=" * 78)
    print(f"Dataset Path : {results['dataset_dir']}")
    print(f"Total Classes: {results['total_classes']}")
    print("-" * 78)
    print(f"{'Class Name':<22} | {'Valid':<8} | {'Corrupt':<8} | {'Resolution(s)':<16} | {'Mode':<8}")
    print("-" * 78)

    total_valid = 0
    total_corrupt = 0

    for cls_name, info in results["classes"].items():
        res_str = ", ".join([f"{w}x{h}" for w, h in info["resolutions"][:2]]) or "N/A"
        modes_str = ", ".join(info["color_modes"]) or "N/A"
        print(f"{cls_name:<22} | {info['valid_count']:<8} | {info['corrupt_count']:<8} | {res_str:<16} | {modes_str:<8}")
        total_valid += info["valid_count"]
        total_corrupt += info["corrupt_count"]

    print("-" * 78)
    print(f"{'TOTAL':<22} | {total_valid:<8} | {total_corrupt:<8} |")
    print("=" * 78)

    if results["corrupted_files"]:
        print("\n[WARNING] Unreadable/Corrupted Files Detected:")
        for path, err in results["corrupted_files"][:10]:
            print(f"  - {path}: {err}")
        if len(results["corrupted_files"]) > 10:
            print(f"  ...and {len(results['corrupted_files']) - 10} more.")
    else:
        print("\n[OK] 100% of images verified successfully. No corrupt or unreadable files found.")


def plot_sample_grid(results: Dict[str, Any], output_path: Optional[str] = None, show: bool = True):
    """
    Generates a sample grid visualizing samples from each detected defect class.
    """
    classes = list(results["sample_images"].keys())
    if not classes:
        print("No samples available to display.")
        return

    n_classes = len(classes)
    cols = 3  # Display 3 samples per class
    fig, axes = plt.subplots(n_classes, cols, figsize=(3 * cols, 2.5 * n_classes))

    if n_classes == 1:
        axes = [axes]

    for row_idx, cls_name in enumerate(classes):
        samples = results["sample_images"][cls_name]
        for col_idx in range(cols):
            ax = axes[row_idx][col_idx] if n_classes > 1 else axes[col_idx]
            if col_idx < len(samples):
                img_path = samples[col_idx]
                try:
                    img = Image.open(img_path)
                    ax.imshow(img, cmap="gray" if img.mode in ("L", "1") else None)
                    fname = os.path.basename(img_path)
                    ax.set_title(f"{cls_name}\n({fname})", fontsize=8)
                except Exception:
                    ax.text(0.5, 0.5, "Error loading", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "No sample", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n[Saved] Class sample grid saved to: {output_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Scan, validate, and inspect NEU steel defect dataset for six-class classification."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset",
        help="Path to dataset root folder containing class directories (default: 'dataset')",
    )
    parser.add_argument(
        "--save-preview",
        type=str,
        default="ml/artifacts/neu_samples_preview.png",
        help="Path to save sample preview grid image (default: 'ml/artifacts/neu_samples_preview.png')",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not attempt to open interactive Matplotlib window (useful for headless / terminal runs)",
    )
    args = parser.parse_args()

    try:
        scan_results = scan_dataset(args.data_dir)
        print_summary_report(scan_results)
        plot_sample_grid(
            scan_results,
            output_path=args.save_preview,
            show=not args.no_display,
        )
    except Exception as e:
        print(f"\n[Error] {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
