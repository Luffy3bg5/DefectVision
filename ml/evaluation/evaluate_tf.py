import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "MobileNetV2 Test Set Confusion Matrix",
):
    """
    Renders and saves a detailed Confusion Matrix heatmap with raw counts and normalized percentages.
    """
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-9)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Defect Label",
        xlabel="Predicted Defect Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Artifact Saved] Confusion matrix plot saved to: {os.path.abspath(output_path)}")


def plot_prediction_samples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    output_path: str,
    max_samples_per_category: int = 4,
):
    """
    Renders visual comparison of correctly classified vs. misclassified test samples.
    """
    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx = np.where(y_true != y_pred)[0]

    n_correct = min(len(correct_idx), max_samples_per_category)
    n_wrong = min(len(wrong_idx), max_samples_per_category)
    total_samples = n_correct + n_wrong

    if total_samples == 0:
        return

    cols = max(n_correct, n_wrong, 1)
    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 7))

    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # Top Row: Correct Predictions
    for col_idx in range(cols):
        ax = axes[0, col_idx]
        if col_idx < n_correct:
            idx = correct_idx[col_idx]
            # Rescale normalized [-1, 1] back to [0, 1] for visualization
            img = (images[idx] + 1.0) / 2.0
            img = np.clip(img, 0.0, 1.0)
            ax.imshow(img)
            true_cls = class_names[y_true[idx]]
            conf = probs[idx, y_pred[idx]] * 100
            ax.set_title(f"CORRECT\nTrue: {true_cls}\nPred: {true_cls} ({conf:.1f}%)", color="green", fontsize=9)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom Row: Misclassified Predictions
    for col_idx in range(cols):
        ax = axes[1, col_idx]
        if col_idx < n_wrong:
            idx = wrong_idx[col_idx]
            img = (images[idx] + 1.0) / 2.0
            img = np.clip(img, 0.0, 1.0)
            ax.imshow(img)
            true_cls = class_names[y_true[idx]]
            pred_cls = class_names[y_pred[idx]]
            conf = probs[idx, y_pred[idx]] * 100
            ax.set_title(f"MISCLASSIFIED\nTrue: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)", color="red", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No more errors\nin sample set", ha="center", va="center", color="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Qualitative Test Set Predictions (Correct vs. Misclassified)", fontsize=13, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Artifact Saved] Qualitative predictions plot saved to: {os.path.abspath(output_path)}")


def evaluate_test_set(
    model_path: str = "ml/artifacts/finetune/best_overall_model.keras",
    manifest_path: str = "ml/artifacts/splits/splits_manifest.json",
    output_dir: str = "ml/artifacts/evaluation",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Executes rigorous evaluation on the strictly held-out test split.
    
    Computes:
      - Overall Accuracy
      - Macro Precision, Recall, and F1-score
      - Full Per-Class Classification Report
      - Confusion Matrix
      - Visual Confusion Matrix Plot
      - Qualitative Correct vs. Misclassified Predictions Plot
    """
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
    from ml.data.tf_pipeline import create_mobilenet_dataloaders

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: '{model_path}'. "
            "Please run training first before evaluating."
        )

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Dataset split manifest not found at: '{manifest_path}'.")

    # 1. Load Data Splits and Extract HELD-OUT TEST SET
    print(f"Loading held-out test split from: {manifest_path}")
    loaders, metadata = create_mobilenet_dataloaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        image_size=(224, 224),
        cache=False,  # Test evaluation only runs once
    )
    test_ds = loaders["test"]
    class_names = metadata.get("class_mapping", {}).get("classes", [])

    # 2. Load Model
    print(f"Loading model checkpoint from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 3. Collect Ground Truth and Predictions across Test Batches
    print("Executing test inference...")
    y_true_list = []
    y_pred_list = []
    probs_list = []
    sample_images = []

    for images, labels in test_ds:
        batch_probs = model.predict(images, verbose=0)
        batch_preds = np.argmax(batch_probs, axis=1)

        y_true_list.extend(labels.numpy().tolist())
        y_pred_list.extend(batch_preds.tolist())
        probs_list.extend(batch_probs.tolist())

        # Collect first few raw image arrays for qualitative error visualization
        if len(sample_images) < 100:
            sample_images.extend(images.numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    probs = np.array(probs_list)
    sample_images = np.array(sample_images)

    # 4. Metric Calculations
    acc = float(accuracy_score(y_true, y_pred))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # Identify Class Weaknesses (lowest F1 scores or high false negatives)
    class_f1_scores = {cls: clf_report[cls]["f1-score"] for cls in class_names if cls in clf_report}
    sorted_weaknesses = sorted(class_f1_scores.items(), key=lambda item: item[1])

    # 5. Save Plots
    cm_plot_path = os.path.join(output_dir, "test_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    samples_plot_path = os.path.join(output_dir, "test_prediction_samples.png")
    plot_prediction_samples(sample_images, y_true[:len(sample_images)], y_pred[:len(sample_images)], probs[:len(sample_images)], class_names, samples_plot_path)

    # 6. Save Structured Metrics JSON
    results = {
        "model_evaluated": os.path.abspath(model_path),
        "manifest_path": os.path.abspath(manifest_path),
        "total_test_samples": len(y_true),
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "class_rankings_by_f1": sorted_weaknesses,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
    }

    metrics_json_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 7. Print Terminal Summary
    print("\n" + "=" * 78)
    print("             HELD-OUT TEST SET EVALUATION REPORT")
    print("=" * 78)
    print(f"Total Test Samples     : {len(y_true)}")
    print(f"Overall Test Accuracy  : {acc * 100:.2f}%")
    print(f"Macro Precision        : {macro_precision:.4f}")
    print(f"Macro Recall           : {macro_recall:.4f}")
    print(f"Macro F1-Score         : {macro_f1:.4f}")
    print("-" * 78)
    print(f"{'Class Name':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 78)
    for cls in class_names:
        stats = clf_report.get(cls, {})
        p = stats.get("precision", 0.0)
        r = stats.get("recall", 0.0)
        f = stats.get("f1-score", 0.0)
        s = int(stats.get("support", 0))
        print(f"{cls:<22} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f} | {s:<8}")
    print("-" * 78)
    print(f"Defect classes most challenging for the model (lowest F1):")
    for cls, f1_val in sorted_weaknesses[:3]:
        print(f"  - {cls}: F1 = {f1_val:.4f}")
    print("=" * 78)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 on held-out test split.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="ml/artifacts/finetune/best_overall_model.keras",
        help="Path to trained model checkpoint (.keras)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="ml/artifacts/splits/splits_manifest.json",
        help="Path to splits_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/artifacts/evaluation",
        help="Directory to save evaluation plots and metrics",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()

    evaluate_test_set(
        model_path=args.model_path,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
