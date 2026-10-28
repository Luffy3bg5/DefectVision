import os
import argparse
import json
from typing import Dict, Any
import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)

from ml.data.dataset import create_dataloaders
from ml.models.mobilenet import build_mobilenet_v2


@torch.no_grad()
def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    split: str = "test",
    num_classes: int = 2,
    batch_size: int = 32,
    output_dir: str = "ml/artifacts",
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    device = torch.device(device_name)
    os.makedirs(output_dir, exist_ok=True)

    loaders = create_dataloaders(data_dir=data_dir, batch_size=batch_size)
    target_loader = loaders.get(split) or loaders.get("val")
    if target_loader is None:
        raise ValueError(f"Could not find '{split}' or 'val' split in {data_dir}")

    model = build_mobilenet_v2(
        num_classes=num_classes,
        pretrained=False,
        checkpoint_path=checkpoint_path,
        device=device_name,
    )
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in target_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    conf_mat = confusion_matrix(all_targets, all_preds).tolist()
    clf_report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)

    results = {
        "split": split,
        "total_samples": len(all_targets),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_mat,
        "classification_report": clf_report,
    }

    # Binary ROC-AUC if applicable
    if num_classes == 2:
        try:
            pos_probs = [p[1] for p in all_probs]
            results["roc_auc"] = float(roc_auc_score(all_targets, pos_probs))
        except Exception:
            results["roc_auc"] = None

    metrics_file = os.path.join(output_dir, f"eval_metrics_{split}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete on {split} set:")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Metrics saved to: {metrics_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of target classes")
    parser.add_argument("--output-dir", type=str, default="ml/artifacts", help="Path to save evaluation metrics")
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
    )
