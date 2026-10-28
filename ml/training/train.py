import os
import time
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim

from ml.data.dataset import create_dataloaders
from ml.models.mobilenet import build_mobilenet_v2


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def train_model(
    data_dir: str,
    output_dir: str = "ml/artifacts",
    num_classes: int = 2,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    device = torch.device(device_name)
    os.makedirs(output_dir, exist_ok=True)

    loaders = create_dataloaders(data_dir=data_dir, batch_size=batch_size)
    if "train" not in loaders or "val" not in loaders:
        raise ValueError(f"Data directory '{data_dir}' must contain 'train' and 'val' subdirectories.")

    model = build_mobilenet_v2(num_classes=num_classes, pretrained=True, device=device_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "best_mobilenet_v2.pt")

    print(f"Starting training on {device} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_metrics["loss"])

        duration = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{duration:.1f}s] - "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "num_classes": num_classes,
            }, best_model_path)
            print(f" -> Saved new best model to {best_model_path} (Acc: {best_val_acc:.4f})")

    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for Defect Detection")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--output-dir", type=str, default="ml/artifacts", help="Path to save artifacts")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of target classes")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
