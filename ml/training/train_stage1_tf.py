import os
import json
import argparse
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt


def plot_training_history(history_dict: Dict[str, list], output_path: str):
    """
    Plots and saves publication-quality training and validation Loss and Accuracy curves.
    """
    epochs_range = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Loss Subplot
    ax1.plot(epochs_range, history_dict["loss"], label="Training Loss", color="#2563eb", linewidth=2.0)
    ax1.plot(epochs_range, history_dict["val_loss"], label="Validation Loss", color="#dc2626", linewidth=2.0, linestyle="--")
    best_loss_epoch = int(history_dict["val_loss"].index(min(history_dict["val_loss"]))) + 1
    best_val_loss = min(history_dict["val_loss"])
    ax1.axvline(x=best_loss_epoch, color="#059669", linestyle=":", label=f"Best Val Loss ({best_val_loss:.4f})")
    ax1.set_title("Stage 1: Training vs. Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Sparse Categorical Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Accuracy Subplot
    ax2.plot(epochs_range, history_dict["accuracy"], label="Training Accuracy", color="#2563eb", linewidth=2.0)
    ax2.plot(epochs_range, history_dict["val_accuracy"], label="Validation Accuracy", color="#dc2626", linewidth=2.0, linestyle="--")
    best_acc_epoch = int(history_dict["val_accuracy"].index(max(history_dict["val_accuracy"]))) + 1
    best_val_acc = max(history_dict["val_accuracy"])
    ax2.axvline(x=best_acc_epoch, color="#059669", linestyle=":", label=f"Best Val Acc ({best_val_acc * 100:.2f}%)")
    ax2.set_title("Stage 1: Training vs. Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("DefectVision MobileNetV2 - Transfer Learning Stage 1 (Frozen Backbone)", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Artifact Saved] Loss & Accuracy curves saved to: {os.path.abspath(output_path)}")


def train_stage1(
    manifest_path: str = "ml/artifacts/splits/splits_manifest.json",
    output_dir: str = "ml/artifacts/stage1",
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes Transfer Learning Stage 1 for MobileNetV2:
      - Backbone weights remain completely frozen.
      - Trains only the GlobalAvgPool + Dropout + Dense Softmax head.
      - Uses SparseCategoricalCrossentropy loss and Adam optimizer.
      - Employs EarlyStopping, ModelCheckpoint (best val_accuracy), and ReduceLROnPlateau.
      - Strictly leaves the test set untouched (evaluated only on val_ds).
    """
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)

    from ml.data.tf_pipeline import create_mobilenet_dataloaders
    from ml.models.mobilenet_tf import build_mobilenet_v2_classifier

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data (ONLY train and val; test set is strictly held out)
    print(f"Loading dataset splits from manifest: {manifest_path}")
    loaders, metadata = create_mobilenet_dataloaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        image_size=(224, 224),
        cache=True,
    )
    train_ds = loaders["train"]
    val_ds = loaders["val"]
    # Explicitly do NOT touch loaders["test"] during training or tuning

    class_names = metadata.get("class_mapping", {}).get("classes", [])
    num_classes = len(class_names) or 6

    # 2. Build MobileNetV2 with Frozen Backbone
    print(f"Building MobileNetV2 with frozen backbone for {num_classes} defect classes...")
    model, base_model = build_mobilenet_v2_classifier(
        num_classes=num_classes,
        input_shape=(224, 224, 3),
        dropout_rate=0.2,
        freeze_backbone=True,
    )

    # 3. Optimizer, Loss & Metrics
    # Adam with standard initial lr=1e-3 is ideal for learning the new dense head weights
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Integer labels from manifest -> SparseCategoricalCrossentropy is mathematically optimal
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        name="sparse_categorical_crossentropy",
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    print(f"Compilation complete. Trainable parameters: {trainable_params:,} | Frozen parameters: {non_trainable_params:,}")

    # 4. Callbacks: Early Stopping, Checkpoint, LR Plateau, CSV Logging
    best_model_path = os.path.join(output_dir, "best_mobilenet_stage1.keras")
    log_csv_path = os.path.join(output_dir, "stage1_training_log.csv")

    callbacks = [
        # Checkpoint: Save only the best performing model weights based on validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # Early Stopping: Prevent overfitting by stopping training when val_loss stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR on Plateau: Drop learning rate by half when validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        # CSV Logger: Record per-epoch metrics to disk
        tf.keras.callbacks.CSVLogger(filename=log_csv_path),
    ]

    # 5. Execute Training (Strictly train_ds and val_ds only!)
    print(f"\nBeginning Transfer Learning Stage 1 for up to {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_dict = history.history

    # 6. Save Training History JSON
    history_json_path = os.path.join(output_dir, "stage1_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    # 7. Plot Loss and Accuracy Curves
    curves_plot_path = os.path.join(output_dir, "stage1_learning_curves.png")
    plot_training_history(history_dict, curves_plot_path)

    final_val_acc = max(history_dict.get("val_accuracy", [0]))
    final_val_loss = min(history_dict.get("val_loss", [0]))
    print("\n" + "=" * 78)
    print("                     STAGE 1 TRAINING COMPLETE")
    print("=" * 78)
    print(f"Best Validation Accuracy : {final_val_acc * 100:.2f}%")
    print(f"Lowest Validation Loss   : {final_val_loss:.4f}")
    print(f"Best Checkpoint Saved    : {os.path.abspath(best_model_path)}")
    print(f"Learning Curves Plot     : {os.path.abspath(curves_plot_path)}")
    print("=" * 78)

    return {
        "best_model_path": best_model_path,
        "curves_plot_path": curves_plot_path,
        "best_val_accuracy": float(final_val_acc),
        "best_val_loss": float(final_val_loss),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Stage 1 (Frozen Backbone).")
    parser.add_argument(
        "--manifest",
        type=str,
        default="ml/artifacts/splits/splits_manifest.json",
        help="Path to splits_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/artifacts/stage1",
        help="Directory to save model checkpoints and plots",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Maximum number of training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    train_stage1(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
    )
