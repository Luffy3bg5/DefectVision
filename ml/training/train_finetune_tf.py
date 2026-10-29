import os
import json
import shutil
import argparse
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt


def plot_combined_learning_curves(
    stage1_history_path: Optional[str],
    stage2_history: Dict[str, list],
    output_path: str,
):
    """
    Plots training and validation learning curves showing the transition
    from Stage 1 (frozen head training) to Stage 2 (backbone fine-tuning).
    """
    stage1_loss = []
    stage1_val_loss = []
    stage1_acc = []
    stage1_val_acc = []

    if stage1_history_path and os.path.exists(stage1_history_path):
        try:
            with open(stage1_history_path, "r", encoding="utf-8") as f:
                s1_data = json.load(f)
            stage1_loss = s1_data.get("loss", [])
            stage1_val_loss = s1_data.get("val_loss", [])
            stage1_acc = s1_data.get("accuracy", [])
            stage1_val_acc = s1_data.get("val_accuracy", [])
        except Exception as e:
            print(f"[Warning] Could not load Stage 1 history for plotting: {e}")

    s1_len = len(stage1_loss)
    s2_len = len(stage2_history["loss"])

    combined_loss = stage1_loss + stage2_history["loss"]
    combined_val_loss = stage1_val_loss + stage2_history["val_loss"]
    combined_acc = stage1_acc + stage2_history["accuracy"]
    combined_val_acc = stage1_val_acc + stage2_history["val_accuracy"]

    total_epochs = range(1, len(combined_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Loss Subplot
    ax1.plot(total_epochs, combined_loss, label="Training Loss", color="#2563eb", linewidth=2.0)
    ax1.plot(total_epochs, combined_val_loss, label="Validation Loss", color="#dc2626", linewidth=2.0, linestyle="--")
    if s1_len > 0:
        ax1.axvline(x=s1_len + 0.5, color="#d97706", linestyle="-.", linewidth=2, label="Fine-Tuning Started")
    ax1.set_title("Training vs. Validation Loss Across Stages", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Cumulative Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Accuracy Subplot
    ax2.plot(total_epochs, combined_acc, label="Training Accuracy", color="#2563eb", linewidth=2.0)
    ax2.plot(total_epochs, combined_val_acc, label="Validation Accuracy", color="#dc2626", linewidth=2.0, linestyle="--")
    if s1_len > 0:
        ax2.axvline(x=s1_len + 0.5, color="#d97706", linestyle="-.", linewidth=2, label="Fine-Tuning Started")
    ax2.set_title("Training vs. Validation Accuracy Across Stages", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cumulative Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("MobileNetV2 Transfer Learning: Stage 1 (Frozen) vs. Stage 2 (Fine-Tuned)", fontsize=13, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Artifact Saved] Comparison learning curves saved to: {os.path.abspath(output_path)}")


def train_finetune(
    stage1_checkpoint: str = "ml/artifacts/stage1/best_mobilenet_stage1.keras",
    manifest_path: str = "ml/artifacts/splits/splits_manifest.json",
    output_dir: str = "ml/artifacts/finetune",
    fine_tune_at: int = 100,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    patience: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes Optional Transfer Learning Stage 2 (Fine-Tuning):
      1. Loads Stage 1 best validation model checkpoint.
      2. Measures Pre-Fine-Tuning validation baseline.
      3. Unfreezes upper backbone layers (from layer fine_tune_at to 153).
         Keeps earlier feature extractor layers 0..fine_tune_at-1 frozen.
      4. Recompiles with a much lower learning rate (e.g. 1e-5) to protect pretrained weights.
      5. Trains with EarlyStopping and ModelCheckpoint.
      6. Measures Post-Fine-Tuning validation performance.
      7. Saves the best overall checkpoint.
      8. Strictly avoids evaluating on or leaking the test set.
    """
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)

    from ml.data.tf_pipeline import create_mobilenet_dataloaders
    from ml.models.mobilenet_tf import unfreeze_backbone_layers

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(stage1_checkpoint):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at: '{stage1_checkpoint}'. "
            "Please run Stage 1 training first (python -m ml.training.train_stage1_tf)."
        )

    # 1. Load Data Loaders (Train & Val ONLY; test set is held out)
    print(f"Loading data splits from: {manifest_path}")
    loaders, metadata = create_mobilenet_dataloaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        image_size=(224, 224),
        cache=True,
    )
    train_ds = loaders["train"]
    val_ds = loaders["val"]

    # 2. Load Pretrained Stage 1 Checkpoint
    print(f"Loading Stage 1 model from: {stage1_checkpoint}")
    model = tf.keras.models.load_model(stage1_checkpoint)

    # 3. Evaluate Pre-Fine-Tuning Baseline Performance on Val Set
    print("\nEvaluating Pre-Fine-Tuning baseline on Validation set...")
    pre_results = model.evaluate(val_ds, verbose=0)
    pre_val_loss = float(pre_results[0])
    pre_val_acc = float(pre_results[1])
    print(f"Baseline Validation Loss     : {pre_val_loss:.4f}")
    print(f"Baseline Validation Accuracy : {pre_val_acc * 100:.2f}%")

    # 4. Unfreeze Selected Later Layers of Backbone
    total_p, trainable_p, non_trainable_p = unfreeze_backbone_layers(
        model, fine_tune_at=fine_tune_at
    )
    print(f"\nUnfrozen backbone layers from index {fine_tune_at} onwards:")
    print(f"  - Total Parameters        : {total_p:,}")
    print(f"  - Trainable Parameters    : {trainable_p:,}")
    print(f"  - Frozen (Non-trainable)  : {non_trainable_p:,}")

    # 5. Recompile Model with Reduced Learning Rate
    # A low learning rate (e.g. 1e-5 vs 1e-3) is critical to prevent catastrophic forgetting
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        name="sparse_categorical_crossentropy",
    )
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # 6. Callbacks
    best_finetuned_checkpoint = os.path.join(output_dir, "best_mobilenet_finetuned.keras")
    log_csv_path = os.path.join(output_dir, "finetune_training_log.csv")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_finetuned_checkpoint,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(filename=log_csv_path),
    ]

    # 7. Execute Fine-Tuning
    print(f"\nBeginning Fine-Tuning Stage 2 for up to {epochs} epochs (lr={learning_rate})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    stage2_history = history.history

    # 8. Measure Post-Fine-Tuning Validation Performance on Best Checkpoint
    if os.path.exists(best_finetuned_checkpoint):
        print(f"\nLoading best saved fine-tuned checkpoint for validation evaluation: {best_finetuned_checkpoint}")
        eval_model = tf.keras.models.load_model(best_finetuned_checkpoint)
        post_results = eval_model.evaluate(val_ds, verbose=0)
    else:
        post_results = model.evaluate(val_ds, verbose=0)

    post_val_loss = float(post_results[0])
    post_val_acc = float(post_results[1])

    delta_loss = post_val_loss - pre_val_loss
    delta_acc = post_val_acc - pre_val_acc

    # 9. Comparison & Selection of Best Overall Model
    # Selection Safeguards:
    # 1. Higher validation accuracy strictly wins.
    # 2. If validation accuracy ties, validation loss is used as secondary criterion.
    # 3. If accuracy ties and Stage 2 does not improve loss, Stage 1 is retained (simpler frozen backbone).
    selection_rule = (
        "Primary: highest validation accuracy. "
        "Secondary (on tie): lowest validation loss. "
        "Tie-breaker: retain Stage 1 (frozen backbone) if loss does not improve."
    )
    best_overall_checkpoint = os.path.join(output_dir, "best_overall_model.keras")

    if post_val_acc > pre_val_acc:
        shutil.copy2(best_finetuned_checkpoint, best_overall_checkpoint)
        selected_winner = "Stage 2 (Fine-Tuned Model)"
        selection_reason = (
            f"Stage 2 fine-tuning achieved higher validation accuracy "
            f"({post_val_acc * 100:.2f}% vs {pre_val_acc * 100:.2f}%)."
        )
    elif abs(post_val_acc - pre_val_acc) < 1e-5 and post_val_loss < pre_val_loss:
        shutil.copy2(best_finetuned_checkpoint, best_overall_checkpoint)
        selected_winner = "Stage 2 (Fine-Tuned Model)"
        selection_reason = (
            f"Validation accuracy tied at {post_val_acc * 100:.2f}%, but Stage 2 achieved "
            f"lower validation loss ({post_val_loss:.4f} vs {pre_val_loss:.4f}, delta: {delta_loss:.4f})."
        )
    else:
        shutil.copy2(stage1_checkpoint, best_overall_checkpoint)
        selected_winner = "Stage 1 (Frozen Backbone Model)"
        if post_val_acc < pre_val_acc:
            selection_reason = (
                f"Stage 1 retained: Stage 2 validation accuracy dropped "
                f"({post_val_acc * 100:.2f}% vs {pre_val_acc * 100:.2f}%)."
            )
        else:
            selection_reason = (
                f"Stage 1 retained: Validation accuracy tied at {post_val_acc * 100:.2f}%, "
                f"and Stage 2 validation loss did not improve ({post_val_loss:.4f} vs {pre_val_loss:.4f}). "
                "Simpler frozen model preferred to avoid unnecessary parameter complexity."
            )

    # Save Comparison Metrics JSON
    comparison_summary = {
        "fine_tune_at_layer": fine_tune_at,
        "learning_rate": learning_rate,
        "selection_rule": selection_rule,
        "selected_best_model": selected_winner,
        "selection_reason": selection_reason,
        "pre_finetune": {
            "val_loss": pre_val_loss,
            "val_accuracy": pre_val_acc,
        },
        "post_finetune": {
            "val_loss": post_val_loss,
            "val_accuracy": post_val_acc,
        },
        "deltas": {
            "val_loss_change": delta_loss,
            "val_accuracy_change": delta_acc,
        },
        "best_overall_checkpoint": os.path.abspath(best_overall_checkpoint),
        "history": stage2_history,
    }

    comparison_json_path = os.path.join(output_dir, "finetune_comparison.json")
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # 10. Plot Curves
    curves_path = os.path.join(output_dir, "finetune_learning_curves.png")
    stage1_dir = os.path.dirname(os.path.abspath(stage1_checkpoint))
    stage1_hist_path = os.path.join(stage1_dir, "stage1_history.json")
    plot_combined_learning_curves(stage1_hist_path, stage2_history, curves_path)

    # Print Clean Formatted Comparison Report
    print("\n" + "=" * 78)
    print("                VALIDATION PERFORMANCE COMPARISON REPORT")
    print("=" * 78)
    print(f"{'Metric':<25} | {'Pre-Fine-Tuning (Stage 1)':<24} | {'Post-Fine-Tuning (Stage 2)':<24}")
    print("-" * 78)
    print(f"{'Validation Loss':<25} | {pre_val_loss:<24.4f} | {post_val_loss:<24.4f}")
    print(f"{'Validation Accuracy':<25} | {pre_val_acc * 100:<23.2f}% | {post_val_acc * 100:<23.2f}%")
    print("-" * 78)
    acc_sign = "+" if delta_acc >= 0 else ""
    loss_sign = "+" if delta_loss >= 0 else ""
    print(f"Accuracy Delta: {acc_sign}{delta_acc * 100:.2f}% | Loss Delta: {loss_sign}{delta_loss:.4f}")
    print(f"Selection Rule                 : {selection_rule}")
    print(f"Selected Checkpoint Winner     : {selected_winner}")
    print(f"Selection Rationale            : {selection_reason}")
    print(f"Best Validation Model Saved As : {best_overall_checkpoint}")
    print("=" * 78)

    return comparison_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DefectVision MobileNetV2 (Stage 2).")
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default="ml/artifacts/stage1/best_mobilenet_stage1.keras",
        help="Path to best Stage 1 checkpoint",
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
        default="ml/artifacts/finetune",
        help="Directory to save fine-tuned checkpoints and plots",
    )
    parser.add_argument(
        "--fine-tune-at",
        type=int,
        default=100,
        help="Backbone layer index from which to unfreeze (default: 100)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Fine-tuning epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Fine-tuning learning rate (default: 1e-5)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    train_finetune(
        stage1_checkpoint=args.stage1_checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        fine_tune_at=args.fine_tune_at,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
    )
