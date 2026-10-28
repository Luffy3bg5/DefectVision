import os
import json
import argparse


def export_defectvision_model(
    output_dir: str = "ml/artifacts/model",
    num_classes: int = 6,
    model_filename: str = "defectvision_mobilenet_v2.keras",
):
    """
    Constructs the MobileNetV2 architecture with ImageNet weights, initializes the 6-class
    classification head, and serializes the complete model to Keras v3 .keras format.
    """
    import tensorflow as tf
    from ml.models.mobilenet_tf import build_mobilenet_v2_classifier

    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, model_filename)

    print(f"Building DefectVision MobileNetV2 model (classes={num_classes})...")
    model, _ = build_mobilenet_v2_classifier(
        num_classes=num_classes,
        input_shape=(224, 224, 3),
        dropout_rate=0.2,
        freeze_backbone=True,
    )

    print(f"Serializing model to Keras .keras format at: {model_save_path}")
    model.save(model_save_path)
    file_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
    print(f"[Success] Model saved successfully ({file_size_mb:.2f} MB)")

    return model_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DefectVision MobileNetV2 in .keras format.")
    parser.add_argument("--output-dir", type=str, default="ml/artifacts/model", help="Directory to save model")
    parser.add_argument("--filename", type=str, default="defectvision_mobilenet_v2.keras", help="Model filename")
    args = parser.parse_args()

    export_defectvision_model(output_dir=args.output_dir, model_filename=args.filename)
