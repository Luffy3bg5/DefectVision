# DefectVision - Isolated Deep Learning (ML) Component

This directory provides an isolated deep learning pipeline for defect classification using **MobileNetV2**. It is completely decoupled from the existing OpenCV pipelines (`spatial`, `edge_shape`, `frequency`, `texture`, and `consensus`).

## Directory Structure

```text
ml/
├── data/              # Dataset loading, splits, and data augmentation pipelines
├── models/            # MobileNetV2 architecture definition & transfer learning head
├── training/          # Training loop, LR scheduling, and checkpoint saving
├── evaluation/        # Validation & test evaluation (Accuracy, F1, Confusion Matrix)
├── artifacts/         # Saved model checkpoints (.pt, .onnx) and metrics reports
├── requirements-ml.txt# Independent Python dependencies for deep learning
└── README.md
```

## Quickstart

### 1. Install ML Dependencies
```bash
pip install -r ml/requirements-ml.txt
```

### 2. Dataset Structure
Organize your dataset inside a folder with standard train/val/test splits:
```text
dataset/
├── train/
│   ├── defective/
│   └── normal/
├── val/
│   ├── defective/
│   └── normal/
└── test/
    ├── defective/
    └── normal/
```

### 3. Training
```bash
python -m ml.training.train --data-dir path/to/dataset --epochs 20 --batch-size 32
```
Trained weights will be saved to `ml/artifacts/best_mobilenet_v2.pt`.

### 4. Evaluation
```bash
python -m ml.evaluation.evaluate --checkpoint ml/artifacts/best_mobilenet_v2.pt --data-dir path/to/dataset --split test
```
Evaluation metrics will be saved to `ml/artifacts/eval_metrics_test.json`.
