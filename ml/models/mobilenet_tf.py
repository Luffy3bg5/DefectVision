import argparse
from typing import Tuple, Optional


def build_mobilenet_v2_classifier(
    num_classes: int = 6,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.2,
    freeze_backbone: bool = True,
    weights: str = "imagenet",
):
    """
    Builds the DefectVision MobileNetV2 transfer learning model in TensorFlow/Keras.
    
    Architecture:
      1. Input Layer (224, 224, 3)
      2. Pretrained MobileNetV2 Backbone (without original 1000-class head)
      3. Frozen backbone weights (feature extraction mode)
      4. GlobalAveragePooling2D (collapses 7x7x1280 feature maps -> 1280 vector)
      5. Dropout (regularization against overfitting)
      6. Dense Softmax Classification Head (outputs 6-class defect probabilities)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    # 1. Input Layer
    inputs = layers.Input(shape=input_shape, name="input_image")

    # 2. Pretrained MobileNetV2 Backbone (Feature Extractor)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )

    # 3. Freeze Backbone
    # Setting trainable=False prevents weight updates to pretrained ImageNet filters
    base_model.trainable = not freeze_backbone

    # Pass inputs through base model in inference mode (keeps BatchNorm running stats stable)
    x = base_model(inputs, training=False)

    # 4. Global Average Pooling Layer
    # Reduces spatial tensor (batch, 7, 7, 1280) to 1D vector (batch, 1280)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)

    # 5. Dropout Layer for Regularization
    # Randomly sets 20% of activations to 0 during training to prevent co-adaptation
    x = layers.Dropout(rate=dropout_rate, name="dropout_regularization")(x)

    # 6. Six-class Softmax Classification Head
    # Outputs probability distribution across the 6 defect classes: sum(probs) = 1.0
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="defect_classification_head",
    )(x)

    # Assemble end-to-end model
    model = Model(inputs=inputs, outputs=outputs, name="DefectVision_MobileNetV2")

    return model, base_model


def unfreeze_backbone_layers(model, fine_tune_at: int = 100) -> Tuple[int, int, int]:
    """
    Unfreezes the top layers of the MobileNetV2 backbone for Stage 2 fine-tuning.
    
    Keeps layers 0 through fine_tune_at-1 FROZEN (generic low-level visual features).
    Unfreezes layers fine_tune_at through the end of the backbone (task-specific adaptation).

    Returns:
        (total_params, trainable_params, non_trainable_params)
    """
    import tensorflow as tf
    from tensorflow.keras import Model

    # Locate the MobileNetV2 base model layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find MobileNetV2 backbone sub-model inside model layers.")

    # Unfreeze the base model container
    base_model.trainable = True

    total_backbone_layers = len(base_model.layers)
    if fine_tune_at > total_backbone_layers:
        raise ValueError(
            f"fine_tune_at ({fine_tune_at}) exceeds total backbone layers ({total_backbone_layers})."
        )

    # Freeze earlier layers; unfreeze later layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    total_count = trainable_count + non_trainable_count

    return total_count, trainable_count, non_trainable_count



def get_architecture_summary_text() -> str:
    """
    Provides deterministic architectural summary and parameter specifications.
    """
    summary = r"""
==================================================================================================
Model: "DefectVision_MobileNetV2" (TensorFlow / Keras Transfer Learning Architecture)
==================================================================================================
 Layer (type)                             Output Shape              Param #     Trainable Status   
==================================================================================================
 input_image (InputLayer)                 [(None, 224, 224, 3)]     0           Non-trainable      
                                                                                                  
 mobilenetv2_1.00_224 (Functional Backbone) (None, 7, 7, 1280)       2,257,984   FROZEN (False)     
   |-- Inverted Residual Bottlenecks (17 blocks)                                                  
   |-- Depthwise Separable Convolutions                                                           
   \-- Linear Bottleneck Projections                                                              
                                                                                                  
 global_average_pooling2d (GlobalAvgPool2D) (None, 1280)             0           Non-trainable      
                                                                                                  
 dropout_regularization (Dropout p=0.2)   (None, 1280)              0           Non-trainable      
                                                                                                  
 defect_classification_head (Dense)       (None, 6)                 7,686       TRAINABLE (True)   
   |-- Weights: (1280, 6) = 7,680 params                                                          
   |-- Bias   : (6,)       = 6 params                                                             
   \-- Activation: Softmax                                                                        
==================================================================================================
Total params: 2,265,670 (8.64 MB)
Trainable params: 7,686 (30.02 KB)
Non-trainable params: 2,257,984 (8.61 MB)
==================================================================================================
Target NEU Defect Classes:
  0: crazing (Cr)
  1: inclusion (In)
  2: patches (Pa)
  3: pitted_surface (PS)
  4: rolled-in_scale (RS)
  5: scratches (Sc)
==================================================================================================
"""
    return summary


def display_model_summary(num_classes: int = 6, dropout_rate: float = 0.2):
    """
    Builds the model (or uses static breakdown if TF runtime is missing) and prints the summary.
    """
    try:
        import tensorflow as tf
        print(f"TensorFlow {tf.__version__} detected. Instantiating Keras MobileNetV2...")
        model, base_model = build_mobilenet_v2_classifier(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=True,
        )
        print("\n" + "=" * 90)
        print("                KERAS FUNCTIONAL MODEL SUMMARY")
        print("=" * 90)
        model.summary(expand_nested=False, line_length=90)

        trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

        print("-" * 90)
        print(f"Base Model Backbone Frozen : {not base_model.trainable}")
        print(f"Total Trainable Parameters : {trainable_count:,} (Only Classification Head)")
        print(f"Non-Trainable Parameters   : {non_trainable_count:,} (ImageNet Backbone Frozen)")
        print("=" * 90)
    except Exception as e:
        print(f"[Note] Detailed architectural breakdown (TensorFlow import fallback: {e}):")
        print(get_architecture_summary_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and summarize DefectVision MobileNetV2 model.")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of defect classes (default: 6)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability rate (default: 0.2)")
    args = parser.parse_args()

    display_model_summary(num_classes=args.num_classes, dropout_rate=args.dropout)
