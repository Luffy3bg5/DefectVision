import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from typing import Optional


class DefectMobileNetV2(nn.Module):
    """
    MobileNetV2 transfer learning model for defect classification.
    Allows fine-tuning or feature extraction by freezing backbone layers.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_mobilenet_v2(
    num_classes: int = 2,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> DefectMobileNetV2:
    """
    Factory function to build and optionally load trained weights for inference or training.
    """
    model = DefectMobileNetV2(num_classes=num_classes, pretrained=pretrained)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Handle checkpoint dicts containing 'model_state_dict' or raw state_dict
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)

    model.to(device)
    return model
