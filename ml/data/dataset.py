import os
from typing import Tuple, Dict, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Returns standard image augmentations and normalization transforms for MobileNetV2.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return {
        "train": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]),
        "val": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]),
        "test": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]),
    }


class DefectDataset(Dataset):
    """
    Custom Dataset loader for defect image datasets.
    Supports directory-based structure: root_dir/<split>/<class_name>/*.png
    or flat lists of image paths and labels.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "train",
        manifest_path: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        import json
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        # 1. Load from JSON manifest if specified
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            self.classes = manifest_data["class_mapping"]["classes"]
            self.class_to_idx = manifest_data["class_mapping"]["class_to_idx"]
            split_items = manifest_data["splits"].get(split, [])
            for item in split_items:
                self.samples.append((item["path"], item["label"]))
        # 2. Or load from standard directory structure root_dir/<split>/<class_name>
        elif root_dir:
            self.split_dir = os.path.join(root_dir, split)
            if os.path.exists(self.split_dir):
                self.classes = sorted([
                    d for d in os.listdir(self.split_dir)
                    if os.path.isdir(os.path.join(self.split_dir, d))
                ])
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

                for cls_name in self.classes:
                    cls_folder = os.path.join(self.split_dir, cls_name)
                    for fname in os.listdir(cls_folder):
                        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                            self.samples.append(
                                (os.path.join(cls_folder, fname), self.class_to_idx[cls_name])
                            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
) -> Dict[str, DataLoader]:
    """
    Creates train, val, and test DataLoaders from dataset folder.
    """
    transform_dict = get_transforms(image_size=image_size)
    loaders = {}

    for split in ["train", "val", "test"]:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            dataset = DefectDataset(
                root_dir=data_dir,
                split=split,
                transform=transform_dict[split],
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

    return loaders
