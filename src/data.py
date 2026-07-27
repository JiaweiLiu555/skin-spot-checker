from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(training: bool, image_size: int = IMAGE_SIZE):
    """Return conservative transforms for clinical close-up lesion images."""
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.08),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class LesionDataset(Dataset):
    """Load images from a manifest with image_path and binary label columns."""

    def __init__(self, manifest: str | Path, training: bool, image_size: int = IMAGE_SIZE):
        self.manifest_path = Path(manifest)
        self.frame = pd.read_csv(self.manifest_path)
        required = {"image_path", "label"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        self.transform = build_transform(training=training, image_size=image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = (self.manifest_path.parent / image_path).resolve()
        with Image.open(image_path) as source:
            image = self.transform(source.convert("RGB"))
        higher_concern = float(row["label"])
        melanoma = float(str(row.get("diagnosis", "")).upper() == "MEL")
        targets = torch.tensor([higher_concern, melanoma], dtype=torch.float32)
        return image, targets
