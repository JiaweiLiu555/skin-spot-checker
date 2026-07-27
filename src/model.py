from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

RGB_ARCHITECTURE = "mobilenet_v3_large_multitask"
CONTOUR_ARCHITECTURE = "contour_aware_mobilenet_v3_large_multitask"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(pretrained: bool = True) -> nn.Module:
    """Build a compact classifier with higher-concern and melanoma logits."""
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


class ContourAwareMobileNetV3(nn.Module):
    """Fuse RGB MobileNet features with a small fixed-Sobel contour stream."""

    architecture = CONTOUR_ARCHITECTURE

    def __init__(
        self,
        pretrained: bool = True,
        source_model: nn.Module | None = None,
        contour_features: int = 64,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        rgb_model = mobilenet_v3_large(weights=weights)
        rgb_model.classifier[-1] = nn.Linear(rgb_model.classifier[-1].in_features, 2)

        if source_model is not None:
            rgb_model.load_state_dict(source_model.state_dict())

        self.features = rgb_model.features
        self.avgpool = rgb_model.avgpool
        self.contour_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, contour_features, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(contour_features),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        rgb_features = rgb_model.classifier[0].in_features
        hidden_features = rgb_model.classifier[0].out_features
        self.classifier = nn.Sequential(
            nn.Linear(rgb_features + contour_features, hidden_features),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(hidden_features, 2),
        )
        self.register_buffer(
            "input_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )

        # Start as the already-validated RGB model. Training can then learn how
        # much the contour stream helps without destroying the source behavior.
        with torch.no_grad():
            self.classifier[0].weight.zero_()
            self.classifier[0].weight[:, :rgb_features].copy_(
                rgb_model.classifier[0].weight
            )
            self.classifier[0].bias.copy_(rgb_model.classifier[0].bias)
            self.classifier[-1].load_state_dict(rgb_model.classifier[-1].state_dict())

    def contour_map(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        rgb = torch.clamp(
            normalized_rgb * self.input_std + self.input_mean, min=0.0, max=1.0
        )
        gray = (
            0.2989 * rgb[:, 0:1]
            + 0.5870 * rgb[:, 1:2]
            + 0.1140 * rgb[:, 2:3]
        )
        horizontal = F.conv2d(gray, self.sobel_x, padding=1)
        vertical = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.clamp(
            (torch.abs(horizontal) + torch.abs(vertical)) / 8.0,
            min=0.0,
            max=1.0,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rgb_features = torch.flatten(self.avgpool(self.features(image)), 1)
        contour_features = torch.flatten(
            self.contour_encoder(self.contour_map(image)), 1
        )
        return self.classifier(torch.cat([rgb_features, contour_features], dim=1))


def build_contour_model(
    pretrained: bool = True,
    source_model: nn.Module | None = None,
) -> ContourAwareMobileNetV3:
    return ContourAwareMobileNetV3(
        pretrained=pretrained,
        source_model=source_model,
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    thresholds: dict[str, float],
    image_size: int,
    metrics: dict[str, float],
    extra_metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": getattr(model, "architecture", RGB_ARCHITECTURE),
        "state_dict": model.state_dict(),
        "thresholds": {name: float(value) for name, value in thresholds.items()},
        "image_size": int(image_size),
        "validation_metrics": metrics,
        "output_classes": ["higher_concern", "melanoma"],
    }
    if extra_metadata:
        checkpoint.update(extra_metadata)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture == RGB_ARCHITECTURE:
        model = build_model(pretrained=False)
    elif architecture == CONTOUR_ARCHITECTURE:
        model = build_contour_model(pretrained=False)
    else:
        raise ValueError("Unsupported checkpoint architecture")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
