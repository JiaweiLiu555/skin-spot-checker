from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageStat

from src.data import build_transform
from src.metrics import multitask_decision_scores


@dataclass(frozen=True)
class ImageQuality:
    accepted: bool
    message: str


def validate_image(image: Image.Image) -> ImageQuality:
    width, height = image.size
    if width < 128 or height < 128:
        return ImageQuality(False, "The image is too small. Use at least 128 × 128 pixels.")
    ratio = max(width, height) / min(width, height)
    if ratio > 4:
        return ImageQuality(False, "The image is unusually narrow. Upload a close-up centered on one lesion.")
    grayscale = image.convert("L").resize((128, 128))
    brightness = ImageStat.Stat(grayscale).mean[0]
    if brightness < 35:
        return ImageQuality(False, "The image is too dark. Retake it in bright, even light.")
    if brightness > 225:
        return ImageQuality(False, "The image is overexposed. Retake it without glare or flash washout.")
    edge_pixels = np.asarray(grayscale.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    edge_variance = float(edge_pixels[2:-2, 2:-2].var())
    if edge_variance < 20:
        return ImageQuality(False, "The image has too little visible detail. Retake it in focus and closer to the lesion.")
    return ImageQuality(True, "Image passed basic technical checks.")


def predict_scores(model, image: Image.Image, device: torch.device, image_size: int = 224) -> dict[str, float]:
    tensor = build_transform(training=False, image_size=image_size)(image.convert("RGB"))
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0).to(device)).squeeze(0)
        scores = torch.sigmoid(logits).cpu().tolist()
        return {"higher_concern": float(scores[0]), "melanoma": float(scores[1])}


def result_label(
    scores: dict[str, float], thresholds: dict[str, float], abstention_margin: float = 0.10
) -> str:
    decision_margin = multitask_decision_scores(
        [[scores["higher_concern"], scores["melanoma"]]], thresholds
    )[0]
    # Borderline but technically usable images are routed to follow-up rather
    # than shown as reassuring or left without a decision.
    if abs(decision_margin) < abstention_margin:
        return "Higher concern"
    return "Higher concern" if decision_margin >= 0 else "Lower concern"
