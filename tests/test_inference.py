from PIL import Image, ImageDraw

from src.inference import result_label, validate_image


def test_small_image_is_rejected():
    quality = validate_image(Image.new("RGB", (64, 64)))
    assert not quality.accepted


def test_normal_image_is_accepted():
    image = Image.new("RGB", (512, 384), "#b98574")
    draw = ImageDraw.Draw(image)
    draw.ellipse((120, 70, 400, 330), fill="#6f3d37", outline="#2f201f", width=12)
    draw.line((140, 190, 380, 190), fill="#d5a18f", width=8)
    draw.line((260, 90, 260, 310), fill="#d5a18f", width=8)
    quality = validate_image(image)
    assert quality.accepted


def test_threshold_is_inclusive():
    thresholds = {"higher_concern": 0.7, "melanoma": 0.8}
    assert result_label({"higher_concern": 0.76, "melanoma": 0.1}, thresholds) == "Higher concern"
    assert result_label({"higher_concern": 0.2, "melanoma": 0.86}, thresholds) == "Higher concern"
    assert result_label({"higher_concern": 0.60, "melanoma": 0.70}, thresholds) == "Lower concern"


def test_near_boundary_routes_to_followup():
    thresholds = {"higher_concern": 0.7, "melanoma": 0.8}
    label = result_label(
        {"higher_concern": 0.68, "melanoma": 0.4}, thresholds, abstention_margin=0.05
    )
    assert label == "Higher concern"


def test_flat_image_is_rejected():
    quality = validate_image(Image.new("RGB", (512, 384), "#888888"))
    assert not quality.accepted
