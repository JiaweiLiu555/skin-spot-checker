from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from src.inference import predict_scores, result_label, validate_image
from src.model import load_checkpoint

CHECKPOINT_PATH = Path("models/skin_lesion_mobilenet_v3.pt")
ICON_PATH = Path("web/public/icon-192.png")

st.set_page_config(
    page_title="AI4ALL Skin Spot Checker",
    page_icon=Image.open(ICON_PATH) if ICON_PATH.exists() else "🔬",
    layout="centered",
)
st.title("Skin Spot Checker")
st.caption("AI4ALL Medical AI · Educational skin-spot image classifier")

st.error(
    "Not a diagnosis: This classroom prototype cannot confirm or rule out melanoma. "
    "Do not delay or change medical care based on its output. A concerning or changing lesion should be evaluated by a qualified clinician."
)

with st.expander("What this model can and cannot do", expanded=False):
    st.write(
        "The model is designed for the same kind of clinical close-up lesion images used during training. "
        "Its performance on ordinary phone photos, different cameras, lighting, or underrepresented skin tones is unknown. "
        "The displayed score is a model output, not a probability that a person has cancer."
    )


@st.cache_resource
def get_model(path: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(path, device)
    return model, checkpoint, device


if not CHECKPOINT_PATH.exists():
    st.info(
        "The interface is ready, but no trained model is installed yet. Run the dataset-preparation and training commands in README.md. "
        f"The app expects the checkpoint at `{CHECKPOINT_PATH}`."
    )

uploaded = st.file_uploader("Upload one close-up lesion image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.warning("That file could not be read as an image.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)
    quality = validate_image(image)
    if not quality.accepted:
        st.warning(f"Unable to assess: {quality.message}")
        st.stop()
    st.caption(quality.message)

    if not CHECKPOINT_PATH.exists():
        st.warning("Prediction is unavailable until a trained checkpoint is added.")
        st.stop()

    try:
        model, checkpoint, device = get_model(str(CHECKPOINT_PATH))
        thresholds = checkpoint.get("thresholds", {"higher_concern": 0.5, "melanoma": 0.5})
        scores = predict_scores(model, image, device, int(checkpoint.get("image_size", 224)))
    except Exception as error:
        st.error(f"The model could not be loaded: {error}")
        st.stop()

    label = result_label(scores, thresholds)
    if label == "Higher concern":
        st.warning("Review recommended: one or more model outputs crossed a validation-selected cutoff.")
    else:
        st.info("No model flag—cancer is not ruled out. False negatives occurred in testing.")
    st.markdown(
        "**This is not a cancer percentage.** The model only compares image patterns with validation-selected cutoffs."
    )
    with st.expander("Show technical model outputs"):
        st.write(f"Higher-concern output: {scores['higher_concern']:.1%} (cutoff {thresholds['higher_concern']:.1%})")
        st.write(f"Melanoma-pattern output: {scores['melanoma']:.1%} (cutoff {thresholds['melanoma']:.1%})")
        st.caption("These uncalibrated outputs are for project transparency and are not cancer probabilities.")

st.divider()
st.caption("For education and research only · No medical decisions · No images are intentionally stored by this app")
