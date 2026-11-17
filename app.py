"""Streamlit web application for Bangla Handwritten Whole-Word Recognition."""

import json
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_DIR = Path("models/hf_model")
LABELS_PATH = Path("labels.json")
PRE_BASE_KAR = {"ি", "ে", "ৈ"}


@st.cache_resource
def load_model_and_labels():
    """Load the fine-tuned Hugging Face model, image processor, and label map."""
    if not MODEL_DIR.exists():
        return None, None, None
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    model.eval()
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels_map = json.load(f)
    return processor, model, labels_map


def pad_to_square(image: Image.Image, fill: int = 0) -> Image.Image:
    """Pad an image to a square while preserving aspect ratio."""
    width, height = image.size
    max_dim = max(width, height)
    pad_w = (max_dim - width) // 2
    pad_h = (max_dim - height) // 2
    padding = (pad_w, pad_h, max_dim - width - pad_w, max_dim - height - pad_h)
    return ImageOps.expand(image, padding, fill=fill)


def segment_word_into_characters(
    canvas_rgba: np.ndarray,
    min_char_width: int = 10,
    min_ink_pixels: int = 25,
) -> list[Image.Image]:
    """Segment a drawn Bangla word into left-to-right character crops.

    Uses vertical projection below the top matra band so characters connected
    along the matra line can still be separated cleanly.
    """
    gray = cv2.cvtColor(canvas_rgba.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    inv = 255 - gray
    _, binary = cv2.threshold(inv, 40, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return []

    x, y, w, h = cv2.boundingRect(coords)
    word_crop = inv[y : y + h, x : x + w]
    word_bin = binary[y : y + h, x : x + w]

    # Analyze vertical projection below the top matra region (~top 25% of height)
    matra_cut = int(h * 0.28) if h > 40 else 0
    body_bin = word_bin[matra_cut:, :] if matra_cut < h - 5 else word_bin
    col_sum = (body_bin > 0).sum(axis=0)

    # Identify active columns with ink
    active = col_sum > max(1, int(0.02 * body_bin.shape[0]))

    segments: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    gap_tolerance = max(2, int(h * 0.03))
    gap_count = 0

    for col_idx, has_ink in enumerate(active):
        if has_ink:
            if not in_seg:
                start = col_idx
                in_seg = True
            gap_count = 0
        elif in_seg:
            gap_count += 1
            if gap_count > gap_tolerance:
                end = col_idx - gap_count + 1
                if end - start >= min_char_width:
                    segments.append((start, end))
                in_seg = False
                gap_count = 0

    if in_seg:
        end = len(active)
        if end - start >= min_char_width:
            segments.append((start, end))

    crops: list[Image.Image] = []
    for x1, x2 in segments:
        char_bin = word_bin[:, x1:x2]
        if (char_bin > 0).sum() < min_ink_pixels:
            continue
        char_coords = cv2.findNonZero(char_bin)
        if char_coords is None:
            continue
        cx, cy, cw, ch = cv2.boundingRect(char_coords)
        char_patch = word_crop[cy : cy + ch, x1 + cx : x1 + cx + cw]

        pil_char = Image.fromarray(char_patch).convert("L")
        margin = max(4, int(max(pil_char.size) * 0.15))
        pil_char = ImageOps.expand(pil_char, border=margin, fill=0)
        pil_char = pad_to_square(pil_char, fill=0).convert("RGB")
        crops.append(pil_char)

    return crops


def join_bangla_characters(chars: list[str]) -> str:
    """Join predicted Bangla characters into a normalized whole word."""
    reordered: list[str] = []
    i = 0
    while i < len(chars):
        if chars[i] in PRE_BASE_KAR and i + 1 < len(chars):
            reordered.append(chars[i + 1])
            reordered.append(chars[i])
            i += 2
        else:
            reordered.append(chars[i])
            i += 1
    return unicodedata.normalize("NFC", "".join(reordered))


def predict_word(
    crops: list[Image.Image],
    processor: AutoImageProcessor,
    model: AutoModelForImageClassification,
) -> tuple[str, list[tuple[str, float]]]:
    """Classify each segmented character crop and join into a whole Bangla word."""
    inputs = processor(images=crops, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        confs, pred_ids = probs.max(dim=-1)

    char_preds: list[tuple[str, float]] = []
    for pred_id, conf in zip(pred_ids.tolist(), confs.tolist()):
        label = model.config.id2label.get(pred_id) or model.config.id2label.get(str(pred_id), "?")
        char_preds.append((label, float(conf)))

    whole_word = join_bangla_characters([c for c, _ in char_preds])
    return whole_word, char_preds


def main() -> None:
    st.set_page_config(page_title="Bangla Handwritten Word Recognition", page_icon="✍️", layout="centered")
    st.title("Bangla Handwritten Word Recognition")
    st.caption("Draw a Bangla word below (e.g., জল, মানুষ, কলম) and click **Predict Word**.")

    processor, model, _ = load_model_and_labels()
    if model is None:
        st.error(
            f"Trained model not found at `{MODEL_DIR}`. "
            "Run `python train.py` (or place your trained `models/hf_model/` directory) first."
        )
        st.stop()

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=10,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=220,
        width=680,
        drawing_mode="freedraw",
        key="bangla_word_canvas",
    )

    if st.button("Predict Word", type="primary"):
        if canvas_result.image_data is None:
            st.warning("Please draw a Bangla word on the canvas first.")
            st.stop()

        crops = segment_word_into_characters(canvas_result.image_data)
        if not crops:
            st.warning("No handwriting detected. Please draw a Bangla word clearly.")
            st.stop()

        whole_word, char_preds = predict_word(crops, processor, model)

        st.subheader("Predicted Word")
        st.markdown(
            f"<div style='font-size: 56px; font-weight: 700; padding: 12px 20px; "
            f"background-color: #f0f2f6; border-radius: 10px; text-align: center; color: #111;'>"
            f"{whole_word}</div>",
            unsafe_allow_html=True,
        )

        with st.expander(f"Segmented Characters ({len(crops)} detected)"):
            cols = st.columns(len(crops))
            for col, crop, (ch, conf) in zip(cols, crops, char_preds):
                with col:
                    st.image(crop, use_container_width=True)
                    st.markdown(f"**{ch}** ({conf:.0%})")


if __name__ == "__main__":
    main()
