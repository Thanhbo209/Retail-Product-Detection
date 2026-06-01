"""Streamlit demo app for retail product detection."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image


DEFAULT_MODEL_PATH = Path("runs/experiments/product_yolov8n/weights/best.pt")
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


def discover_model_paths() -> list[str]:
    candidates = [DEFAULT_MODEL_PATH]
    runs_dir = Path("runs")
    if runs_dir.exists():
        candidates.extend(sorted(runs_dir.rglob("*.pt")))

    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        path_text = str(path)
        if path_text not in seen:
            unique_paths.append(path_text)
            seen.add(path_text)
    return unique_paths


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path)


def image_to_temp_file(image: Image.Image, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        image.save(temp_file.name)
        return temp_file.name


def class_name_from_model(model_names: Any, class_id: int) -> str:
    if isinstance(model_names, dict):
        return str(model_names.get(class_id, f"class_{class_id}"))
    if isinstance(model_names, list) and 0 <= class_id < len(model_names):
        return str(model_names[class_id])
    return f"class_{class_id}"


def result_to_record(result: Any, image_name: str, model_names: Any) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            bbox_xyxy = [int(round(value)) for value in box.xyxy[0].tolist()]
            class_name = class_name_from_model(model_names, class_id)

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": bbox_xyxy,
                }
            )

    counts = Counter(detection["class_name"] for detection in detections)
    return {
        "image": image_name,
        "detections": detections,
        "counts": dict(sorted(counts.items())),
    }


def run_prediction(
    model: Any,
    image: Image.Image,
    image_name: str,
    confidence: float,
) -> tuple[dict[str, Any], Image.Image]:
    suffix = Path(image_name).suffix or ".jpg"
    temp_image_path = image_to_temp_file(image, suffix)

    try:
        results = model.predict(
            source=temp_image_path,
            conf=confidence,
            verbose=False,
        )
    finally:
        Path(temp_image_path).unlink(missing_ok=True)
    result = results[0]
    record = result_to_record(result, image_name, getattr(model, "names", {}))

    # Ultralytics returns an OpenCV-style BGR array from result.plot().
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_image = Image.fromarray(annotated_rgb)
    return record, annotated_image


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def counts_to_dataframe(record: dict[str, Any]) -> pd.DataFrame:
    counts = record["counts"]
    if not counts:
        return pd.DataFrame(columns=["class_name", "count"])

    return pd.DataFrame(
        [{"class_name": class_name, "count": count} for class_name, count in counts.items()]
    )


def main() -> None:
    st.set_page_config(page_title="Retail Product Detection", layout="wide")
    st.title("Retail Product Detection")

    model_candidates = discover_model_paths()
    default_model_text = model_candidates[0] if model_candidates else str(DEFAULT_MODEL_PATH)

    with st.sidebar:
        st.header("Settings")
        selected_model = st.selectbox("Model path preset", model_candidates)
        model_path_text = st.text_input("Model path", value=selected_model or default_model_text)

        confidence = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=SUPPORTED_IMAGE_TYPES,
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    model_path = Path(model_path_text)

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return

    if st.button("Run detection", type="primary"):
        try:
            with st.spinner("Running inference..."):
                model = load_model(str(model_path))
                record, annotated_image = run_prediction(
                    model=model,
                    image=image,
                    image_name=uploaded_file.name,
                    confidence=confidence,
                )
        except ImportError:
            st.error("ultralytics is not installed. Run: pip install -r requirements.txt")
            return
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

        with right_column:
            st.subheader("Annotated")
            st.image(annotated_image, use_container_width=True)
            st.download_button(
                label="Download annotated image",
                data=image_to_png_bytes(annotated_image),
                file_name=f"{Path(uploaded_file.name).stem}_annotated.png",
                mime="image/png",
            )

        st.subheader("Product Counts")
        counts_table = counts_to_dataframe(record)
        if counts_table.empty:
            st.write("No detections.")
        else:
            st.dataframe(counts_table, hide_index=True, use_container_width=True)

        st.subheader("Prediction JSON")
        st.json(json.loads(json.dumps(record)))


if __name__ == "__main__":
    main()
