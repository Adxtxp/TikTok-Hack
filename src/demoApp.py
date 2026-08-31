"""
src/demo_app.py
---------------
Interactive Gradio demo for the AI-image detector.

Upload an image -> get a REAL / AI-GENERATED verdict with a confidence bar.
Optionally apply a degradation transform to see robustness live.

Runs on a GPU (Colab). Reuses the trained head + fused DINOv2/FFT features —
no model logic is redefined here.

Usage (in a Colab cell, from the repo root, after the repo is cloned and
head_augmented.pt exists in outputs/):

    !pip install -q gradio
    !python -m src.demo_app            # launches with a public share link

or import and call launch_demo() from a notebook cell.
"""

import numpy as np
from PIL import Image

from src.features import get_fused_embeddings, DinoV2Embedder, fft_feature_vector
from src.model import load_head
from src.transforms import apply_transform, TRANSFORM_NAMES

# ---- load the model once at import ----------------------------------------
HEAD_PATH = "outputs/head_augmented.pt"
_head = None
_embedder = None


def _get_models():
    """Lazy-load the head and embedder once, reuse across calls."""
    global _head, _embedder
    if _head is None:
        _head = load_head(HEAD_PATH)
    if _embedder is None:
        _embedder = DinoV2Embedder()
    return _head, _embedder


def _embed_single_pil(img_pil):
    """Fuse DINOv2 + FFT features for one already-loaded PIL image.
    Mirrors get_fused_embeddings but works on an in-memory image so the
    Gradio upload doesn't need to be written to disk first."""
    head, embedder = _get_models()
    img_pil = img_pil.convert("RGB")
    dino = embedder.embed_batch([img_pil])          # [1, 768]
    fft = np.asarray(fft_feature_vector(img_pil), dtype=np.float32)[None, :]  # [1, 32]
    fused = np.concatenate([dino, fft], axis=1)      # [1, 800]
    return fused


def predict(image, transform_name):
    """Gradio callback. image: PIL image from the upload. transform_name: str.
    Returns (annotated verdict string, confidence-label dict, transformed image)."""
    if image is None:
        return "Upload an image to begin.", {}, None

    # optionally degrade the image first (to demo robustness live)
    shown = image.convert("RGB")
    if transform_name and transform_name != "clean":
        shown = apply_transform(shown, transform_name)

    head, _ = _get_models()
    feats = _embed_single_pil(shown)
    prob = float(head.predict_proba(feats)[0])       # P(AI-generated)

    verdict = "AI-GENERATED" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else (1.0 - prob)
    emoji = "🤖" if prob > 0.5 else "📷"
    header = f"{emoji}  {verdict}  —  {confidence:.0%} confident"

    # gradio Label component wants {class: score}
    label = {"AI-generated": prob, "Real": 1.0 - prob}
    return header, label, shown


def launch_demo(share=True):
    import gradio as gr

    transform_choices = ["clean"] + [t for t in TRANSFORM_NAMES if t != "clean"]

    with gr.Blocks(title="AI Image Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Robust AI-Generated Image Detector")
        gr.Markdown(
            "Upload an image to check whether it's **AI-generated** or a **real photo**. "
            "Try applying a degradation below to see how the detector holds up under "
            "compression, blur, noise, and resizing."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Upload an image")
                transform = gr.Dropdown(
                    choices=transform_choices, value="clean",
                    label="Apply degradation (optional — test robustness)",
                )
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                verdict = gr.Markdown("### Result will appear here")
                score = gr.Label(label="Probability", num_top_classes=2)
                shown = gr.Image(type="pil", label="Image analyzed (after degradation)")

        btn.click(predict, inputs=[inp, transform], outputs=[verdict, score, shown])
        # also run when the transform changes, so the slider feels live
        transform.change(predict, inputs=[inp, transform], outputs=[verdict, score, shown])

    demo.launch(share=share)


if __name__ == "__main__":
    launch_demo(share=True)
