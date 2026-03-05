"""
app.py — Gradio web interface for the Forensic Audio Authentication System.

Features:
    - Audio upload (wav / flac / mp3)
    - Real-time tampering probability + verdict
    - Grad-CAM spectrogram overlay
    - SHAP feature importance bar chart
    - Exportable PDF court report

Usage:
    python app/app.py
"""

import os
import sys
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference     import predict
from xai.gradcam       import generate_gradcam_plot
from xai.shap_explain  import explain_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("app")

_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

TEMP_DIR = tempfile.mkdtemp(prefix="forensic_audio_")


def _temp_path(suffix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(TEMP_DIR, f"{ts}{suffix}")


def analyze_audio(audio_path: str) -> tuple:
    """
    Full analysis pipeline called by Gradio on each submission.

    Args:
        audio_path: Path provided by gr.Audio component.

    Returns:
        (verdict_md, gradcam_img_path, shap_img_path, status_msg)
    """
    if audio_path is None:
        return (
            "## ⚠️ No file uploaded",
            None,
            None,
            "Please upload an audio file to begin analysis.",
        )

    logger.info("Analyzing: %s", audio_path)

    try:
        # ── 1. Prediction ──────────────────────────────────────────────
        result        = predict(audio_path)
        verdict       = result["verdict"]
        tampered_pct  = result["tampered_prob"]  * 100
        authentic_pct = result["authentic_prob"] * 100
        confidence    = result["confidence"]     * 100

        verdict_emoji = "🔴 TAMPERED" if verdict == "TAMPERED" else "🟢 AUTHENTIC"
        risk_level    = (
            "HIGH RISK"   if tampered_pct >= 70 else
            "MEDIUM RISK" if tampered_pct >= 40 else
            "LOW RISK"
        )

        verdict_md = f"""
## Verdict: {verdict_emoji}

| Metric | Value |
|--------|-------|
| **Tampered Probability**  | {tampered_pct:.1f}% |
| **Authentic Probability** | {authentic_pct:.1f}% |
| **Confidence**            | {confidence:.1f}% |
| **Risk Level**            | {risk_level} |

*Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # ── 2. Grad-CAM ────────────────────────────────────────────────
        gradcam_path = _temp_path("_gradcam.png")
        try:
            generate_gradcam_plot(audio_path, output_path=gradcam_path)
        except Exception as exc:
            logger.error("Grad-CAM failed: %s", exc)
            gradcam_path = None

        # ── 3. SHAP ────────────────────────────────────────────────────
        shap_path = _temp_path("_shap.png")
        try:
            explain_prediction(audio_path, output_path=shap_path)
        except Exception as exc:
            logger.error("SHAP failed: %s", exc)
            shap_path = None

        status = f"✅ Analysis complete — {os.path.basename(audio_path)}"
        return verdict_md, gradcam_path, shap_path, status

    except FileNotFoundError as exc:
        msg = f"## ❌ Model Not Found\n\n{exc}\n\nRun `python model/train.py` first."
        return msg, None, None, "Error: model not found"

    except Exception as exc:
        logger.exception("Unexpected error during analysis")
        msg = f"## ❌ Analysis Failed\n\n```\n{exc}\n```"
        return msg, None, None, f"Error: {exc}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
.verdict-box { border-radius: 12px; padding: 16px; }
.gr-button-primary { background: #1a1a2e !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Forensic Audio Authentication System",
    theme=gr.themes.Base(
        primary_hue="slate",
        secondary_hue="blue",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
    css=CSS,
) as demo:

    gr.Markdown(
        """
        # 🎙️ Forensic Audio Authentication System
        **Deep Learning-based tamper detection for legal and investigative use.**
        Upload an audio file to detect splicing, speed manipulation, or deepfake injection.
        All predictions are accompanied by explainable AI visualizations for court admissibility.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload Audio File",
                type="filepath",
                sources=["upload"],
            )
            analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")
            status_box  = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Column(scale=2):
            verdict_out = gr.Markdown(value="*Upload a file and click Analyze to begin.*")

    gr.Markdown("---")

    with gr.Row():
        gradcam_out = gr.Image(
            label="Grad-CAM — Suspicious Frequency-Time Regions",
            type="filepath",
        )
        shap_out = gr.Image(
            label="SHAP — Feature Importance per LFCC Coefficient",
            type="filepath",
        )

    gr.Markdown(
        """
        ---
        ### How to interpret results
        - **Grad-CAM**: Brighter/hotter regions indicate frequency-time areas the model
          flagged as suspicious — typical edit boundaries appear as vertical bright bands.
        - **SHAP**: Taller bars indicate LFCC coefficients that contributed most to the
          tampering prediction. High-frequency coefficients (LFCC-15 to LFCC-19) are
          particularly sensitive to deepfake artifacts.
        - **EER target**: < 5% | **FAR target**: 1% (judicial standard)
        """
    )

    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input],
        outputs=[verdict_out, gradcam_out, shap_out, status_box],
    )

    gr.Examples(
        examples=[],
        inputs=[audio_input],
        label="Example Files (add .wav files to app/examples/)",
    )


if __name__ == "__main__":
    port  = int(_cfg["app"]["port"])
    share = bool(_cfg["app"]["share"])
    logger.info("Launching Forensic Audio Authentication app on port %d", port)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )
