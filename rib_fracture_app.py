"""
Streamlit page for Rib Fracture Analysis using YOLOv8.
"""

import streamlit as st
import json
import io
from datetime import datetime
from typing import Dict

from PIL import Image

from rib_fracture_analyzer import (
    RibFractureAnalyzer,
    SEVERITY_LEVELS,
    generate_report,
    YOLO_AVAILABLE,
)


def inject_styles():
    """Inject CSS styles for rib fracture page."""
    st.markdown("""
    <style>
        .rf-header {
            background: linear-gradient(135deg, #1E3A5F, #2563EB);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .rf-disclaimer {
            background-color: #FEE2E2;
            border: 1px solid #EF4444;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .rf-flag {
            background-color: #FEF3C7;
            border-left: 4px solid #F59E0B;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 0 5px 5px 0;
        }
        .severity-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            color: white;
        }
        .severity-mild { background-color: #F59E0B; }
        .severity-moderate { background-color: #F97316; }
        .severity-severe { background-color: #EF4444; }
        .stat-card {
            background-color: #F3F4F6;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
        }
        .detection-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        .detection-table th {
            background-color: #1F2937;
            color: white;
            padding: 0.6rem 1rem;
            text-align: left;
        }
        .detection-table td {
            padding: 0.5rem 1rem;
            border-bottom: 1px solid #E5E7EB;
        }
    </style>
    """, unsafe_allow_html=True)


def display_header():
    st.markdown("""
    <div class="rf-header">
        <h1>Rib Fracture Detector</h1>
        <p>AI-Powered Rib Fracture Detection using YOLOv8</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">
            Object Detection &bull; Severity Classification &bull; Educational & Research Use Only
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_disclaimer():
    st.markdown("""
    <div class="rf-disclaimer">
        <strong>Medical Disclaimer:</strong> This tool is for <b>educational and research
        purposes only</b>. Rib fracture detection results must be validated by a qualified
        radiologist. Do not use for clinical diagnosis or treatment decisions.
    </div>
    """, unsafe_allow_html=True)


def display_upload_section() -> Image.Image | None:
    st.header("Upload Chest Image")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Choose a chest X-ray or CT image",
            type=["png", "jpg", "jpeg", "tiff", "bmp", "dcm"],
            help="Upload a high-quality chest radiograph (max 10 MB)",
        )

        if uploaded is not None:
            if uploaded.size > 10 * 1024 * 1024:
                st.error("File exceeds 10 MB limit.")
                return None

            image = Image.open(uploaded)
            st.session_state.rf_uploaded_image = image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.info(
                f"**Image Details:**\n"
                f"- Size: {image.size[0]} x {image.size[1]} px\n"
                f"- Mode: {image.mode}\n"
                f"- File size: {uploaded.size / 1024:.1f} KB"
            )
            return image

    with col2:
        st.markdown("""
        ### Guidelines
        - Use PA (posteroanterior) chest X-rays for best results
        - CT scout images are also supported
        - Ensure adequate contrast
        - DICOM support is experimental

        ### Supported Formats
        - PNG, JPG, JPEG
        - TIFF, BMP
        - Max size: 10 MB
        """)

    return None


def display_analysis_controls(analyzer: RibFractureAnalyzer, image: Image.Image):
    st.header("Fracture Detection")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        conf_thresh = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Minimum detection confidence to display",
        )

    with col_b:
        model_status = "YOLOv8 loaded" if analyzer.model else "Demo mode (model not loaded)"
        st.caption(f"Model status: {model_status}")

    if st.button("Run Fracture Detection", type="primary", use_container_width=True):
        with st.spinner("Analyzing image for rib fractures..."):
            import time
            progress = st.progress(0)
            for pct in range(100):
                time.sleep(0.05)
                progress.progress(pct + 1)

            result = analyzer.analyze(image, confidence_threshold=conf_thresh)
            st.session_state.rf_result = result
            st.success("Detection complete!")


def display_results(result: Dict, image: Image.Image, analyzer: RibFractureAnalyzer):
    st.header("Detection Results")

    # ---- Summary metrics ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Fractures Found", result["fracture_count"])
    with c2:
        st.metric("Overall Confidence", f"{result['overall_confidence']:.1f}%")
    with c3:
        severe_count = result["severity_summary"].get("severe", 0)
        st.metric("Severe", severe_count, delta=None)
    with c4:
        st.metric("Clinical Flags", len(result["clinical_flags"]))

    # ---- Annotated image ----
    st.subheader("Annotated Image")
    annotated = analyzer.draw_annotations(image, result)
    st.image(annotated, caption="Detected Fractures", use_column_width=True)

    # ---- Detection table ----
    if result["detections"]:
        st.subheader("Detection Details")
        table_html = '<table class="detection-table"><tr><th>#</th><th>Rib</th><th>Side</th><th>Severity</th><th>Confidence</th></tr>'
        for i, det in enumerate(result["detections"], 1):
            sev_cls = f"severity-{det['severity']}"
            table_html += (
                f"<tr><td>{i}</td>"
                f"<td>{det['rib']}</td>"
                f"<td>{det['side']}</td>"
                f'<td><span class="severity-badge {sev_cls}">{det["severity_label"]}</span></td>'
                f"<td>{det['confidence']:.1f}%</td></tr>"
            )
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No rib fractures detected on this image.")

    # ---- Clinical flags ----
    if result["clinical_flags"]:
        st.subheader("Clinical Alerts")
        for flag in result["clinical_flags"]:
            st.markdown(f'<div class="rf-flag">{flag}</div>', unsafe_allow_html=True)

    # ---- Interpretation ----
    st.subheader("Clinical Interpretation")
    st.write(result["interpretation"])

    st.caption(f"Analysis performed at: {result['analysis_time']}")


def display_report_section(result: Dict):
    st.header("Generate Report")

    col1, col2 = st.columns(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with col1:
        st.download_button(
            label="Download Text Report",
            data=generate_report(result),
            file_name=f"rib_fracture_report_{ts}.txt",
            mime="text/plain",
        )

    with col2:
        st.download_button(
            label="Download JSON Data",
            data=json.dumps(result, indent=2),
            file_name=f"rib_fracture_data_{ts}.json",
            mime="application/json",
        )


# ------------------------------------------------------------------
# Entry point — called from main app
# ------------------------------------------------------------------

def run():
    """Render the full rib fracture analysis page."""
    inject_styles()
    display_header()
    display_disclaimer()

    # Sidebar settings
    with st.sidebar:
        st.header("Rib Fracture Settings")
        model_path = st.text_input(
            "Custom YOLO model path",
            value="",
            help="Leave blank to use default YOLOv8n or demo mode",
        )
        if not YOLO_AVAILABLE:
            st.warning(
                "ultralytics package not installed. Running in **demo mode** "
                "with simulated detections."
            )

    # Session state init
    if "rf_result" not in st.session_state:
        st.session_state.rf_result = None
    if "rf_uploaded_image" not in st.session_state:
        st.session_state.rf_uploaded_image = None

    analyzer = RibFractureAnalyzer(model_path=model_path or None)

    image = display_upload_section()

    if image is not None:
        display_analysis_controls(analyzer, image)

    if st.session_state.rf_result is not None:
        display_image = st.session_state.rf_uploaded_image or image
        if display_image:
            display_results(st.session_state.rf_result, display_image, analyzer)
            display_report_section(st.session_state.rf_result)

        if st.button("New Analysis"):
            st.session_state.rf_result = None
            st.session_state.rf_uploaded_image = None
            st.rerun()
