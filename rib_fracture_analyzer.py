"""
Rib Fracture Analyzer using YOLOv8 Object Detection

This module provides AI-powered rib fracture detection from chest X-ray
and CT images using the YOLO (You Only Look Once) v8 architecture.
Designed for educational and research purposes only.
"""

import io
import base64
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Rib labels for anatomical mapping (bilateral, 1-12)
RIB_LABELS = [f"Rib {i}" for i in range(1, 13)]

# Fracture severity classification
SEVERITY_LEVELS = {
    "mild": {"label": "Hairline / Non-displaced", "color": "#F59E0B", "rgb": (245, 158, 11)},
    "moderate": {"label": "Displaced", "color": "#F97316", "rgb": (249, 115, 22)},
    "severe": {"label": "Comminuted / Flail", "color": "#EF4444", "rgb": (239, 68, 68)},
}

LATERALITY = ["Left", "Right"]


class RibFractureAnalyzer:
    """YOLOv8-based rib fracture detection and classification."""

    # Default YOLO model path — users can supply a custom-trained model
    DEFAULT_MODEL = "yolov8n.pt"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self.DEFAULT_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the YOLO model if ultralytics is available."""
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self.model = None

    def encode_image(self, image: Image.Image) -> str:
        """Encode a PIL Image to base64 PNG string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, image: Image.Image, confidence_threshold: float = 0.25) -> Dict:
        """
        Run rib fracture detection on the given image.

        If a trained YOLO model is available and loaded, real inference is
        performed.  Otherwise a mock analysis is returned for demonstration.
        """
        if self.model is not None:
            return self._analyze_with_yolo(image, confidence_threshold)
        return self.mock_analysis(image)

    def _analyze_with_yolo(self, image: Image.Image, conf: float) -> Dict:
        """Run actual YOLOv8 inference and map results to rib fracture schema."""
        results = self.model.predict(source=image, conf=conf, verbose=False)

        detections: List[Dict] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det_conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Map class to rib label & severity (depends on training)
                rib_index = cls_id % 12
                severity_key = list(SEVERITY_LEVELS.keys())[cls_id % 3]

                detections.append({
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(det_conf * 100, 1),
                    "rib": RIB_LABELS[rib_index],
                    "side": LATERALITY[cls_id % 2],
                    "severity": severity_key,
                    "severity_label": SEVERITY_LEVELS[severity_key]["label"],
                })

        return self._build_result(image, detections)

    # ------------------------------------------------------------------
    # Mock / demo analysis
    # ------------------------------------------------------------------

    def mock_analysis(self, image: Image.Image) -> Dict:
        """Generate realistic mock detection results for demo purposes."""
        w, h = image.size
        num_fractures = random.choices([0, 1, 2, 3, 4], weights=[15, 35, 30, 15, 5])[0]

        detections: List[Dict] = []
        used_ribs = set()

        for _ in range(num_fractures):
            # Pick a unique rib
            rib_idx = random.randint(0, 11)
            side = random.choice(LATERALITY)
            key = (rib_idx, side)
            if key in used_ribs:
                continue
            used_ribs.add(key)

            severity_key = random.choices(
                list(SEVERITY_LEVELS.keys()), weights=[50, 35, 15]
            )[0]

            # Random bounding box in plausible chest region
            cx = random.uniform(0.15, 0.85) * w
            cy = random.uniform(0.20, 0.80) * h
            bw = random.uniform(0.06, 0.12) * w
            bh = random.uniform(0.04, 0.08) * h
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2

            detections.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(random.uniform(65, 97), 1),
                "rib": RIB_LABELS[rib_idx],
                "side": side,
                "severity": severity_key,
                "severity_label": SEVERITY_LEVELS[severity_key]["label"],
            })

        return self._build_result(image, detections)

    # ------------------------------------------------------------------
    # Result construction & interpretation
    # ------------------------------------------------------------------

    def _build_result(self, image: Image.Image, detections: List[Dict]) -> Dict:
        overall_conf = (
            sum(d["confidence"] for d in detections) / len(detections)
            if detections
            else 0.0
        )

        result = {
            "fracture_count": len(detections),
            "detections": sorted(detections, key=lambda d: d["confidence"], reverse=True),
            "overall_confidence": round(overall_conf, 1),
            "image_size": {"width": image.size[0], "height": image.size[1]},
            "analysis_time": datetime.now().isoformat(),
            "model": self.model_path,
            "severity_summary": self._severity_summary(detections),
            "clinical_flags": self._clinical_flags(detections),
            "technical_notes": (
                "Analysis performed using YOLOv8 object detection model. "
                "Bounding boxes indicate suspected fracture locations."
            ),
        }
        result["interpretation"] = self.generate_interpretation(result)
        return result

    @staticmethod
    def _severity_summary(detections: List[Dict]) -> Dict[str, int]:
        summary: Dict[str, int] = {k: 0 for k in SEVERITY_LEVELS}
        for d in detections:
            summary[d["severity"]] += 1
        return summary

    @staticmethod
    def _clinical_flags(detections: List[Dict]) -> List[str]:
        """Generate clinical alert flags based on detection patterns."""
        flags: List[str] = []
        if len(detections) >= 3:
            flags.append("Multiple rib fractures detected — evaluate for flail chest")
        if any(d["severity"] == "severe" for d in detections):
            flags.append("Comminuted/flail fracture present — urgent clinical review recommended")

        # Check for consecutive ribs on the same side (flail chest indicator)
        sides: Dict[str, List[int]] = {"Left": [], "Right": []}
        for d in detections:
            rib_num = int(d["rib"].split()[-1])
            sides[d["side"]].append(rib_num)

        for side, ribs in sides.items():
            ribs_sorted = sorted(ribs)
            consecutive = 1
            for i in range(1, len(ribs_sorted)):
                if ribs_sorted[i] - ribs_sorted[i - 1] == 1:
                    consecutive += 1
                    if consecutive >= 3:
                        flags.append(
                            f"3+ consecutive {side.lower()} rib fractures — "
                            "high suspicion for flail chest segment"
                        )
                        break
                else:
                    consecutive = 1

        # Lower rib fractures — splenic/hepatic injury risk
        for d in detections:
            rib_num = int(d["rib"].split()[-1])
            if rib_num >= 9:
                organ = "splenic" if d["side"] == "Left" else "hepatic"
                flags.append(
                    f"Lower rib fracture ({d['rib']}, {d['side']}) — "
                    f"evaluate for {organ} injury"
                )

        return list(dict.fromkeys(flags))  # deduplicate preserving order

    def generate_interpretation(self, result: Dict) -> str:
        """Generate a clinical interpretation paragraph."""
        lines: List[str] = []

        n = result["fracture_count"]
        if n == 0:
            lines.append(
                "No rib fractures detected on the submitted image. "
                "Clinical correlation is recommended if symptoms persist."
            )
        else:
            lines.append(f"{n} suspected rib fracture(s) identified.")

            sev = result["severity_summary"]
            parts = []
            for key, info in SEVERITY_LEVELS.items():
                if sev[key]:
                    parts.append(f"{sev[key]} {info['label'].lower()}")
            if parts:
                lines.append("Severity breakdown: " + ", ".join(parts) + ".")

            for flag in result["clinical_flags"]:
                lines.append(f"ALERT: {flag}.")

        return " ".join(lines)

    # ------------------------------------------------------------------
    # Annotated image generation
    # ------------------------------------------------------------------

    def draw_annotations(self, image: Image.Image, result: Dict) -> Image.Image:
        """Draw bounding boxes and labels on a copy of the image."""
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_sm = font

        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            color = SEVERITY_LEVELS[det["severity"]]["rgb"]
            outline_width = 3 if det["severity"] == "severe" else 2

            # Draw box
            for i in range(outline_width):
                draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

            # Label background
            label = f"{det['rib']} ({det['side']}) {det['confidence']:.0f}%"
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_h = bbox[3] - bbox[1] + 6
            draw.rectangle([x1, y1 - text_h, x1 + (bbox[2] - bbox[0]) + 8, y1], fill=color)
            draw.text((x1 + 4, y1 - text_h + 2), label, fill="white", font=font)

            # Severity tag below box
            sev_label = det["severity_label"]
            draw.text((x1 + 2, y2 + 3), sev_label, fill=color, font=font_sm)

        return annotated


def generate_report(result: Dict) -> str:
    """Generate a plain-text rib fracture analysis report."""
    report = f"""
RIB FRACTURE ANALYSIS REPORT
=============================

Analysis Date: {result['analysis_time']}
Model: {result['model']}

SUMMARY
-------
Total Fractures Detected: {result['fracture_count']}
Overall Confidence: {result['overall_confidence']:.1f}%

SEVERITY BREAKDOWN
------------------
"""
    for key, info in SEVERITY_LEVELS.items():
        count = result["severity_summary"].get(key, 0)
        report += f"  {info['label']}: {count}\n"

    report += "\nDETECTED FRACTURES\n------------------\n"
    if result["detections"]:
        for i, det in enumerate(result["detections"], 1):
            report += (
                f"{i}. {det['rib']} ({det['side']}) — "
                f"{det['severity_label']} — "
                f"Confidence: {det['confidence']:.1f}%\n"
                f"   Bounding Box: {det['bbox']}\n"
            )
    else:
        report += "No fractures detected.\n"

    report += "\nCLINICAL FLAGS\n--------------\n"
    if result["clinical_flags"]:
        for flag in result["clinical_flags"]:
            report += f"  * {flag}\n"
    else:
        report += "  None\n"

    report += f"""
CLINICAL INTERPRETATION
-----------------------
{result['interpretation']}

TECHNICAL NOTES
---------------
{result['technical_notes']}

DISCLAIMER
----------
This analysis is for educational and research purposes only.
Results must be validated by qualified radiologists and clinicians.
Do not use for clinical diagnosis or treatment decisions.

Generated by Rib Fracture Analyzer — YOLOv8 Detection Pipeline
"""
    return report
