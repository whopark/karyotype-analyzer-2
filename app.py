import streamlit as st
import base64
from PIL import Image
import io
from datetime import datetime
import json
import re
from typing import Dict, Optional, List, Tuple
from collections import Counter
from enum import Enum
import numpy as np

# Computer Vision for chromosome detection
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Vision-Language Model API clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class APIProvider(Enum):
    OPENAI = "OpenAI GPT-4 Vision"
    ANTHROPIC = "Anthropic Claude Vision"
    GEMINI = "Google Gemini Vision"
    CONSENSUS = "Multi-Model Consensus"
    CV_VLM = "CV + VLM (Hybrid)"
    TWO_STAGE = "Two-Stage Pipeline (CV + VLM)"
    MOCK = "Demo Mode (No API)"


class ChromosomeDetector:
    """Computer Vision-based chromosome detection and counting"""

    def __init__(self):
        # Area thresholds will be scaled based on image size
        # Reduced min_area for better detection in high-res karyogram images
        self.base_min_area = 200  # Base minimum area (reduced from 500)
        self.base_max_area = 150000  # Base maximum area (increased)

    def detect_chromosomes(self, image: Image.Image) -> Dict:
        """
        Detect and count chromosomes using computer vision techniques.
        Uses multiple detection strategies and picks the best result.
        Returns detection results including count, bounding boxes, and group estimates.
        """
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not available", "count": 0}

        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            gray = img_array
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        img_height, img_width = gray.shape
        img_area = img_height * img_width

        # Scale area thresholds based on image size
        # For a karyogram with 46 chromosomes, each should be ~1-3% of total area
        scale_factor = img_area / (1000 * 1000)  # Normalize to 1MP image
        min_area = max(100, int(self.base_min_area * scale_factor))
        max_area = int(self.base_max_area * scale_factor)

        # Collect results from multiple detection methods
        all_results = []
        kernel = np.ones((3, 3), np.uint8)
        kernel_small = np.ones((2, 2), np.uint8)

        # Method 1: Otsu thresholding with morphology
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        result1 = self._detect_from_threshold(otsu_thresh, kernel, min_area, max_area, "otsu")
        all_results.append(result1)

        # Method 2: Adaptive thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )
        result2 = self._detect_from_threshold(adaptive_thresh, kernel, min_area, max_area, "adaptive")
        all_results.append(result2)

        # Method 3: Simple threshold at multiple levels
        for thresh_val in [100, 127, 150, 180]:
            _, simple_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            result_simple = self._detect_from_threshold(simple_thresh, kernel_small, min_area, max_area, f"simple_{thresh_val}")
            all_results.append(result_simple)

        # Method 4: Reduced area threshold (for small chromosomes)
        small_min_area = max(50, min_area // 3)
        result4 = self._detect_from_threshold(otsu_thresh, kernel, small_min_area, max_area, "otsu_small")
        all_results.append(result4)

        # Pick the best result (closest to 46, but accepting 45-48 as normal range)
        best_result = None
        best_diff = float('inf')

        for result in all_results:
            count = result["count"]
            # Prefer counts in the normal range (45-48)
            if 45 <= count <= 48:
                diff = abs(count - 46)
                if diff < best_diff:
                    best_diff = diff
                    best_result = result
            elif best_result is None or abs(count - 46) < best_diff:
                best_diff = abs(count - 46)
                best_result = result

        if best_result is None:
            best_result = all_results[0]

        # Extract best result
        chromosomes = best_result["chromosomes"]
        bounding_boxes = best_result["boxes"]
        areas = best_result["areas"]

        # Sort by area to help identify groups
        if areas:
            sorted_indices = np.argsort(areas)[::-1]
            sorted_areas = [areas[i] for i in sorted_indices]
            sorted_boxes = [bounding_boxes[i] for i in sorted_indices]
        else:
            sorted_areas = []
            sorted_boxes = []

        # Estimate Denver groups based on size distribution
        group_counts = self._estimate_denver_groups(sorted_areas, len(chromosomes))

        # Detect sex chromosomes region (typically bottom-right of karyogram)
        sex_chr_info = self._detect_sex_chromosome_region(image, sorted_boxes)

        return {
            "count": len(chromosomes),
            "bounding_boxes": sorted_boxes,
            "areas": sorted_areas,
            "group_counts": group_counts,
            "sex_chromosome_region": sex_chr_info,
            "detection_method": best_result["method"]
        }

    def _detect_from_threshold(self, thresh_img, kernel, min_area, max_area, method_name) -> Dict:
        """Apply morphology and extract chromosome contours from thresholded image"""
        # Apply morphological operations
        processed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        chromosomes = []
        bounding_boxes = []
        areas = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0

                # Chromosomes have reasonable aspect ratios (not too extreme)
                if 0.15 < aspect_ratio < 20:
                    chromosomes.append(contour)
                    bounding_boxes.append((x, y, w, h))
                    areas.append(area)

        return {
            "count": len(chromosomes),
            "chromosomes": chromosomes,
            "boxes": bounding_boxes,
            "areas": areas,
            "method": method_name
        }

    def _estimate_denver_groups(self, areas: List[float], total_count: int) -> Dict:
        """Estimate chromosome groups based on size distribution"""
        if not areas or total_count == 0:
            return {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0}

        # Normalize areas
        max_area = max(areas) if areas else 1
        normalized = [a / max_area for a in areas]

        # Estimate groups based on size percentiles
        # Group A (1-3): Largest, ~13% of chromosomes
        # Group B (4-5): Large, ~9%
        # Group C (6-12+X): Medium-large, ~30%
        # Group D (13-15): Medium, ~13%
        # Group E (16-18): Medium-small, ~13%
        # Group F (19-20): Small, ~9%
        # Group G (21-22+Y): Smallest, ~13%

        groups = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0}

        for norm_area in normalized:
            if norm_area > 0.85:
                groups["A"] += 1
            elif norm_area > 0.70:
                groups["B"] += 1
            elif norm_area > 0.50:
                groups["C"] += 1
            elif norm_area > 0.35:
                groups["D"] += 1
            elif norm_area > 0.25:
                groups["E"] += 1
            elif norm_area > 0.15:
                groups["F"] += 1
            else:
                groups["G"] += 1

        return groups

    def _detect_sex_chromosome_region(self, image: Image.Image, boxes: List[Tuple]) -> Dict:
        """Analyze the sex chromosome region (typically last row of karyogram)"""
        if not boxes:
            return {"x_count": 0, "y_count": 0, "estimated": "unknown"}

        # In a standard karyogram, sex chromosomes are at the bottom-right
        # Find chromosomes in the lower portion of the image
        img_height = image.height
        img_width = image.width

        # Look at bottom 20% of image
        bottom_region_y = img_height * 0.8
        right_region_x = img_width * 0.6

        sex_chr_candidates = []
        for box in boxes:
            x, y, w, h = box
            center_y = y + h/2
            center_x = x + w/2
            if center_y > bottom_region_y and center_x > right_region_x:
                sex_chr_candidates.append(box)

        # Count and analyze sizes
        num_candidates = len(sex_chr_candidates)

        # Estimate based on count and relative sizes
        if num_candidates == 0:
            return {"x_count": 0, "y_count": 0, "estimated": "unknown", "region_count": 0}

        # Get areas of sex chromosome candidates
        candidate_areas = [w * h for (x, y, w, h) in sex_chr_candidates]

        if num_candidates >= 2:
            # Check if one is notably smaller (Y chromosome)
            min_area = min(candidate_areas)
            max_area = max(candidate_areas)

            if min_area < max_area * 0.6:  # Y is typically smaller
                # Likely XY configuration
                return {"x_count": 1, "y_count": 1, "estimated": "XY", "region_count": num_candidates}
            else:
                # Similar sizes - likely XX
                return {"x_count": 2, "y_count": 0, "estimated": "XX", "region_count": num_candidates}
        elif num_candidates == 1:
            # Single sex chromosome - likely Turner syndrome (45,X)
            return {"x_count": 1, "y_count": 0, "estimated": "X", "region_count": 1}

        return {"x_count": num_candidates, "y_count": 0, "estimated": "unknown", "region_count": num_candidates}

    def detect_karyogram_positions(self, image: Image.Image) -> Dict:
        """
        Detect chromosome positions in an ARRANGED karyogram image.
        Uses grid-based detection to count chromosomes at specific positions.

        Standard karyogram layout:
        Row 1: Groups A,B (1-5)
        Row 2: Group C (6-12, X)
        Row 3: Groups D,E (13-18)
        Row 4: Groups F,G (19-22, Y)
        """
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not available"}

        # First, detect all chromosomes
        detection = self.detect_chromosomes(image)
        boxes = detection.get("bounding_boxes", [])
        total_count = detection.get("count", 0)

        if total_count == 0:
            return {"error": "No chromosomes detected", "total": 0}

        img_width = image.width
        img_height = image.height

        # Define grid regions for standard karyogram layout
        # Row 4 (bottom 25%) contains Groups F and G (positions 19-22, Y)
        row4_y_start = img_height * 0.75

        # Within row 4, Group G (21, 22, Y) is typically in the right portion
        # Group G spans roughly the last 30% of the width
        group_g_x_start = img_width * 0.55

        # Position 21 is typically first in Group G (left side of Group G region)
        # Position 22 is middle, Y (or X for females) is at the end
        pos21_x_start = img_width * 0.55
        pos21_x_end = img_width * 0.70
        pos22_x_start = img_width * 0.70
        pos22_x_end = img_width * 0.82
        sex_x_start = img_width * 0.82

        # Count chromosomes in each region
        pos21_count = 0
        pos22_count = 0
        sex_chr_count = 0
        group_g_boxes = []

        for box in boxes:
            x, y, w, h = box
            center_x = x + w/2
            center_y = y + h/2

            # Check if in row 4 (Group G region)
            if center_y > row4_y_start:
                if pos21_x_start < center_x < pos21_x_end:
                    pos21_count += 1
                    group_g_boxes.append(("21", box))
                elif pos22_x_start < center_x < pos22_x_end:
                    pos22_count += 1
                    group_g_boxes.append(("22", box))
                elif center_x >= sex_x_start:
                    sex_chr_count += 1
                    group_g_boxes.append(("sex", box))

        # Analyze sex chromosomes based on size
        sex_region = detection.get("sex_chromosome_region", {})

        # Build position counts
        position_counts = {
            "position_21": pos21_count,
            "position_22": pos22_count,
            "sex_chromosomes": sex_chr_count,
            "sex_chr_estimated": sex_region.get("estimated", "unknown")
        }

        # Determine karyotype based on counts
        karyotype_analysis = self._analyze_karyotype_from_counts(
            total_count,
            pos21_count,
            sex_region
        )

        return {
            "total_count": total_count,
            "position_counts": position_counts,
            "group_g_details": group_g_boxes,
            "karyotype_analysis": karyotype_analysis,
            "detection_method": "karyogram_grid_analysis"
        }

    def _analyze_karyotype_from_counts(self, total: int, pos21_count: int, sex_info: Dict) -> Dict:
        """Analyze karyotype based on CV counts"""
        analysis = {
            "total_chromosomes": total,
            "position_21_count": pos21_count,
            "sex_chromosomes": sex_info.get("estimated", "unknown"),
            "preliminary_diagnosis": "Unknown",
            "confidence": "low"
        }

        # Basic decision logic based on CV counts
        if total == 46:
            if pos21_count == 2:
                if sex_info.get("estimated") == "XY":
                    analysis["preliminary_diagnosis"] = "46,XY (Normal male)"
                elif sex_info.get("estimated") == "XX":
                    analysis["preliminary_diagnosis"] = "46,XX (Normal female)"
                else:
                    analysis["preliminary_diagnosis"] = "46,?? (Normal count, sex uncertain)"
                analysis["confidence"] = "medium"
            else:
                analysis["preliminary_diagnosis"] = f"46 total but pos21={pos21_count} (needs review)"
                analysis["confidence"] = "low"

        elif total == 47:
            if pos21_count >= 3:
                # Likely Down syndrome
                sex = sex_info.get("estimated", "??")
                analysis["preliminary_diagnosis"] = f"47,{sex},+21 (Likely Down syndrome)"
                analysis["confidence"] = "medium"
            elif pos21_count == 2:
                # Sex chromosome abnormality likely
                sex = sex_info.get("estimated", "unknown")
                if sex in ["XXY", "XXX", "XYY"]:
                    analysis["preliminary_diagnosis"] = f"47,{sex} (Sex chromosome abnormality)"
                else:
                    analysis["preliminary_diagnosis"] = "47,?? (Trisomy, location uncertain)"
                analysis["confidence"] = "medium"
            else:
                analysis["preliminary_diagnosis"] = "47 total (abnormality location uncertain)"
                analysis["confidence"] = "low"

        elif total == 45:
            analysis["preliminary_diagnosis"] = "45,? (Possible monosomy)"
            analysis["confidence"] = "low"
        else:
            analysis["preliminary_diagnosis"] = f"{total} chromosomes (unusual count)"
            analysis["confidence"] = "low"

        return analysis

    def create_annotated_image(self, image: Image.Image, detection_result: Dict) -> Image.Image:
        """Create an annotated image with detected chromosomes highlighted"""
        if not CV2_AVAILABLE:
            return image

        img_array = np.array(image.convert('RGB'))

        # Draw bounding boxes
        for i, (x, y, w, h) in enumerate(detection_result.get("bounding_boxes", [])):
            color = (0, 255, 0)  # Green
            cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_array, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Add count text
        count = detection_result.get("count", 0)
        cv2.putText(img_array, f"Detected: {count} chromosomes", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return Image.fromarray(img_array)


# 페이지 설정
st.set_page_config(
    page_title="Chromosome Karyotype Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        background-color: #3B82F6;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .disclaimer {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #10B981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF4444;
        font-weight: bold;
    }
    .notation-display {
        background-color: #1F2937;
        color: #F9FAFB;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .api-available {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .api-unavailable {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .consensus-agree {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .consensus-partial {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .consensus-disagree {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .voting-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .voting-table th, .voting-table td {
        border: 1px solid #E5E7EB;
        padding: 0.75rem;
        text-align: center;
    }
    .voting-table th {
        background-color: #F3F4F6;
    }
    .vote-winner {
        background-color: #D1FAE5;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'raw_response' not in st.session_state:
    st.session_state.raw_response = None
if 'consensus_api_keys' not in st.session_state:
    st.session_state.consensus_api_keys = {
        'openai': None,
        'anthropic': None,
        'gemini': None
    }
if 'consensus_settings' not in st.session_state:
    st.session_state.consensus_settings = {
        'use_openai': True,
        'use_anthropic': True,
        'use_gemini': True,
        'min_agreement': 'majority'
    }
if 'cv_detection' not in st.session_state:
    st.session_state.cv_detection = None


# Karyotype analysis system prompt - Enhanced with Chain-of-Thought and Verification
KARYOTYPE_ANALYSIS_PROMPT = """You are an expert clinical cytogeneticist analyzing a KARYOGRAM image.

## WHAT IS A KARYOGRAM?

A karyogram is an ARRANGED display of chromosomes where:
- Chromosomes are organized by NUMBER (1-22, X, Y)
- Each position has a LABEL showing which chromosome it is
- Most positions show a PAIR (2 chromosomes side by side)
- Abnormalities show 1 (monosomy) or 3 (trisomy) at a position

## STEP 1: READ THE LABELS IN THE IMAGE

Look at the image and find the numeric labels:
- Labels "1" through "22" mark the autosome positions
- Labels "X" and "Y" mark the sex chromosome positions
- These labels are typically printed below or near each chromosome group

## STEP 2: COUNT CHROMOSOMES AT EACH LABELED POSITION

For each labeled position, count the chromosome objects:

**Autosomes (positions 1-22):**
- Normal: 2 chromosomes at each position
- Trisomy: 3 chromosomes at one position (e.g., 3 at position "21" = Down syndrome)

**Sex chromosomes:**
- Normal female: 2 chromosomes at position X, 0 at Y → XX
- Normal male: 1 chromosome at position X, 1 at Y → XY
- Klinefelter: 2 at X, 1 at Y → XXY
- Triple X: 3 at X, 0 at Y → XXX

## STEP 3: DETERMINE THE KARYOTYPE

**Look specifically at position "21" in the image:**
- Count how many chromosome objects are shown under/near the "21" label
- If you see THREE chromosomes at position 21 → This is TRISOMY 21 (Down Syndrome)
- If you see TWO chromosomes at position 21 → Normal for position 21

**Look at the sex chromosome positions (X and Y):**
- Count chromosomes at position X
- Count chromosomes at position Y (or check if Y position is empty)

## STEP 4: CALCULATE TOTAL

Add up all chromosomes:
- Positions 1-22: Count at each position (normally 2 each = 44)
- Position X: Count (1 or 2 or 3)
- Position Y: Count (0 or 1 or 2)
- TOTAL should be 45, 46, or 47

## COMMON KARYOTYPES

| Karyotype | Position 21 | X count | Y count | Total |
|-----------|-------------|---------|---------|-------|
| 46,XY (Normal male) | 2 | 1 | 1 | 46 |
| 46,XX (Normal female) | 2 | 2 | 0 | 46 |
| 47,XY,+21 (Down syndrome male) | **3** | 1 | 1 | 47 |
| 47,XX,+21 (Down syndrome female) | **3** | 2 | 0 | 47 |
| 47,XXY (Klinefelter) | 2 | 2 | 1 | 47 |
| 47,XXX (Triple X) | 2 | 3 | 0 | 47 |

## KEY DISTINCTION: DOWN SYNDROME vs KLINEFELTER

Both have 47 chromosomes, but:
- **Down Syndrome**: Position 21 shows THREE small chromosomes, sex chromosomes are normal (XX or XY)
- **Klinefelter**: Position 21 shows TWO chromosomes (normal), sex chromosomes show XXY

**THE CRITICAL QUESTION:** Does position 21 have 2 or 3 chromosomes?
- If 3 → Down syndrome
- If 2 → Check sex chromosomes for XXY, XXX, etc.

## OUTPUT FORMAT

Return ONLY a valid JSON object:
{
    "notation": "ISCN notation (e.g., 46,XY or 47,XX,+21)",
    "chromosome_count": number,
    "sex_chromosomes": "XX/XY/XXY/XXX/X",
    "chromosome_21_count": number (2 or 3),
    "position_counts": {
        "autosomes_1_to_20": "2 each (normal) or specify abnormalities",
        "position_21": number,
        "position_22": number,
        "position_X": number,
        "position_Y": number
    },
    "abnormalities": [
        {"type": "type", "chromosome": "affected", "description": "description"}
    ],
    "confidence": number (0-100),
    "interpretation": "clinical interpretation",
    "detailed_findings": "I see [N] chromosomes at position 21, [N] at X, [N] at Y"
}

## BEFORE ANSWERING, VERIFY:

1. What number label do you see at position 21? How many chromosomes are there?
2. What do you see at the X and Y positions?
3. Does your total match your individual position counts?

If position 21 shows 3 chromosomes, report Down syndrome (47,XX,+21 or 47,XY,+21).
If position 21 shows 2 chromosomes but total is 47, check sex chromosomes (XXY = Klinefelter, XXX = Triple X)."""


# CV+VLM Interpretation Prompt - VLM interprets CV counts (no visual counting needed)
CV_VLM_INTERPRETATION_PROMPT = """You are a clinical cytogeneticist interpreting computer vision analysis results.

## YOUR TASK
A computer vision system has analyzed a karyogram image and provided chromosome counts.
Your job is to INTERPRET these counts and provide a clinical diagnosis.

DO NOT try to count chromosomes yourself - trust the CV system's counts.

## CV ANALYSIS RESULTS
{cv_results}

## INTERPRETATION RULES

Based on the CV counts above, determine the karyotype:

**Total = 46 with Position 21 = 2:**
- If sex chromosomes = XY → 46,XY (Normal male)
- If sex chromosomes = XX → 46,XX (Normal female)

**Total = 47 with Position 21 = 3:**
- If sex chromosomes = XY → 47,XY,+21 (Down syndrome, male)
- If sex chromosomes = XX → 47,XX,+21 (Down syndrome, female)

**Total = 47 with Position 21 = 2:**
- If sex chromosomes = XXY → 47,XXY (Klinefelter syndrome)
- If sex chromosomes = XXX → 47,XXX (Triple X syndrome)
- If sex chromosomes = XYY → 47,XYY (Jacob syndrome)

**Total = 45:**
- If sex chromosomes = X only → 45,X (Turner syndrome)

## OUTPUT FORMAT

Return ONLY a valid JSON object:
{{
    "notation": "ISCN notation based on CV counts",
    "chromosome_count": {total_count},
    "sex_chromosomes": "XX/XY/XXY/XXX/X based on CV data",
    "chromosome_21_count": {pos21_count},
    "abnormalities": [
        {{"type": "type", "chromosome": "affected", "description": "description"}}
    ],
    "confidence": number (0-100),
    "interpretation": "clinical interpretation",
    "cv_analysis_summary": "Summary of what CV detected",
    "analysis_method": "CV+VLM"
}}

Provide your interpretation based ONLY on the CV counts provided above."""


class KaryotypeAnalyzer:
    """염색체 핵형 분석기 클래스 - 다중 API 제공자 지원"""

    def __init__(self, provider: APIProvider, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key

    def encode_image_base64(self, image: Image.Image) -> str:
        """이미지를 base64로 인코딩"""
        buffered = io.BytesIO()
        # Convert to RGB if necessary (for RGBA, P, or grayscale images)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()

    def analyze(self, image: Image.Image, consensus_keys: Optional[Dict] = None) -> Dict:
        """선택된 API 제공자를 사용하여 분석 수행"""
        if self.provider == APIProvider.OPENAI:
            return self._analyze_with_openai(image)
        elif self.provider == APIProvider.ANTHROPIC:
            return self._analyze_with_anthropic(image)
        elif self.provider == APIProvider.GEMINI:
            return self._analyze_with_gemini(image)
        elif self.provider == APIProvider.CONSENSUS:
            return self._analyze_with_consensus(image, consensus_keys or {})
        elif self.provider == APIProvider.TWO_STAGE:
            return self._analyze_with_two_stage(image)
        elif self.provider == APIProvider.CV_VLM:
            return self._analyze_with_cv_vlm(image)
        else:
            return self._mock_analysis()

    def _analyze_with_openai(self, image: Image.Image) -> Dict:
        """OpenAI GPT-4 Vision API를 사용한 분석"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        client = OpenAI(api_key=self.api_key)
        image_base64 = self.encode_image_base64(image)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a board-certified clinical cytogeneticist. CRITICAL: You must COUNT chromosomes accurately before diagnosis. A normal karyotype has EXACTLY 46 chromosomes. Do NOT assume abnormalities - verify by counting each Denver group. Only report an abnormality if you can identify the SPECIFIC extra or missing chromosome. Always provide results as valid JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": KARYOTYPE_ANALYSIS_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2500,
            temperature=0.1
        )

        raw_content = response.choices[0].message.content
        st.session_state.raw_response = raw_content
        return self._parse_response(raw_content, "OpenAI GPT-4 Vision")

    def _analyze_with_anthropic(self, image: Image.Image) -> Dict:
        """Anthropic Claude Vision API를 사용한 분석"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        client = anthropic.Anthropic(api_key=self.api_key)
        image_base64 = self.encode_image_base64(image)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            system="You are a board-certified clinical cytogeneticist. CRITICAL: You must COUNT chromosomes accurately before diagnosis. A normal karyotype has EXACTLY 46 chromosomes. Do NOT assume abnormalities - verify by counting each Denver group. Only report an abnormality if you can identify the SPECIFIC extra or missing chromosome. Always provide results as valid JSON.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": KARYOTYPE_ANALYSIS_PROMPT
                        }
                    ]
                }
            ]
        )

        raw_content = response.content[0].text
        st.session_state.raw_response = raw_content
        return self._parse_response(raw_content, "Anthropic Claude Vision")

    def _analyze_with_gemini(self, image: Image.Image) -> Dict:
        """Google Gemini Vision API를 사용한 분석"""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI package not installed. Run: pip install google-genai")
        if not self.api_key:
            raise ValueError("Google API key is required")

        client = genai.Client(api_key=self.api_key)

        # Convert image for Gemini
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')

        # Prepend system instruction to the prompt
        system_instruction = "CRITICAL: You must COUNT chromosomes accurately before diagnosis. A normal karyotype has EXACTLY 46 chromosomes. Do NOT assume abnormalities - verify by counting each Denver group. Only report an abnormality if you can identify the SPECIFIC extra or missing chromosome.\n\n"

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[system_instruction + KARYOTYPE_ANALYSIS_PROMPT, image],
            config={
                'temperature': 0.1,
                'max_output_tokens': 2500
            }
        )

        raw_content = response.text
        st.session_state.raw_response = raw_content
        return self._parse_response(raw_content, "Google Gemini Vision")

    def _analyze_with_two_stage(self, image: Image.Image) -> Dict:
        """Two-stage analysis: CV detection + VLM classification"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not installed. Run: pip install opencv-python")
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Stage 2 VLM analysis")

        # Stage 1: Computer Vision Detection
        detector = ChromosomeDetector()
        detection_result = detector.detect_chromosomes(image)

        cv_count = detection_result.get("count", 0)
        cv_groups = detection_result.get("group_counts", {})
        sex_chr_info = detection_result.get("sex_chromosome_region", {})

        # Store detection result for display
        st.session_state.cv_detection = detection_result

        # Stage 2: VLM Classification with CV-informed prompt
        two_stage_prompt = f"""You are an expert clinical cytogeneticist. A computer vision system has pre-analyzed this karyotype image.

## COMPUTER VISION DETECTION RESULTS (Stage 1)

**Detected Chromosome Count: {cv_count}**

Estimated Denver Group Distribution:
- Group A (1-3): {cv_groups.get('A', 0)} chromosomes
- Group B (4-5): {cv_groups.get('B', 0)} chromosomes
- Group C (6-12+X): {cv_groups.get('C', 0)} chromosomes
- Group D (13-15): {cv_groups.get('D', 0)} chromosomes
- Group E (16-18): {cv_groups.get('E', 0)} chromosomes
- Group F (19-20): {cv_groups.get('F', 0)} chromosomes
- Group G (21-22+Y): {cv_groups.get('G', 0)} chromosomes

Sex Chromosome Region Analysis:
- Estimated configuration: {sex_chr_info.get('estimated', 'unknown')}
- Chromosomes in region: {sex_chr_info.get('region_count', 0)}

## YOUR TASK (Stage 2)

Based on the CV detection count of **{cv_count} chromosomes**, determine the karyotype:

**CRITICAL DECISION RULES:**

1. **If CV count = 46**: This is likely a NORMAL karyotype
   - Check sex chromosomes: XX (female) or XY (male)
   - Report as 46,XX or 46,XY

2. **If CV count = 45**: This suggests MONOSOMY (missing chromosome)
   - Most common: Turner syndrome (45,X) - missing one sex chromosome
   - Look for single X chromosome in sex region

3. **If CV count = 47**: This suggests TRISOMY (extra chromosome)
   - Check WHERE the extra chromosome is:
   - If extra in Group G (small): Likely Trisomy 21 (Down syndrome) → 47,XX,+21 or 47,XY,+21
   - If extra in Group C (medium X): Could be Triple X (47,XXX) or Klinefelter (47,XXY)
   - Distinguish by checking Y chromosome presence:
     * NO Y chromosome → 47,XXX (Triple X, female)
     * Y chromosome present → 47,XXY (Klinefelter, male)

4. **If CV count differs significantly from expected**: Trust the CV count over visual estimation.

## VISUAL VERIFICATION

Look at the image to verify:
1. The sex chromosome region (bottom-right of karyogram)
2. Whether Group G has 4 (normal), 5 (trisomy 21), or other count
3. The presence/absence of Y chromosome (small, distinct from 21/22)

## OUTPUT FORMAT

Return ONLY valid JSON:
{{
    "notation": "ISCN notation based on CV count of {cv_count}",
    "chromosome_count": {cv_count},
    "sex_chromosomes": "XX/XY/X/XXY/XXX",
    "cv_detection": {{
        "count": {cv_count},
        "groups": {json.dumps(cv_groups)},
        "sex_region": "{sex_chr_info.get('estimated', 'unknown')}"
    }},
    "abnormalities": [
        {{"type": "type", "chromosome": "chr", "description": "desc"}}
    ],
    "confidence": number (0-100),
    "interpretation": "clinical interpretation",
    "detailed_findings": "how CV detection informed the diagnosis"
}}

IMPORTANT: The CV system detected {cv_count} chromosomes. Use this as your primary count reference."""

        # Call OpenAI with CV-informed prompt
        client = OpenAI(api_key=self.api_key)
        image_base64 = self.encode_image_base64(image)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a cytogeneticist. A CV system detected {cv_count} chromosomes. Use this count as ground truth and determine the karyotype classification. Do not re-count - trust the CV detection."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": two_stage_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2500,
            temperature=0.1
        )

        raw_content = response.choices[0].message.content
        st.session_state.raw_response = raw_content

        result = self._parse_response(raw_content, "Two-Stage Pipeline (CV + VLM)")
        result['cv_detection'] = detection_result
        result['pipeline'] = 'two_stage'
        return result

    def _parse_response(self, raw_content: str, provider_name: str) -> Dict:
        """API 응답에서 JSON 파싱"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            # Ensure required fields exist
            result.setdefault('notation', 'Unable to determine')
            result.setdefault('chromosome_count', 0)
            result.setdefault('sex_chromosomes', 'Unknown')
            result.setdefault('abnormalities', [])
            result.setdefault('confidence', 0)
            result.setdefault('interpretation', 'Analysis incomplete')
            result.setdefault('detailed_findings', '')

            # Add metadata
            result['analysis_time'] = datetime.now().isoformat()
            result['technical_notes'] = f"Analysis performed using {provider_name}"
            result['provider'] = provider_name

            return result

        except (json.JSONDecodeError, ValueError) as e:
            # Return a structured error response
            return {
                'notation': 'Parse Error',
                'chromosome_count': 0,
                'sex_chromosomes': 'Unknown',
                'abnormalities': [],
                'confidence': 0,
                'interpretation': f'Failed to parse API response: {str(e)}',
                'detailed_findings': f'Raw response preview: {raw_content[:500]}...',
                'analysis_time': datetime.now().isoformat(),
                'technical_notes': f'Error parsing response from {provider_name}',
                'provider': provider_name
            }

    def _analyze_with_cv_vlm(self, image: Image.Image) -> Dict:
        """CV counts chromosomes, VLM interprets the counts. Falls back to VLM-only if CV unreliable."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not available. Run: pip install opencv-python")

        # Stage 1: CV Detection and Counting
        detector = ChromosomeDetector()
        cv_result = detector.detect_karyogram_positions(image)

        # Store CV detection for display
        st.session_state.cv_detection = cv_result

        # Extract key counts
        total_count = cv_result.get("total_count", 0)
        position_counts = cv_result.get("position_counts", {})
        pos21_count = position_counts.get("position_21", 0)
        sex_chr_count = position_counts.get("sex_chromosomes", 0)  # Actual count in sex region
        sex_chr_estimated = position_counts.get("sex_chr_estimated", "unknown")
        karyotype_analysis = cv_result.get("karyotype_analysis", {})

        # FALLBACK: If CV count is unreliable (outside 44-50 range), use VLM with CV hints
        CV_MIN_RELIABLE = 44
        CV_MAX_RELIABLE = 50

        if total_count < CV_MIN_RELIABLE or total_count > CV_MAX_RELIABLE:
            # CV total count is unreliable, but position data may still be useful
            st.warning(f"⚠️ CV detected {total_count} chromosomes (outside 44-50 range). Using VLM with CV position hints.")

            # Use VLM analysis WITH CV position hints (not pure VLM-only)
            result = self._analyze_with_cv_hints(image, pos21_count, sex_chr_estimated, sex_chr_count)

            # Add fallback note
            result["fallback_used"] = True
            result["cv_detection"] = {
                "total_detected": total_count,
                "position_21_count": pos21_count,
                "cv_unreliable": True,
                "fallback_reason": f"CV count ({total_count}) outside reliable range (44-50)"
            }
            result["technical_notes"] = f"CV total ({total_count}) unreliable. VLM analyzed with CV position hints (pos21={pos21_count})."
            result["analysis_method"] = "CV+VLM (CV-assisted fallback)"

            return result

        # Format CV results for VLM
        cv_results_text = f"""
COMPUTER VISION ANALYSIS RESULTS:
- Total chromosomes detected: {total_count}
- Position 21 count: {pos21_count}
- Position 22 count: {position_counts.get("position_22", 0)}
- Sex chromosome region count: {position_counts.get("sex_chromosomes", 0)}
- Estimated sex chromosomes: {sex_chr_estimated}
- CV preliminary diagnosis: {karyotype_analysis.get("preliminary_diagnosis", "Unknown")}
- CV confidence: {karyotype_analysis.get("confidence", "low")}
"""

        # Stage 2: VLM Interpretation
        client = OpenAI(api_key=self.api_key)

        # Fill in the prompt template
        interpretation_prompt = CV_VLM_INTERPRETATION_PROMPT.format(
            cv_results=cv_results_text,
            total_count=total_count,
            pos21_count=pos21_count
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical cytogeneticist. Interpret the computer vision chromosome counts provided. Do NOT try to count visually - trust the CV data and provide clinical interpretation based on the counts."
                },
                {
                    "role": "user",
                    "content": interpretation_prompt
                }
            ],
            max_tokens=1500,
            temperature=0.1
        )

        raw_content = response.choices[0].message.content
        st.session_state.raw_response = f"CV Results:\n{cv_results_text}\n\nVLM Interpretation:\n{raw_content}"

        # Parse the VLM response
        result = self._parse_response(raw_content, "CV + VLM (Hybrid)")

        # LOW-CONFIDENCE FALLBACK: If VLM confidence < 50%, re-run with CV hints (visual analysis)
        LOW_CONFIDENCE_THRESHOLD = 50
        vlm_confidence = result.get("confidence", 0)

        if vlm_confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ Initial analysis confidence low ({vlm_confidence}%). Re-analyzing with visual verification...")

            # Re-run with CV hints - this uses visual analysis with position hints
            result = self._analyze_with_cv_hints(image, pos21_count, sex_chr_estimated)

            # Add fallback note
            result["fallback_used"] = True
            result["cv_detection"] = {
                "total_detected": total_count,
                "position_21_count": pos21_count,
                "cv_unreliable": False,
                "fallback_reason": f"Initial confidence ({vlm_confidence}%) below threshold ({LOW_CONFIDENCE_THRESHOLD}%)"
            }
            result["technical_notes"] = f"CV detected {total_count} chromosomes. Low confidence ({vlm_confidence}%) triggered visual re-analysis with CV hints."
            result["analysis_method"] = "CV+VLM (low-confidence fallback)"

            return result

        # Add CV-specific data to result
        result["cv_detection"] = {
            "total_detected": total_count,
            "position_21_count": pos21_count,
            "sex_chr_estimated": sex_chr_estimated,
            "cv_preliminary": karyotype_analysis.get("preliminary_diagnosis", "Unknown")
        }
        result["analysis_method"] = "CV+VLM"
        result["technical_notes"] = f"CV detected {total_count} chromosomes, {pos21_count} at position 21. VLM interpreted the counts."

        return result

    def _analyze_with_cv_hints(self, image: Image.Image, cv_pos21_count: int, cv_sex_estimate: str, cv_sex_region_count: int = 0) -> Dict:
        """VLM analysis with CV position hints when total count is unreliable"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Prepare image
        buffered = io.BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Detect potential Triple X pattern: CV may misclassify 3 X chromosomes as position 21
        # If pos21=3 but sex region count is low, this could indicate Triple X, not Down syndrome
        triple_x_warning = ""
        if cv_pos21_count >= 3 and cv_sex_region_count <= 1:
            triple_x_warning = """
⚠️ **CRITICAL WARNING - POTENTIAL TRIPLE X MISCLASSIFICATION**:
CV detected 3 chromosomes near position 21 but few in the sex region. This pattern often occurs
when THREE X CHROMOSOMES are misdetected as position 21 due to image layout.

VISUALLY VERIFY: Are the 3 chromosomes at "position 21" actually:
- THREE SMALL acrocentric chromosomes (true Down syndrome - trisomy 21), OR
- THREE MEDIUM-SIZED submetacentric chromosomes (Triple X syndrome - 47,XXX)?

If they are MEDIUM-sized and similar to each other → This is likely 47,XXX (Triple X), NOT Down syndrome!
"""

        # Create prompt with CV hints
        cv_hint_prompt = f"""You are an expert clinical cytogeneticist analyzing a karyotype image.

## CV DETECTION HINTS (use as guidance, VERIFY VISUALLY - CV may be incorrect)
- CV detected approximately {cv_pos21_count} chromosome(s) near position 21 region
- CV detected {cv_sex_region_count} chromosome(s) in the sex chromosome region
- CV estimated sex chromosomes: {cv_sex_estimate}
- Note: CV grid detection may misclassify chromosomes due to image layout variations
{triple_x_warning}

## ANALYSIS ORDER (FOLLOW THIS SEQUENCE)

### STEP 1: COUNT SEX CHROMOSOMES FIRST (Most Important!)
Look at the sex chromosome region (typically bottom-right, after position 22):
- X chromosome: MEDIUM-SIZED (~6-7% of haploid genome), submetacentric, similar to chromosomes 6-12
- Y chromosome: SMALL (~2% of genome), acrocentric, much smaller than X

**Count how many X chromosomes you see:**
- 1 X chromosome = possible Turner (45,X) or male (XY)
- 2 X chromosomes = normal female (XX) or Klinefelter if Y present (XXY)
- 3 X chromosomes = TRIPLE X SYNDROME (47,XXX) - THREE medium-sized X's, NO Y

**TRIPLE X KEY IDENTIFIER**: Look for THREE similar-sized MEDIUM chromosomes grouped together in the sex region. They will all look alike (all X's are submetacentric and similar size).

### STEP 2: CHECK FOR Y CHROMOSOME
- If Y present with 2 X's → Klinefelter (47,XXY)
- If NO Y with 3 X's → Triple X (47,XXX)
- If NO Y with 1 X → Turner (45,X)

### STEP 3: COUNT POSITION 21 (Group G)
Position 21 contains SMALL acrocentric chromosomes (much smaller than X):
- 2 at position 21 = Normal
- 3 at position 21 = Down syndrome (trisomy 21)

## CRITICAL DISTINCTION: Triple X vs Down Syndrome

| Feature | Triple X (47,XXX) | Down Syndrome (47,+21) |
|---------|-------------------|------------------------|
| Extra chromosome location | SEX REGION | POSITION 21 |
| Extra chromosome size | MEDIUM (X-sized) | SMALL (Group G) |
| Extra chromosome shape | Submetacentric | Acrocentric |
| Sex chromosomes | XXX (3 medium) | XX or XY (2 normal) |
| Position 21 count | 2 (normal) | 3 (trisomy) |

**If you see 47 chromosomes with 3 MEDIUM chromosomes in sex region → 47,XXX (Triple X)**
**If you see 47 chromosomes with 3 SMALL chromosomes at position 21 → 47,+21 (Down)**

## SYNDROME DETECTION CHECKLIST
1. Total = 45, sex region has only 1 X → Turner syndrome (45,X)
2. Total = 47, sex region has XXY → Klinefelter syndrome (47,XXY)
3. Total = 47, sex region has XXX (3 medium, no Y) → Triple X syndrome (47,XXX)
4. Total = 47, position 21 has 3 small acrocentric → Down syndrome (47,XX,+21 or 47,XY,+21)

## OUTPUT FORMAT
Return ONLY valid JSON:
{{
    "notation": "ISCN notation",
    "chromosome_count": number,
    "sex_chromosomes": "XX/XY/X/XXY/XXX/XYY",
    "abnormalities": [
        {{"type": "trisomy/monosomy/etc", "chromosome": "21/X/etc", "description": "details"}}
    ],
    "confidence": number (0-100),
    "interpretation": "clinical interpretation with syndrome name",
    "detailed_findings": "Explicitly state: (1) how many X chromosomes counted, (2) Y present or absent, (3) count at position 21"
}}"""

        client = OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a board-certified clinical cytogeneticist. CRITICAL: Count sex chromosomes FIRST before checking position 21. For 47-chromosome karyotypes, distinguish: (1) Triple X (47,XXX) = THREE medium-sized X chromosomes in sex region, NO Y, normal position 21; (2) Down syndrome (47,+21) = normal XX or XY sex chromosomes, THREE small acrocentric at position 21. X chromosomes are MEDIUM-sized, chromosome 21 is SMALL. Always return valid JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cv_hint_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
            temperature=0.1
        )

        raw_content = response.choices[0].message.content
        st.session_state.raw_response = raw_content

        # Parse response
        result = self._parse_response(raw_content, "CV+VLM (CV-assisted)")

        return result

    def _mock_analysis(self) -> Dict:
        """Mock 분석 결과 (데모용)"""
        import random

        chromosome_count = random.choice([46, 47, 45])
        sex_chromosomes = random.choice(["XX", "XY"])

        abnormalities = []
        confidence = random.uniform(75, 92)

        if chromosome_count == 47:
            trisomy = random.choice([21, 18, 13])
            abnormalities.append({
                "type": "trisomy",
                "chromosome": str(trisomy),
                "description": f"Trisomy {trisomy} detected - extra copy of chromosome {trisomy}"
            })
            notation = f"47,{sex_chromosomes},+{trisomy}"
        elif chromosome_count == 45:
            if sex_chromosomes == "XX":
                abnormalities.append({
                    "type": "monosomy",
                    "chromosome": "X",
                    "description": "Turner syndrome (Monosomy X) - missing one X chromosome"
                })
                notation = "45,X"
                sex_chromosomes = "X"
            else:
                notation = f"45,{sex_chromosomes},-21"
                abnormalities.append({
                    "type": "monosomy",
                    "chromosome": "21",
                    "description": "Monosomy 21 detected"
                })
        else:
            notation = f"46,{sex_chromosomes}"

        # Add structural abnormality occasionally
        if random.random() < 0.15 and chromosome_count == 46:
            abnormalities.append({
                "type": "translocation",
                "chromosome": "9;22",
                "description": "Balanced translocation t(9;22)(q34;q11) - Philadelphia chromosome"
            })
            notation += ",t(9;22)(q34;q11)"

        interpretation = self._generate_interpretation(chromosome_count, abnormalities, sex_chromosomes)

        return {
            "notation": notation,
            "chromosome_count": chromosome_count,
            "sex_chromosomes": sex_chromosomes,
            "abnormalities": abnormalities,
            "confidence": round(confidence, 1),
            "interpretation": interpretation,
            "detailed_findings": "Demo mode: This is simulated data for demonstration purposes only.",
            "analysis_time": datetime.now().isoformat(),
            "technical_notes": "Demo Mode - No actual image analysis performed",
            "provider": "Demo Mode"
        }

    def _generate_interpretation(self, count: int, abnormalities: list, sex: str) -> str:
        """분석 결과에 대한 임상적 해석 생성"""
        if count == 46 and not abnormalities:
            gender = "female" if sex == "XX" else "male"
            return f"Normal {gender} karyotype with no apparent numerical or structural abnormalities detected."

        interpretations = []
        if count != 46:
            interpretations.append(f"Numerical abnormality: {count} chromosomes detected (normal is 46).")

        for abnormality in abnormalities:
            atype = abnormality.get("type", "")
            chrom = abnormality.get("chromosome", "")

            if atype == "trisomy":
                if chrom == "21":
                    interpretations.append("Trisomy 21 is associated with Down syndrome, characterized by intellectual disability, distinctive facial features, and increased risk of certain medical conditions.")
                elif chrom == "18":
                    interpretations.append("Trisomy 18 (Edwards syndrome) is a severe condition with multiple organ system abnormalities and limited survival.")
                elif chrom == "13":
                    interpretations.append("Trisomy 13 (Patau syndrome) involves severe intellectual disability, heart defects, and other organ abnormalities.")
            elif atype == "monosomy" and chrom == "X":
                interpretations.append("Turner syndrome (45,X) affects females and is characterized by short stature, ovarian insufficiency, and possible cardiac/renal anomalies.")
            elif atype == "translocation":
                interpretations.append(f"Translocation {abnormality.get('description', '')} detected. Clinical significance depends on whether balanced or unbalanced.")

        return " ".join(interpretations)

    def _analyze_with_consensus(self, image: Image.Image, api_keys: Dict) -> Dict:
        """Run analysis on multiple providers and vote on results"""
        results = []
        errors = []
        providers_used = []

        # Run analysis with each available provider
        provider_configs = [
            ('openai', APIProvider.OPENAI, OPENAI_AVAILABLE, "GPT-4 Vision"),
            ('anthropic', APIProvider.ANTHROPIC, ANTHROPIC_AVAILABLE, "Claude Vision"),
            ('gemini', APIProvider.GEMINI, GEMINI_AVAILABLE, "Gemini Vision")
        ]

        settings = st.session_state.consensus_settings

        for key_name, provider_enum, is_available, display_name in provider_configs:
            use_flag = f'use_{key_name}'
            if not settings.get(use_flag, True):
                continue
            if not is_available:
                errors.append(f"{display_name}: Package not installed")
                continue
            if not api_keys.get(key_name):
                errors.append(f"{display_name}: API key not provided")
                continue

            try:
                # Create temporary analyzer for this provider
                temp_analyzer = KaryotypeAnalyzer(provider=provider_enum, api_key=api_keys[key_name])

                if provider_enum == APIProvider.OPENAI:
                    result = temp_analyzer._analyze_with_openai(image)
                elif provider_enum == APIProvider.ANTHROPIC:
                    result = temp_analyzer._analyze_with_anthropic(image)
                elif provider_enum == APIProvider.GEMINI:
                    result = temp_analyzer._analyze_with_gemini(image)

                results.append(result)
                providers_used.append(display_name)

            except Exception as e:
                errors.append(f"{display_name}: {str(e)}")

        # Calculate consensus from results
        if not results:
            return {
                'notation': 'No Results',
                'chromosome_count': 0,
                'sex_chromosomes': 'Unknown',
                'abnormalities': [],
                'confidence': 0,
                'interpretation': 'No successful analyses to aggregate.',
                'detailed_findings': f"Errors: {'; '.join(errors)}",
                'analysis_time': datetime.now().isoformat(),
                'technical_notes': 'Consensus analysis failed - no providers returned results',
                'provider': 'Multi-Model Consensus',
                'is_consensus': True,
                'individual_results': [],
                'agreement_level': 0,
                'voting_breakdown': {},
                'errors': errors
            }

        consensus = self._calculate_consensus(results, providers_used, errors)
        return consensus

    def _calculate_consensus(self, results: List[Dict], providers_used: List[str], errors: List[str]) -> Dict:
        """Calculate consensus from multiple analysis results using majority voting"""
        if len(results) == 1:
            # Only one result - return it with consensus metadata
            single = results[0]
            single['is_consensus'] = True
            single['individual_results'] = results
            single['agreement_level'] = 1.0
            single['providers_used'] = providers_used
            single['voting_breakdown'] = {
                'chromosome_count': {single['chromosome_count']: 1},
                'sex_chromosomes': {single['sex_chromosomes']: 1},
                'notation': {single['notation']: 1}
            }
            single['provider'] = f"Consensus ({providers_used[0]} only)"
            single['errors'] = errors
            return single

        total = len(results)

        # Count votes for each field
        count_votes = Counter(r.get('chromosome_count', 0) for r in results)
        sex_votes = Counter(r.get('sex_chromosomes', 'Unknown') for r in results)
        notation_votes = Counter(r.get('notation', 'Unknown') for r in results)

        # Determine consensus (most common value)
        consensus_count, count_agreement = count_votes.most_common(1)[0]
        consensus_sex, sex_agreement = sex_votes.most_common(1)[0]
        consensus_notation, notation_agreement = notation_votes.most_common(1)[0]

        # Calculate overall agreement level (0.0 - 1.0)
        agreement_level = count_agreement / total

        # Aggregate abnormalities with tracking
        abnormality_counts = {}
        for r in results:
            for abnorm in r.get('abnormalities', []):
                key = f"{abnorm.get('type', 'unknown')}:{abnorm.get('chromosome', 'N/A')}"
                if key not in abnormality_counts:
                    abnormality_counts[key] = {
                        'abnormality': abnorm,
                        'detected_by': [],
                        'count': 0
                    }
                abnormality_counts[key]['count'] += 1
                abnormality_counts[key]['detected_by'].append(r.get('provider', 'Unknown'))

        # Build merged abnormalities list with agreement info
        merged_abnormalities = []
        for key, data in abnormality_counts.items():
            abnorm = data['abnormality'].copy()
            abnorm['agreement'] = f"{data['count']}/{total} models"
            abnorm['detected_by'] = data['detected_by']
            merged_abnormalities.append(abnorm)

        # Calculate consensus confidence
        base_confidence = sum(r.get('confidence', 0) for r in results) / total
        agreement_bonus = agreement_level * 10  # +10% for unanimous
        final_confidence = min(99, base_confidence + agreement_bonus)

        # Generate consensus interpretation
        if agreement_level == 1.0:
            agreement_text = "All models unanimously agree on this analysis."
        elif agreement_level >= 0.67:
            agreement_text = f"Majority agreement ({count_agreement}/{total} models)."
        else:
            agreement_text = f"Models disagree - review individual results carefully."

        # Combine interpretations
        interpretations = [r.get('interpretation', '') for r in results if r.get('interpretation')]
        consensus_interpretation = f"{agreement_text}\n\n" + "\n---\n".join(
            f"**{providers_used[i]}**: {interpretations[i]}"
            for i in range(len(interpretations))
        )

        # Build voting breakdown for display
        voting_breakdown = {
            'chromosome_count': dict(count_votes),
            'sex_chromosomes': dict(sex_votes),
            'notation': dict(notation_votes)
        }

        return {
            'notation': consensus_notation,
            'chromosome_count': consensus_count,
            'sex_chromosomes': consensus_sex,
            'abnormalities': merged_abnormalities,
            'confidence': round(final_confidence, 1),
            'interpretation': consensus_interpretation,
            'detailed_findings': f"Consensus from {total} models: {', '.join(providers_used)}",
            'analysis_time': datetime.now().isoformat(),
            'technical_notes': f"Multi-model consensus voting with {agreement_level:.0%} agreement",
            'provider': 'Multi-Model Consensus',
            'is_consensus': True,
            'individual_results': results,
            'agreement_level': agreement_level,
            'providers_used': providers_used,
            'voting_breakdown': voting_breakdown,
            'errors': errors
        }


def display_header():
    """헤더 표시"""
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Chromosome Karyotype Analyzer</h1>
        <p>AI-Powered Cytogenetic Analysis Tool</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">ISCN 2020 Compliant • Multi-Provider AI • Educational & Research Use Only</p>
    </div>
    """, unsafe_allow_html=True)


def display_disclaimer():
    """의료 면책 조항 표시"""
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and research purposes only.
        Results must be validated by qualified cytogenetics professionals. Do not use for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


def display_api_status():
    """API 상태 표시 - GPT-4 Vision only"""
    st.sidebar.subheader("📦 Package Status")

    status_html = ""
    if OPENAI_AVAILABLE:
        status_html += '<div class="api-status api-available">✓ OpenAI GPT-4 Vision ready</div>'
    else:
        status_html += '<div class="api-status api-unavailable">✗ OpenAI package missing - run: pip install openai</div>'

    st.sidebar.markdown(status_html, unsafe_allow_html=True)


def display_sidebar_settings() -> tuple:
    """사이드바 설정 및 API 키 입력"""
    with st.sidebar:
        st.header("⚙️ Settings")

        # API 상태 표시
        display_api_status()

        st.divider()

        # API 제공자 선택
        st.subheader("🔌 API Provider")

        # Available provider options
        provider_options = ["Demo Mode (No API)"]
        if OPENAI_AVAILABLE:
            provider_options.insert(0, "OpenAI GPT-4 Vision")
            if CV2_AVAILABLE:
                provider_options.insert(1, "CV + VLM (Hybrid)")

        selected_provider = st.selectbox(
            "Select AI Provider",
            provider_options,
            help="CV+VLM uses computer vision for counting, VLM for interpretation"
        )

        # Map selection to enum
        provider_map = {
            "OpenAI GPT-4 Vision": APIProvider.OPENAI,
            "CV + VLM (Hybrid)": APIProvider.CV_VLM,
            "Demo Mode (No API)": APIProvider.MOCK
        }
        provider = provider_map.get(selected_provider, APIProvider.MOCK)

        # Show CV+VLM explanation if selected
        if provider == APIProvider.CV_VLM:
            st.info("**CV + VLM Mode**: Computer Vision counts chromosomes → VLM interprets the counts. More accurate than VLM-only for counting.")

        st.divider()

        # API 키 입력
        api_key = None
        consensus_keys = {}  # Keep for compatibility

        if provider in [APIProvider.OPENAI, APIProvider.CV_VLM]:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key (starts with sk-)"
            )
            st.caption("Get your key at [platform.openai.com](https://platform.openai.com/api-keys)")

        st.divider()

        # 정보 섹션
        st.markdown("""
        ### About
        This tool analyzes chromosome metaphase spreads and generates ISCN 2020 compliant karyotype notations.

        **Available Modes:**
        - **GPT-4 Vision**: VLM-only analysis
        - **CV + VLM (Hybrid)**: Computer Vision counts → VLM interprets

        The CV+VLM hybrid mode provides more accurate chromosome counting by:
        1. Using OpenCV to detect and count chromosomes
        2. Identifying position 21 and sex chromosome regions
        3. Passing counts to VLM for clinical interpretation

        ### Resources
        - [ISCN 2020 Standards](https://karger.com/books/book/367/An-International-System-for-Human-Cytogenomic)
        - [Karyotype Analysis Guide](https://www.ncbi.nlm.nih.gov/books/NBK557817/)
        """)

        return provider, api_key, consensus_keys


def display_upload_section():
    """이미지 업로드 섹션"""
    st.header("📤 Upload Metaphase Spread Image")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a high-quality metaphase spread image (max 10MB)"
        )

        if uploaded_file is not None:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File size exceeds 10MB limit. Please upload a smaller image.")
                return None

            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image

            st.image(image, caption="Uploaded Image", width="stretch")

            st.info(f"**Image Details:**\n"
                   f"- Format: {image.format or 'N/A'}\n"
                   f"- Size: {image.size[0]} x {image.size[1]} pixels\n"
                   f"- Mode: {image.mode}\n"
                   f"- File size: {uploaded_file.size / 1024:.1f} KB")

            return image

    with col2:
        st.markdown("""
        ### Guidelines:
        - Use high-resolution images
        - Ensure clear chromosome spread
        - Avoid overlapping chromosomes
        - Good contrast and lighting

        ### Supported Formats:
        - PNG, JPG, JPEG
        - TIFF, BMP
        - Max size: 10MB

        ### Best Results:
        - G-banded metaphase spreads
        - Individual well-separated chromosomes
        - Consistent staining quality
        """)

    return None


def display_analysis_section(analyzer: KaryotypeAnalyzer, image: Image.Image, provider: APIProvider, consensus_keys: Optional[Dict] = None):
    """분석 섹션"""
    st.header("🔬 Chromosome Analysis")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        provider_name = provider.value
        button_text = f"🚀 Analyze with {provider_name}"

        if provider == APIProvider.MOCK:
            st.info("💡 Demo Mode: Results will be simulated. Select an API provider for real analysis.")

        if provider == APIProvider.CV_VLM:
            st.info("🔬 **CV + VLM Hybrid Mode**: OpenCV will detect and count chromosomes at each position, then GPT-4 will interpret the counts. More accurate for chromosome counting.")

        if provider == APIProvider.CONSENSUS:
            # Show which models will be used
            active_models = []
            if consensus_keys.get('openai') and OPENAI_AVAILABLE:
                active_models.append("GPT-4")
            if consensus_keys.get('anthropic') and ANTHROPIC_AVAILABLE:
                active_models.append("Claude")
            if consensus_keys.get('gemini') and GEMINI_AVAILABLE:
                active_models.append("Gemini")
            st.info(f"🗳️ Consensus Voting: Will analyze with {', '.join(active_models)} and combine results")

        if st.button(button_text, type="primary", width="stretch"):
            if provider == APIProvider.CONSENSUS:
                spinner_text = f"Running multi-model consensus analysis... This may take 2-3 minutes."
            else:
                spinner_text = f"Analyzing chromosomes using {provider_name}... This may take up to 60 seconds."

            with st.spinner(spinner_text):
                try:
                    progress_bar = st.progress(0)
                    progress_bar.progress(10)

                    if provider == APIProvider.CONSENSUS:
                        # Pass consensus_keys to the analyzer
                        result = analyzer.analyze(image, consensus_keys=consensus_keys)
                    else:
                        result = analyzer.analyze(image)

                    progress_bar.progress(100)
                    st.session_state.analysis_result = result

                    if result.get('is_consensus'):
                        agreement = result.get('agreement_level', 0)
                        if agreement == 1.0:
                            st.success("✅ Consensus analysis complete - All models agree!")
                        elif agreement >= 0.67:
                            st.success("✅ Consensus analysis complete - Majority agreement achieved!")
                        else:
                            st.warning("⚠️ Consensus analysis complete - Models disagree, review carefully.")
                        st.balloons()
                    elif result.get('pipeline') == 'two_stage':
                        cv_count = result.get('cv_detection', {}).get('count', 0)
                        st.success(f"✅ Two-Stage Analysis complete - CV detected {cv_count} chromosomes!")
                        st.balloons()
                    elif result.get('confidence', 0) > 0:
                        st.success("✅ Analysis completed successfully!")
                        st.balloons()
                    else:
                        st.warning("⚠️ Analysis completed with issues. Check results below.")

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)


def get_confidence_class(confidence: float) -> str:
    """신뢰도에 따른 CSS 클래스 반환"""
    if confidence >= 85:
        return "confidence-high"
    elif confidence >= 65:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_results(result: Dict):
    """분석 결과 표시"""
    st.header("📊 Analysis Results")

    # Provider badge
    provider = result.get('provider', 'Unknown')
    st.caption(f"🤖 Analyzed by: **{provider}**")

    # ISCN Notation
    st.markdown(f"""
    <div class="notation-display">
        {result.get('notation', 'N/A')}
    </div>
    """, unsafe_allow_html=True)

    # 주요 결과 요약
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Chromosome Count", result.get('chromosome_count', 'N/A'))

    with col2:
        st.metric("Sex Chromosomes", result.get('sex_chromosomes', 'N/A'))

    with col3:
        confidence = result.get('confidence', 0)
        confidence_class = get_confidence_class(confidence)
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Confidence Score</h4>
            <p class="{confidence_class}" style="font-size: 2rem;">{confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # 상세 결과
    st.markdown("""<div class="result-card">""", unsafe_allow_html=True)

    st.subheader("🔍 Detected Abnormalities")
    abnormalities = result.get('abnormalities', [])
    if abnormalities:
        for i, abnormality in enumerate(abnormalities):
            chrom = abnormality.get('chromosome', 'N/A')
            desc = abnormality.get('description', 'No description')
            atype = abnormality.get('type', 'unknown')
            st.write(f"{i+1}. **{atype.title()}** (Chromosome {chrom}): {desc}")
    else:
        st.write("✓ No chromosomal abnormalities detected.")

    st.subheader("📝 Clinical Interpretation")
    st.write(result.get('interpretation', 'No interpretation available.'))

    if result.get('detailed_findings'):
        st.subheader("🔎 Detailed Findings")
        st.write(result.get('detailed_findings'))

    st.subheader("🔧 Technical Notes")
    st.write(result.get('technical_notes', 'N/A'))
    st.caption(f"Analysis performed at: {result.get('analysis_time', 'N/A')}")

    st.markdown("""</div>""", unsafe_allow_html=True)

    # Raw response expander (for debugging)
    if st.session_state.raw_response:
        with st.expander("📄 View Raw API Response"):
            st.code(st.session_state.raw_response, language="json")


def display_consensus_results(result: Dict):
    """Display consensus voting results with breakdown"""
    st.header("🗳️ Multi-Model Consensus Results")

    # Agreement level indicator
    agreement_level = result.get('agreement_level', 0)
    providers_used = result.get('providers_used', [])
    total_models = len(providers_used)

    if agreement_level == 1.0:
        agreement_class = "consensus-agree"
        agreement_icon = "✅"
        agreement_text = f"All {total_models} models agree!"
    elif agreement_level >= 0.67:
        agreement_class = "consensus-partial"
        agreement_icon = "🟡"
        agreement_text = f"Majority agreement ({int(agreement_level * total_models)}/{total_models} models)"
    else:
        agreement_class = "consensus-disagree"
        agreement_icon = "⚠️"
        agreement_text = "Models disagree - review individual results"

    st.markdown(f"""
    <div class="{agreement_class}">
        <h3>{agreement_icon} {agreement_text}</h3>
        <p>Providers: {', '.join(providers_used)}</p>
    </div>
    """, unsafe_allow_html=True)

    # Consensus ISCN Notation
    st.markdown(f"""
    <div class="notation-display">
        <strong>Consensus:</strong> {result.get('notation', 'N/A')}
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Chromosome Count", result.get('chromosome_count', 'N/A'))

    with col2:
        st.metric("Sex Chromosomes", result.get('sex_chromosomes', 'N/A'))

    with col3:
        confidence = result.get('confidence', 0)
        confidence_class = get_confidence_class(confidence)
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Confidence</h4>
            <p class="{confidence_class}" style="font-size: 1.5rem;">{confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Agreement</h4>
            <p style="font-size: 1.5rem; font-weight: bold;">{agreement_level:.0%}</p>
        </div>
        """, unsafe_allow_html=True)

    # Voting breakdown table
    st.subheader("🗳️ Voting Breakdown")

    voting_breakdown = result.get('voting_breakdown', {})
    individual_results = result.get('individual_results', [])

    if individual_results:
        # Create comparison table
        table_data = []
        for i, r in enumerate(individual_results):
            provider_name = providers_used[i] if i < len(providers_used) else r.get('provider', 'Unknown')
            table_data.append({
                'Model': provider_name,
                'Notation': r.get('notation', 'N/A'),
                'Count': r.get('chromosome_count', 'N/A'),
                'Sex': r.get('sex_chromosomes', 'N/A'),
                'Confidence': f"{r.get('confidence', 0):.1f}%"
            })

        # Display as table
        import pandas as pd
        df = pd.DataFrame(table_data)

        # Highlight consensus values
        def highlight_consensus(row):
            styles = [''] * len(row)
            if row['Count'] == result.get('chromosome_count'):
                styles[2] = 'background-color: #D1FAE5'
            if row['Sex'] == result.get('sex_chromosomes'):
                styles[3] = 'background-color: #D1FAE5'
            return styles

        styled_df = df.style.apply(highlight_consensus, axis=1)
        st.dataframe(styled_df, hide_index=True, use_container_width=True)

    # Abnormalities with agreement info
    st.subheader("🔍 Detected Abnormalities")
    abnormalities = result.get('abnormalities', [])

    if abnormalities:
        for i, abnormality in enumerate(abnormalities):
            chrom = abnormality.get('chromosome', 'N/A')
            desc = abnormality.get('description', 'No description')
            atype = abnormality.get('type', 'unknown')
            agreement = abnormality.get('agreement', 'N/A')
            detected_by = abnormality.get('detected_by', [])

            st.markdown(f"""
            **{i+1}. {atype.title()}** (Chromosome {chrom})
            - {desc}
            - *Agreement: {agreement}*
            - *Detected by: {', '.join(detected_by)}*
            """)
    else:
        st.success("✓ No chromosomal abnormalities detected by consensus.")

    # Individual model results (expandable)
    st.subheader("📋 Individual Model Results")

    for i, r in enumerate(individual_results):
        provider_name = providers_used[i] if i < len(providers_used) else r.get('provider', 'Unknown')
        with st.expander(f"🤖 {provider_name}: {r.get('notation', 'N/A')}"):
            st.write(f"**Chromosome Count:** {r.get('chromosome_count', 'N/A')}")
            st.write(f"**Sex Chromosomes:** {r.get('sex_chromosomes', 'N/A')}")
            st.write(f"**Confidence:** {r.get('confidence', 0):.1f}%")
            st.write(f"**Interpretation:** {r.get('interpretation', 'N/A')}")

            if r.get('abnormalities'):
                st.write("**Abnormalities:**")
                for abnorm in r.get('abnormalities', []):
                    st.write(f"- {abnorm.get('type', 'unknown')}: {abnorm.get('description', 'N/A')}")

    # Errors section (if any)
    errors = result.get('errors', [])
    if errors:
        with st.expander("⚠️ Provider Errors"):
            for error in errors:
                st.warning(error)


def display_two_stage_results(result: Dict):
    """Display two-stage pipeline results with CV detection info"""
    st.header("🔬 Two-Stage Pipeline Results")

    # Pipeline badge
    st.info("📊 **Stage 1:** Computer Vision Detection → **Stage 2:** VLM Classification")

    # CV Detection results
    cv_detection = result.get('cv_detection', st.session_state.cv_detection or {})

    if cv_detection:
        st.subheader("🖥️ Stage 1: Computer Vision Detection")

        cv_col1, cv_col2, cv_col3 = st.columns(3)

        with cv_col1:
            cv_count = cv_detection.get('count', 0)
            st.metric("CV Detected Count", cv_count)

        with cv_col2:
            sex_region = cv_detection.get('sex_chromosome_region', {})
            sex_est = sex_region.get('estimated', 'unknown')
            st.metric("Estimated Sex Chr", sex_est)

        with cv_col3:
            method = cv_detection.get('detection_method', 'N/A')
            st.metric("Detection Method", method.replace('_', ' ').title())

        # Denver group estimates
        groups = cv_detection.get('group_counts', {})
        if groups:
            with st.expander("📊 CV Denver Group Distribution"):
                group_data = {
                    'Group': ['A (1-3)', 'B (4-5)', 'C (6-12+X)', 'D (13-15)', 'E (16-18)', 'F (19-20)', 'G (21-22+Y)'],
                    'Count': [groups.get('A', 0), groups.get('B', 0), groups.get('C', 0),
                             groups.get('D', 0), groups.get('E', 0), groups.get('F', 0), groups.get('G', 0)],
                    'Expected': [6, 4, 15, 6, 6, 4, 5]  # For normal male
                }
                import pandas as pd
                df = pd.DataFrame(group_data)
                st.dataframe(df, hide_index=True, use_container_width=True)

                total_cv = sum(groups.values())
                st.caption(f"**CV Total:** {total_cv} chromosomes")

    # Stage 2: VLM Classification results
    st.subheader("🤖 Stage 2: VLM Classification")

    # ISCN Notation
    st.markdown(f"""
    <div class="notation-display">
        {result.get('notation', 'N/A')}
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Final Chromosome Count", result.get('chromosome_count', 'N/A'))

    with col2:
        st.metric("Sex Chromosomes", result.get('sex_chromosomes', 'N/A'))

    with col3:
        confidence = result.get('confidence', 0)
        confidence_class = get_confidence_class(confidence)
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Confidence Score</h4>
            <p class="{confidence_class}" style="font-size: 2rem;">{confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Abnormalities
    st.subheader("🔍 Detected Abnormalities")
    abnormalities = result.get('abnormalities', [])
    if abnormalities:
        for i, abnormality in enumerate(abnormalities):
            chrom = abnormality.get('chromosome', 'N/A')
            desc = abnormality.get('description', 'No description')
            atype = abnormality.get('type', 'unknown')
            st.write(f"{i+1}. **{atype.title()}** (Chromosome {chrom}): {desc}")
    else:
        st.success("✓ No chromosomal abnormalities detected.")

    st.subheader("📝 Clinical Interpretation")
    st.write(result.get('interpretation', 'No interpretation available.'))

    if result.get('detailed_findings'):
        st.subheader("🔎 Detailed Findings")
        st.write(result.get('detailed_findings'))

    # Technical notes
    st.subheader("🔧 Technical Notes")
    st.write(f"Pipeline: Two-Stage (CV + VLM)")
    st.caption(f"Analysis performed at: {result.get('analysis_time', 'N/A')}")

    # Raw response
    if st.session_state.raw_response:
        with st.expander("📄 View Raw API Response"):
            st.code(st.session_state.raw_response, language="json")


def generate_report(result: Dict) -> str:
    """텍스트 보고서 생성"""
    abnormalities_text = ""
    if result.get('abnormalities'):
        for i, abnormality in enumerate(result['abnormalities']):
            abnormalities_text += f"{i+1}. {abnormality.get('type', 'Unknown')}: {abnormality.get('description', 'N/A')}\n"
    else:
        abnormalities_text = "No chromosomal abnormalities detected.\n"

    report = f"""
CHROMOSOME KARYOTYPE ANALYSIS REPORT
====================================

Analysis Date: {result.get('analysis_time', 'N/A')}
AI Provider: {result.get('provider', 'N/A')}

ISCN 2020 Notation: {result.get('notation', 'N/A')}

SUMMARY
-------
Chromosome Count: {result.get('chromosome_count', 'N/A')}
Sex Chromosomes: {result.get('sex_chromosomes', 'N/A')}
Confidence Score: {result.get('confidence', 0):.1f}%

DETECTED ABNORMALITIES
---------------------
{abnormalities_text}

CLINICAL INTERPRETATION
----------------------
{result.get('interpretation', 'N/A')}

DETAILED FINDINGS
----------------
{result.get('detailed_findings', 'N/A')}

TECHNICAL NOTES
--------------
{result.get('technical_notes', 'N/A')}

DISCLAIMER
----------
This analysis is for educational and research purposes only.
Results must be validated by qualified cytogenetics professionals.
Do not use for clinical diagnosis or medical decision-making.

Generated by Chromosome Karyotype Analyzer
Powered by Vision-Language AI Models
"""

    return report


def display_report_section(result: Dict):
    """보고서 다운로드 섹션"""
    st.header("📄 Generate Report")

    col1, col2 = st.columns(2)

    with col1:
        report_text = generate_report(result)
        st.download_button(
            label="📥 Download Text Report",
            data=report_text,
            file_name=f"karyotype_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    with col2:
        st.download_button(
            label="📥 Download JSON Data",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=f"karyotype_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """메인 애플리케이션"""
    # 헤더 표시
    display_header()

    # 면책 조항
    display_disclaimer()

    # 사이드바 설정 (API 제공자 및 키)
    provider, api_key, consensus_keys = display_sidebar_settings()

    # 분석기 초기화
    analyzer = KaryotypeAnalyzer(provider=provider, api_key=api_key)

    # 이미지 업로드
    image = display_upload_section()

    # 분석 섹션
    if image is not None:
        # API 키 확인 (Mock 모드가 아닌 경우)
        if provider == APIProvider.CONSENSUS:
            # Check if at least 2 API keys are provided for consensus
            valid_keys = sum(1 for k, v in consensus_keys.items() if v)
            if valid_keys < 2:
                st.warning("⚠️ Please provide API keys for at least 2 models in the sidebar for consensus voting.")
            else:
                display_analysis_section(analyzer, image, provider, consensus_keys)
        elif provider != APIProvider.MOCK and not api_key:
            st.warning(f"⚠️ Please enter your {provider.value} API key in the sidebar to proceed with analysis.")
        else:
            display_analysis_section(analyzer, image, provider)

    # 결과 표시
    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result

        # Use appropriate display function based on result type
        if result.get('is_consensus'):
            display_consensus_results(result)
        elif result.get('pipeline') == 'two_stage':
            display_two_stage_results(result)
        else:
            display_results(result)

        display_report_section(result)

        # 새 분석 버튼
        if st.button("🔄 Start New Analysis"):
            st.session_state.analysis_result = None
            st.session_state.uploaded_image = None
            st.session_state.raw_response = None
            st.rerun()


if __name__ == "__main__":
    main()
