"""
Edge / Shape Defect Detection Pipeline

Flow:
Grayscale
→ Gaussian smoothing
→ Canny edge detection
→ contour extraction
→ region filtering
→ anomaly mask

Calculates shape features per region:
- area
- perimeter
- width
- height
- aspect ratio
- circularity
"""
from typing import Dict, Any, Optional
import cv2
import numpy as np
import time
import base64


def encode_image_to_base64(img: np.ndarray, format: str = ".png") -> str:
    """Helper function to convert OpenCV image array to base64 data URI string."""
    success, buffer = cv2.imencode(format, img)
    if not success:
        raise ValueError("Failed to encode image buffer")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime_type = "image/png" if format == ".png" else "image/jpeg"
    return f"data:{mime_type};base64,{b64_str}"


def run_edge_shape_pipeline(preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes ONLY Edge / Shape Defect Analysis.

    Input: Output dictionary from preprocess_image() containing 'original_bgr', 'grayscale', 'width', 'height'.
    
    Output dictionary:
        - edge_map: Base64 data URI string
        - contour_visualization: Base64 data URI string
        - anomaly_mask: Base64 data URI string
        - overlay: Base64 data URI string
        - shape_features: List of feature dictionaries per detected contour region
        - anomaly_area: int (pixel count)
        - region_count: int
        - processing_time: float (milliseconds)
    """
    start_time = time.perf_counter()

    gray = preprocessed_data["grayscale"]
    original_bgr = preprocessed_data["original_bgr"]
    width = preprocessed_data["width"]
    height = preprocessed_data["height"]

    # Step 1: Gaussian Smoothing
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Step 2: Canny Edge Detection with Otsu-guided thresholding
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edge_map = cv2.Canny(blurred, low_thresh, high_thresh)

    # Step 3: Contour Extraction
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Region Filtering & Shape Features Calculation
    min_contour_area = 15.0
    valid_contours = []
    shape_features = []
    anomaly_mask = np.zeros_like(gray, dtype=np.uint8)

    region_idx = 1
    for c in contours:
        area = float(cv2.contourArea(c))
        if area >= min_contour_area:
            perimeter = float(cv2.arcLength(c, True))
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = round(w / float(h), 2) if h > 0 else 0.0
            circularity = round((4.0 * np.pi * area) / (perimeter ** 2), 2) if perimeter > 0 else 0.0

            valid_contours.append(c)
            # Fill region contour on anomaly mask
            cv2.drawContours(anomaly_mask, [c], -1, 255, -1)

            shape_features.append({
                "region_id": region_idx,
                "area": round(area, 2),
                "perimeter": round(perimeter, 2),
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio,
                "circularity": circularity
            })
            region_idx += 1

    # Step 5: Contour Visualization Canvas
    contour_vis = np.zeros_like(original_bgr)
    for c in valid_contours:
        # Draw contour boundary in bright cyan
        cv2.drawContours(contour_vis, [c], -1, (255, 255, 0), 2)
        # Draw bounding rectangle in yellow
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(contour_vis, (x, y), (x + w, y + h), (0, 255, 255), 1)

    # Step 6: Visual Overlay (Highlight defect contours & tint on original image)
    overlay = original_bgr.copy()
    tint_mask = np.zeros_like(original_bgr)
    tint_mask[anomaly_mask == 255] = [255, 0, 0]  # Blue/Cyan tint for shape defects
    overlay = cv2.addWeighted(overlay, 0.75, tint_mask, 0.25, 0)
    cv2.drawContours(overlay, valid_contours, -1, (0, 255, 0), 2)

    anomaly_area = int(cv2.countNonZero(anomaly_mask))
    region_count = len(shape_features)
    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "pipeline": "Edge / Shape Analysis",
        "status": "success",
        "edge_map": encode_image_to_base64(edge_map),
        "contour_visualization": encode_image_to_base64(contour_vis),
        "anomaly_mask": encode_image_to_base64(anomaly_mask),
        "overlay": encode_image_to_base64(overlay),
        "shape_features": shape_features,
        "anomaly_area": anomaly_area,
        "region_count": region_count,
        "processing_time": processing_time_ms
    }


def edge_shape_pipeline_stub(preprocessed_data: Optional[Dict[str, Any]] = None) -> dict:
    """Stub helper for system health check."""
    if preprocessed_data:
        return run_edge_shape_pipeline(preprocessed_data)
    return {
        "name": "Edge & Shape Analysis",
        "status": "ready_for_implementation",
        "description": "Contour analysis, edge detection, and geometric shape defect metrics."
    }
