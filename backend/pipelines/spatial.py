"""
Spatial / Intensity Defect Detection Pipeline

Flow:
Grayscale
→ contrast enhancement (CLAHE)
→ thresholding (Otsu adaptive threshold)
→ morphological cleanup (Opening/Closing)
→ connected components
→ anomaly mask
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


def run_spatial_pipeline(preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes ONLY Spatial / Intensity Defect Analysis.

    Input: Output dictionary from preprocess_image() containing 'original_bgr', 'grayscale', 'width', 'height'.
    
    Output dictionary:
        - enhanced_image: Base64 data URI string
        - threshold_image: Base64 data URI string
        - final_mask: Base64 data URI string
        - overlay: Base64 data URI string
        - anomaly_detected: bool
        - anomaly_area: int (pixel count)
        - anomaly_area_percentage: float (%)
        - region_count: int
        - processing_time: float (milliseconds)
    """
    start_time = time.perf_counter()

    gray = preprocessed_data["grayscale"]
    original_bgr = preprocessed_data["original_bgr"]
    width = preprocessed_data["width"]
    height = preprocessed_data["height"]
    total_pixels = width * height

    # Step 1: Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 2: Gaussian Blur + Adaptive Otsu Thresholding
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, threshold_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: Morphological Cleanup (Opening removes small noise, Closing bridges defect gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    # Step 4: Connected Components Analysis & Anomaly Mask Generation
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8)

    # Minimum area filter to discard spurious pixel noise (< 15 pixels)
    min_defect_area = 15
    final_mask = np.zeros_like(gray, dtype=np.uint8)
    region_count = 0
    anomaly_area = 0

    for i in range(1, num_labels):  # Skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_defect_area:
            final_mask[labels == i] = 255
            region_count += 1
            anomaly_area += int(area)

    anomaly_detected = bool(region_count > 0 and anomaly_area > 0)
    anomaly_area_percentage = round((anomaly_area / total_pixels) * 100, 2)

    # Step 5: Generate Visual Overlay (Highlight defect contours & transparent red tint on original BGR image)
    overlay = original_bgr.copy()
    red_mask = np.zeros_like(original_bgr)
    red_mask[final_mask == 255] = [0, 0, 255]  # Red highlight for defect regions
    overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    # Calculate processing time in milliseconds
    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "pipeline": "Spatial / Intensity Analysis",
        "status": "success",
        "enhanced_image": encode_image_to_base64(enhanced),
        "threshold_image": encode_image_to_base64(threshold_img),
        "final_mask": encode_image_to_base64(final_mask),
        "overlay": encode_image_to_base64(overlay),
        "anomaly_detected": anomaly_detected,
        "anomaly_area": anomaly_area,
        "anomaly_area_percentage": anomaly_area_percentage,
        "region_count": region_count,
        "processing_time": processing_time_ms
    }


def spatial_pipeline_stub(preprocessed_data: Optional[Dict[str, Any]] = None) -> dict:
    """Stub helper for system health check."""
    if preprocessed_data:
        return run_spatial_pipeline(preprocessed_data)
    return {
        "name": "Spatial Domain Analysis",
        "status": "ready_for_implementation",
        "description": "Spatial domain filtering, thresholding, and morphologic defect isolation."
    }
