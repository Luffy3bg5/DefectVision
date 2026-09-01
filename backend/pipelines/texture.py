"""
Texture / Gabor Defect Detection Pipeline

Uses Gabor Filter Bank with:
- Orientations: 0°, 45°, 90°, 135°
- Multiple frequencies / scales (lambda: 4.0, 8.0, 12.0)

Flow:
Grayscale
→ Gabor filter bank (multiple orientations & scales)
→ combine filter responses
→ threshold (Otsu adaptive threshold)
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


def calculate_entropy(img_array: np.ndarray) -> float:
    """Calculates normalized Shannon entropy of response magnitudes."""
    hist, _ = np.histogram(img_array.ravel(), bins=32, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    hist = hist[hist > 0]
    return round(-float(np.sum(hist * np.log2(hist))), 2)


def run_texture_pipeline(preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes ONLY Texture / Gabor Filter Analysis.

    Input: Output dictionary from preprocess_image() containing 'original_bgr', 'grayscale', 'width', 'height'.
    
    Output dictionary:
        - gabor_0_response: Base64 data URI string
        - gabor_45_response: Base64 data URI string
        - gabor_90_response: Base64 data URI string
        - gabor_135_response: Base64 data URI string
        - combined_response: Base64 data URI string
        - anomaly_mask: Base64 data URI string
        - overlay: Base64 data URI string
        - texture_statistics: dict (mean, variance, energy, contrast, entropy)
        - anomaly_area: int (pixel count)
        - region_count: int
        - processing_time: float (milliseconds)
    """
    start_time = time.perf_counter()

    gray = preprocessed_data["grayscale"]
    original_bgr = preprocessed_data["original_bgr"]
    rows, cols = gray.shape

    # Step 1: Define Gabor Filter Parameters (4 Orientations x 3 Scales)
    orientations = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135°
    lambdas = [4.0, 8.0, 12.0]  # Multiple spatial scales / frequencies
    ksize = 21

    orientation_responses = [np.zeros((rows, cols), dtype=np.float32) for _ in range(4)]
    combined_response = np.zeros((rows, cols), dtype=np.float32)

    # Step 2: Gabor Filter Bank Processing
    for ori_idx, theta in enumerate(orientations):
        for lm in lambdas:
            kernel = cv2.getGaborKernel(
                ksize=(ksize, ksize),
                sigma=3.0,
                theta=theta,
                lambd=lm,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            abs_filtered = np.abs(filtered)

            # Store maximum response across scales for each orientation
            orientation_responses[ori_idx] = np.maximum(orientation_responses[ori_idx], abs_filtered)
            combined_response = np.maximum(combined_response, abs_filtered)

    # Step 3: Normalize Filter Responses to uint8 [0, 255] for Visualizations
    gabor_0_vis = cv2.normalize(orientation_responses[0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gabor_45_vis = cv2.normalize(orientation_responses[1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gabor_90_vis = cv2.normalize(orientation_responses[2], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gabor_135_vis = cv2.normalize(orientation_responses[3], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    combined_vis = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 4: Texture Statistics Calculation
    mean_resp = round(float(np.mean(combined_response)), 2)
    variance_resp = round(float(np.var(combined_response)), 2)
    energy_resp = round(float(np.mean(combined_response ** 2)), 2)
    contrast_resp = round(float(np.std(combined_response)), 2)
    entropy_resp = calculate_entropy(combined_vis)

    texture_statistics = {
        "mean_response": mean_resp,
        "variance": variance_resp,
        "energy": energy_resp,
        "contrast": contrast_resp,
        "entropy": entropy_resp
    }

    # Step 5: Thresholding & Anomaly Mask Generation
    _, thresh_img = cv2.threshold(combined_vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)

    # Connected Components Filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8)
    min_defect_area = 15
    final_mask = np.zeros_like(gray, dtype=np.uint8)
    region_count = 0
    anomaly_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_defect_area:
            final_mask[labels == i] = 255
            region_count += 1
            anomaly_area += int(area)

    # Step 6: Visual Overlay (Highlight texture defects with emerald green tint & contours)
    overlay = original_bgr.copy()
    emerald_mask = np.zeros_like(original_bgr)
    emerald_mask[final_mask == 255] = [0, 255, 128]  # Emerald green tint for texture anomalies
    overlay = cv2.addWeighted(overlay, 0.75, emerald_mask, 0.25, 0)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "pipeline": "Texture / Gabor Filter Analysis",
        "status": "success",
        "gabor_0_response": encode_image_to_base64(gabor_0_vis),
        "gabor_45_response": encode_image_to_base64(gabor_45_vis),
        "gabor_90_response": encode_image_to_base64(gabor_90_vis),
        "gabor_135_response": encode_image_to_base64(gabor_135_vis),
        "combined_response": encode_image_to_base64(combined_vis),
        "anomaly_mask": encode_image_to_base64(final_mask),
        "overlay": encode_image_to_base64(overlay),
        "texture_statistics": texture_statistics,
        "anomaly_area": anomaly_area,
        "region_count": region_count,
        "processing_time": processing_time_ms
    }


def texture_pipeline_stub(preprocessed_data: Optional[Dict[str, Any]] = None) -> dict:
    """Stub helper for system health check."""
    if preprocessed_data:
        return run_texture_pipeline(preprocessed_data)
    return {
        "name": "Texture Analysis",
        "status": "ready_for_implementation",
        "description": "GLCM, Gabor filters, and Local Binary Pattern (LBP) texture defect detection."
    }
