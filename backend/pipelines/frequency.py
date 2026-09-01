"""
Frequency / FFT Defect Detection Pipeline

Flow:
Grayscale
→ 2D FFT (np.fft.fft2)
→ FFT shift (np.fft.fftshift)
→ magnitude spectrum (20 * log(|fshift| + 1))
→ frequency filtering (High-pass / Band-pass Gaussian filter)
→ inverse FFT (np.fft.ifft2)
→ threshold (Otsu adaptive threshold)
→ anomaly mask

Calculates genuine frequency domain metrics:
- total frequency energy
- low-frequency energy
- high-frequency energy
- high/low frequency ratio
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


def run_frequency_pipeline(preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes ONLY Frequency / FFT Defect Analysis.

    Input: Output dictionary from preprocess_image() containing 'original_bgr', 'grayscale', 'width', 'height'.
    
    Output dictionary:
        - fft_spectrum: Base64 data URI string
        - filtered_frequency_representation: Base64 data URI string
        - reconstructed_image: Base64 data URI string
        - anomaly_mask: Base64 data URI string
        - overlay: Base64 data URI string
        - total_frequency_energy: float
        - low_frequency_energy: float
        - high_frequency_energy: float
        - high_to_low_frequency_ratio: float
        - anomaly_area: int (pixel count)
        - region_count: int
        - processing_time: float (milliseconds)
    """
    start_time = time.perf_counter()

    gray = preprocessed_data["grayscale"]
    original_bgr = preprocessed_data["original_bgr"]
    rows, cols = gray.shape

    # Step 1: 2D FFT & FFT Shift
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Step 2: Compute Magnitude Spectrum for Visualization
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1.0)
    magnitude_spectrum_vis = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 3: Spectral Energy Metrics (Total, Low-Frequency, High-Frequency)
    power_spectrum = np.abs(fshift) ** 2
    total_energy = float(np.sum(power_spectrum))

    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    radius_low = min(rows, cols) * 0.12  # Central radius defining low-frequency zone

    low_freq_mask = dist_from_center <= radius_low
    high_freq_mask = ~low_freq_mask

    low_frequency_energy = float(np.sum(power_spectrum[low_freq_mask]))
    high_frequency_energy = float(np.sum(power_spectrum[high_freq_mask]))
    hl_ratio = round(high_frequency_energy / (low_frequency_energy + 1e-8), 4)

    # Step 4: Frequency Domain High-Pass / Band-Pass Filtering
    # Smooth Gaussian High-Pass Filter mask to attenuate low frequencies and highlight localized defects
    filter_mask = 1.0 - np.exp(-((dist_from_center ** 2) / (2 * (radius_low ** 2))))
    fshift_filtered = fshift * filter_mask

    # Filtered Frequency Spectrum Visualization
    filtered_magnitude = 20 * np.log(np.abs(fshift_filtered) + 1.0)
    filtered_spectrum_vis = cv2.normalize(filtered_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 5: Inverse 2D FFT & Spatial Reconstruction
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_reconstructed = np.abs(np.fft.ifft2(f_ishift))
    reconstructed_vis = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 6: Thresholding & Anomaly Mask Generation
    _, thresh_img = cv2.threshold(reconstructed_vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)

    # Connected Components Filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
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

    # Step 7: Visual Overlay (Highlight spectral defects with magenta tint & contours)
    overlay = original_bgr.copy()
    magenta_mask = np.zeros_like(original_bgr)
    magenta_mask[final_mask == 255] = [255, 0, 255]  # Magenta tint for frequency anomalies
    overlay = cv2.addWeighted(overlay, 0.75, magenta_mask, 0.25, 0)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 255), 2)

    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "pipeline": "Frequency Domain Analysis (FFT)",
        "status": "success",
        "fft_spectrum": encode_image_to_base64(magnitude_spectrum_vis),
        "filtered_frequency_representation": encode_image_to_base64(filtered_spectrum_vis),
        "reconstructed_image": encode_image_to_base64(reconstructed_vis),
        "anomaly_mask": encode_image_to_base64(final_mask),
        "overlay": encode_image_to_base64(overlay),
        "total_frequency_energy": float(f"{total_energy:.2e}"),
        "low_frequency_energy": float(f"{low_frequency_energy:.2e}"),
        "high_frequency_energy": float(f"{high_frequency_energy:.2e}"),
        "high_to_low_frequency_ratio": hl_ratio,
        "anomaly_area": anomaly_area,
        "region_count": region_count,
        "processing_time": processing_time_ms
    }


def frequency_pipeline_stub(preprocessed_data: Optional[Dict[str, Any]] = None) -> dict:
    """Stub helper for system health check."""
    if preprocessed_data:
        return run_frequency_pipeline(preprocessed_data)
    return {
        "name": "Frequency Domain Analysis",
        "status": "ready_for_implementation",
        "description": "Fourier Transform (FFT) and bandpass frequency defect inspection."
    }
