import cv2
import numpy as np
import base64
from typing import Dict, Any, Union, Tuple


def encode_image_to_base64(img: np.ndarray, format: str = ".png") -> str:
    """Helper function to encode an OpenCV image array to a base64 data URI string."""
    success, buffer = cv2.imencode(format, img)
    if not success:
        raise ValueError("Failed to encode image to base64 buffer")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime_type = "image/png" if format == ".png" else "image/jpeg"
    return f"data:{mime_type};base64,{b64_str}"


def preprocess_image(image_input: Union[bytes, np.ndarray]) -> Dict[str, Any]:
    """
    Common preprocessing for DefectVision image pipelines.

    Input:
        image_input: Raw image bytes OR decoded BGR numpy array (OpenCV format)

    Output dictionary containing:
        - 'original_bgr': Original BGR numpy array
        - 'grayscale': Grayscale uint8 numpy array
        - 'normalized': Contrast normalized & smoothed uint8 numpy array
        - 'width': Image width in pixels
        - 'height': Image height in pixels
        - 'original_base64': Base64 data URL string for original image
        - 'grayscale_base64': Base64 data URL string for grayscale image
        - 'normalized_base64': Base64 data URL string for normalized image
    """
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from provided bytes.")
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
        if len(img_bgr.shape) == 2:
            # Grayscale image passed; convert to BGR for consistency
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    else:
        raise TypeError("Input must be image bytes or a numpy ndarray.")

    height, width = img_bgr.shape[:2]

    # 1. Grayscale Conversion
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Normalization & Preprocessing (Min-Max contrast scaling + gentle Denoising)
    # Scale intensity range to full [0, 255]
    norm_gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Apply Gaussian Blur to smooth sensor noise while preserving structural edges
    normalized = cv2.GaussianBlur(norm_gray, (3, 3), 0)

    # Generate Base64 previews for frontend consumption
    original_b64 = encode_image_to_base64(img_bgr)
    gray_b64 = encode_image_to_base64(gray)
    norm_b64 = encode_image_to_base64(normalized)

    return {
        "original_bgr": img_bgr,
        "grayscale": gray,
        "normalized": normalized,
        "width": width,
        "height": height,
        "original_base64": original_b64,
        "grayscale_base64": gray_b64,
        "normalized_base64": norm_b64
    }
