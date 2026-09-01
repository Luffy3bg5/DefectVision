import time
import base64
import cv2
import numpy as np
from typing import Dict, List, Any

def run_consensus_pipeline(
    original_bgr: np.ndarray,
    masks_dict: Dict[str, np.ndarray],
    min_agreement: int = 2
) -> Dict[str, Any]:
    """
    Computes a pixel-wise voting consensus across executed defect detection pipeline masks.
    
    Args:
        original_bgr: Original image matrix (BGR, HxWx3)
        masks_dict: Dictionary mapping pipeline names to binary uint8 masks (255=anomaly, 0=background)
        min_agreement: Minimum supporting pipeline vote threshold (default >= 2)
        
    Returns:
        Dictionary containing consensus_mask, consensus_overlay, supporting_pipeline_count,
        consensus_area, agreement_summary, and processing_time.
    """
    start_time = time.time()
    
    h, w = original_bgr.shape[:2]
    total_executed = len(masks_dict)
    
    if total_executed < 2:
        raise ValueError("Consensus requires at least 2 executed pipelines.")
        
    # Stack masks into 3D array (H, W, K)
    mask_list = []
    for name, mask in masks_dict.items():
        if mask.shape[:2] != (h, w):
            resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_list.append((resized > 0).astype(np.uint8))
        else:
            mask_list.append((mask > 0).astype(np.uint8))
            
    # Calculate pixel-wise vote sum across all executed pipelines
    vote_accumulator = np.sum(np.dstack(mask_list), axis=2) # Shape (H, W) values 0..K
    
    # 2+ Pipeline Consensus binary mask
    consensus_binary = (vote_accumulator >= min_agreement).astype(np.uint8) * 255
    consensus_area = int(np.count_nonzero(consensus_binary))
    
    # Generate Consensus Visualization Overlay
    overlay_bgr = original_bgr.copy()
    
    # Multi-level color coding for consensus votes
    # 2 votes: Blue/Purple [255, 128, 0]
    # 3 votes: Cyan [255, 255, 0]
    # 4 votes: Emerald [0, 255, 128]
    color_map = {
        2: (255, 128, 0),   # Purple/Blue accent
        3: (255, 255, 0),   # Cyan accent
        4: (0, 255, 128)    # Emerald accent
    }
    
    # Color pixels by vote level
    color_layer = np.zeros_like(original_bgr)
    for vote_cnt in range(min_agreement, total_executed + 1):
        vote_mask = (vote_accumulator == vote_cnt)
        color = color_map.get(vote_cnt, (0, 255, 255))
        color_layer[vote_mask] = color
        
    # Blend overlay with original image
    blend_mask = vote_accumulator >= min_agreement
    overlay_bgr[blend_mask] = cv2.addWeighted(original_bgr[blend_mask], 0.45, color_layer[blend_mask], 0.55, 0)
    
    # Draw consensus boundary contours
    contours, _ = cv2.findContours(consensus_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 242, 254), 2)
    
    # Encode Base64
    _, mask_png = cv2.imencode('.png', consensus_binary)
    consensus_mask_b64 = "data:image/png;base64," + base64.b64encode(mask_png).decode('utf-8')
    
    _, overlay_png = cv2.imencode('.png', overlay_bgr)
    consensus_overlay_b64 = "data:image/png;base64," + base64.b64encode(overlay_png).decode('utf-8')
    
    # Find max agreement count among pixels in consensus region
    max_supported = int(np.max(vote_accumulator)) if consensus_area > 0 else 0
    agreement_summary = f"{max_supported} of {total_executed} executed pipelines support this region." if consensus_area > 0 else f"0 of {total_executed} executed pipelines support this region."
    
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "consensus_mask": consensus_mask_b64,
        "consensus_overlay": consensus_overlay_b64,
        "supporting_pipeline_count": total_executed,
        "consensus_area": consensus_area,
        "agreement_summary": agreement_summary,
        "processing_time": elapsed_ms
    }
