import os
import sys
import io
import json
import base64
from fastapi.testclient import TestClient

# Add backend directory to sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from main import app
from services.ml_service import ml_service

SAMPLE_PATH = os.path.join(os.path.dirname(backend_dir), "frontend", "public", "sample_defect.png")

def run_e2e_tests():
    print("=" * 80)
    print("   DEFECTVISION COMPLETE SYSTEM VERIFICATION SUITE")
    print("=" * 80)

    assert os.path.exists(SAMPLE_PATH), f"Sample image missing at: {SAMPLE_PATH}"
    with open(SAMPLE_PATH, "rb") as f:
        sample_bytes = f.read()

    passed = 0
    total = 0

    def check(name, condition, details=""):
        nonlocal passed, total
        total += 1
        if condition:
            print(f" [PASS] {name} {details}")
            passed += 1
        else:
            print(f" [FAIL] {name} - FAILED {details}")
            raise AssertionError(f"Test failed: {name} - {details}")

    with TestClient(app) as client:
        # ---------------------------------------------------------
        # 1. Health & Model Loading Verification
        # ---------------------------------------------------------
        print("\n--- 1. Health Check & Model Startup Status ---")
        res = client.get("/api/health")
        check("Health Endpoint HTTP 200", res.status_code == 200)
        hdata = res.json()
        check("Health Status Healthy", hdata.get("status") == "healthy")
        check("Spatial Stub Present", "spatial" in hdata["pipelines"])
        check("Edge/Shape Stub Present", "edge_shape" in hdata["pipelines"])
        check("Frequency Stub Present", "frequency" in hdata["pipelines"])
        check("Texture Stub Present", "texture" in hdata["pipelines"])
        check("Classification Service Online", hdata["pipelines"]["classification"]["status"] == "ready")

        # ---------------------------------------------------------
        # 2. Four Independent OpenCV Pipelines Verification
        # ---------------------------------------------------------
        print("\n--- 2. Independent OpenCV Classical Pipelines ---")
        
        # Spatial Pipeline
        res_spatial = client.post("/api/pipelines/spatial", files={"file": ("sample.png", sample_bytes, "image/png")})
        check("Spatial Pipeline HTTP 200", res_spatial.status_code == 200)
        spatial_data = res_spatial.json()["result"]
        check("Spatial Mask Produced", spatial_data["final_mask"].startswith("data:image/png;base64,"))
        check("Spatial Anomaly Area Present", "anomaly_area" in spatial_data)

        # Edge/Shape Pipeline
        res_edge = client.post("/api/pipelines/edge-shape", files={"file": ("sample.png", sample_bytes, "image/png")})
        check("Edge/Shape Pipeline HTTP 200", res_edge.status_code == 200)
        edge_data = res_edge.json()["result"]
        check("Edge Map Produced", edge_data["edge_map"].startswith("data:image/png;base64,"))
        check("Shape Features Extracted", isinstance(edge_data.get("shape_features"), list))

        # Frequency (FFT) Pipeline
        res_freq = client.post("/api/pipelines/frequency", files={"file": ("sample.png", sample_bytes, "image/png")})
        check("Frequency Pipeline HTTP 200", res_freq.status_code == 200)
        freq_data = res_freq.json()["result"]
        check("FFT Spectrum Produced", freq_data["fft_spectrum"].startswith("data:image/png;base64,"))
        check("High/Low Ratio Computed", freq_data.get("high_to_low_frequency_ratio") is not None)

        # Texture (Gabor) Pipeline
        res_tex = client.post("/api/pipelines/texture", files={"file": ("sample.png", sample_bytes, "image/png")})
        check("Texture Pipeline HTTP 200", res_tex.status_code == 200)
        tex_data = res_tex.json()["result"]
        check("Gabor 4-Orientation Responses", all(k in tex_data for k in ["gabor_0_response", "gabor_45_response", "gabor_90_response", "gabor_135_response"]))
        check("Texture Statistics Computed", "entropy" in tex_data.get("texture_statistics", {}))

        # ---------------------------------------------------------
        # 3. Consensus Pipeline Verification
        # ---------------------------------------------------------
        print("\n--- 3. 2+ Pipeline Voting Consensus Engine ---")
        consensus_payload = {
            "spatial_mask": spatial_data["final_mask"],
            "edge_shape_mask": edge_data["anomaly_mask"],
            "frequency_mask": freq_data["anomaly_mask"],
            "texture_mask": tex_data["anomaly_mask"]
        }
        res_cons = client.post(
            "/api/pipelines/consensus",
            files={"file": ("sample.png", sample_bytes, "image/png")},
            data=consensus_payload
        )
        check("Consensus HTTP 200", res_cons.status_code == 200)
        cons_data = res_cons.json()["result"]
        check("Consensus Mask Produced", cons_data["consensus_mask"].startswith("data:image/png;base64,"))
        check("Consensus Agreement Summary", "supporting_pipeline_count" in cons_data and cons_data["supporting_pipeline_count"] >= 2)

        # Consensus failure with < 2 masks
        res_cons_fail = client.post(
            "/api/pipelines/consensus",
            files={"file": ("sample.png", sample_bytes, "image/png")},
            data={"spatial_mask": spatial_data["final_mask"]}
        )
        check("Consensus <2 Masks Rejected (HTTP 400)", res_cons_fail.status_code == 400)

        # ---------------------------------------------------------
        # 4. MobileNetV2 Deep Learning Classification Verification
        # ---------------------------------------------------------
        print("\n--- 4. MobileNetV2 Classification Endpoint ---")
        res_ml = client.post("/api/ml/classify", files={"file": ("sample.png", sample_bytes, "image/png")})
        check("ML Endpoint HTTP 200", res_ml.status_code == 200)
        ml_data = res_ml.json()
        check("Valid Predicted Class", ml_data.get("predicted_class") in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"])
        check("Valid Class Short Code", ml_data.get("short_code") in ["Cr", "In", "Pa", "PS", "RS", "Sc"])
        check("Valid Confidence Percentage", "%" in ml_data.get("confidence_percentage", ""))
        check("All 6 Class Scores Present", len(ml_data.get("scores", {})) == 6)
        check("Softmax Scores Sum ~1.0", abs(sum(ml_data["scores"].values()) - 1.0) < 0.01)
        check("Real Inference Runtime (>0 ms)", ml_data.get("inference_runtime_ms", 0) > 0)

        # ---------------------------------------------------------
        # 5. Robust Error Handling Verification
        # ---------------------------------------------------------
        print("\n--- 5. Edge Cases & Error Handling ---")
        
        # Invalid file type (.txt)
        res_bad_type = client.post("/api/ml/classify", files={"file": ("test.txt", b"plain text", "text/plain")})
        check("Invalid File Type Rejected (HTTP 400)", res_bad_type.status_code == 400)

        # Corrupted image bytes
        res_corrupt = client.post("/api/ml/classify", files={"file": ("corrupt.png", b"CORRUPTED_PNG_BYTES_1234", "image/png")})
        check("Corrupted Image Rejected (HTTP 400)", res_corrupt.status_code == 400)

        # Empty 0-byte file
        res_empty = client.post("/api/ml/classify", files={"file": ("empty.jpg", b"", "image/jpeg")})
        check("Empty File Rejected (HTTP 400)", res_empty.status_code == 400)

        # Missing model handling (simulated unready state)
        orig_loaded = ml_service._is_loaded
        orig_err = ml_service._load_error
        try:
            ml_service._is_loaded = False
            ml_service._load_error = "Model weights artifact unreadable"
            res_missing_model = client.post("/api/ml/classify", files={"file": ("sample.png", sample_bytes, "image/png")})
            check("Missing Model Artifact Handled (HTTP 503)", res_missing_model.status_code == 503)
        finally:
            ml_service._is_loaded = orig_loaded
            ml_service._load_error = orig_err

        # ---------------------------------------------------------
        # 6. Pipeline Isolation Verification
        # ---------------------------------------------------------
        print("\n--- 6. Pipeline Isolation Verification ---")
        from unittest.mock import patch
        import main as main_module

        with patch.object(main_module, 'run_spatial_pipeline') as m_spatial, \
             patch.object(main_module, 'run_edge_shape_pipeline') as m_edge, \
             patch.object(main_module, 'run_frequency_pipeline') as m_freq, \
             patch.object(main_module, 'run_texture_pipeline') as m_texture:
            
            res_iso = client.post("/api/ml/classify", files={"file": ("sample.png", sample_bytes, "image/png")})
            check("Classify Under Isolation HTTP 200", res_iso.status_code == 200)
            check("Spatial Pipeline Not Called", not m_spatial.called)
            check("Edge Pipeline Not Called", not m_edge.called)
            check("Frequency Pipeline Not Called", not m_freq.called)
            check("Texture Pipeline Not Called", not m_texture.called)

    print("\n" + "=" * 80)
    print(f"   VERIFICATION COMPLETE: {passed}/{total} tests passed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_e2e_tests()
