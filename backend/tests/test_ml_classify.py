import os
import sys
import io
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from main import app
from services.ml_service import ml_service





def test_classify_valid_image(client):
    """
    Test 1: Valid image upload returns 200, predicted class, scores for all 6 classes,
    and valid runtime latency.
    """
    sample_path = os.path.join(os.path.dirname(backend_dir), "frontend", "public", "sample_defect.png")
    assert os.path.exists(sample_path), f"Sample image not found at: {sample_path}"

    with open(sample_path, "rb") as f:
        response = client.post(
            "/api/ml/classify",
            files={"file": ("sample_defect.png", f, "image/png")}
        )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["filename"] == "sample_defect.png"
    assert "predicted_class" in data
    assert "short_code" in data
    assert "confidence" in data
    assert "confidence_percentage" in data
    assert "scores" in data
    assert "inference_runtime_ms" in data

    # Verify all 6 NEU defect classes exist in scores
    expected_classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    for cls in expected_classes:
        assert cls in data["scores"]

    # Verify probability distribution sums to 1.0
    scores_sum = sum(data["scores"].values())
    assert abs(scores_sum - 1.0) < 0.01

    # Verify runtime is positive
    assert data["inference_runtime_ms"] > 0


def test_classify_invalid_file_type(client):
    """
    Test 2: Non-image file type (e.g., .txt) is rejected with HTTP 400 Bad Request.
    """
    fake_txt = io.BytesIO(b"This is not a defect image.")
    response = client.post(
        "/api/ml/classify",
        files={"file": ("notes.txt", fake_txt, "text/plain")}
    )

    assert response.status_code == 400
    data = response.json()
    assert "Invalid file type" in data["detail"]


def test_classify_corrupted_image(client):
    """
    Test 3: Corrupted binary bytes with valid image extension returns HTTP 400.
    """
    corrupt_bytes = io.BytesIO(b"NOT_A_REAL_PNG_HEADER_GARBAGE_BYTES")
    response = client.post(
        "/api/ml/classify",
        files={"file": ("defect.png", corrupt_bytes, "image/png")}
    )

    assert response.status_code == 400
    data = response.json()
    assert "Could not decode image bytes" in data["detail"]


def test_classify_empty_image(client):
    """
    Test 4: Empty 0-byte file returns HTTP 400.
    """
    empty_bytes = io.BytesIO(b"")
    response = client.post(
        "/api/ml/classify",
        files={"file": ("empty.jpg", empty_bytes, "image/jpeg")}
    )

    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"].lower()


def test_missing_model_handling(client):
    """
    Test 5: If ML service is marked offline/unavailable, endpoint gracefully returns HTTP 503.
    """
    # Temporarily simulate unready state
    original_state = ml_service._is_loaded
    try:
        ml_service._is_loaded = False
        ml_service._load_error = "Simulated missing model file"

        sample_path = os.path.join(os.path.dirname(backend_dir), "frontend", "public", "sample_defect.png")
        with open(sample_path, "rb") as f:
            response = client.post(
                "/api/ml/classify",
                files={"file": ("sample.png", f, "image/png")}
            )

        assert response.status_code == 503
        data = response.json()
        assert "Service is unavailable" in data["detail"]
    finally:
        # Restore actual state
        ml_service._is_loaded = original_state
        ml_service._load_error = None


def test_classical_pipelines_not_executed(client):
    """
    Test 6: Verify that calling /api/ml/classify NEVER triggers any of the
    four classical OpenCV pipelines or consensus engine.
    """
    from unittest.mock import patch
    import main as main_module

    sample_path = os.path.join(os.path.dirname(backend_dir), "frontend", "public", "sample_defect.png")
    
    with patch.object(main_module, 'run_spatial_pipeline') as mock_spatial, \
         patch.object(main_module, 'run_edge_shape_pipeline') as mock_edge, \
         patch.object(main_module, 'run_frequency_pipeline') as mock_freq, \
         patch.object(main_module, 'run_texture_pipeline') as mock_texture:

        with open(sample_path, "rb") as f:
            response = client.post(
                "/api/ml/classify",
                files={"file": ("sample.png", f, "image/png")}
            )

        assert response.status_code == 200
        # Assert none of the 4 classical pipelines were invoked
        mock_spatial.assert_not_called()
        mock_edge.assert_not_called()
        mock_freq.assert_not_called()
        mock_texture.assert_not_called()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("      RUNNING FASTAPI POST /api/ml/classify TEST SUITE")
    print("=" * 70)
    
    with TestClient(app) as test_client:
        tests = [
            ("Test 1: Valid Image Classification (200 OK)", test_classify_valid_image),
            ("Test 2: Invalid File Type Handling (400 Bad Request)", test_classify_invalid_file_type),
            ("Test 3: Corrupted Image Bytes Handling (400 Bad Request)", test_classify_corrupted_image),
            ("Test 4: Empty File Handling (400 Bad Request)", test_classify_empty_image),
            ("Test 5: Missing Model Artifact Handling (503 Service Unavailable)", test_missing_model_handling),
            ("Test 6: Strict Isolation from Classical Pipelines", test_classical_pipelines_not_executed),
        ]

        passed = 0
        for name, test_fn in tests:
            try:
                test_fn(test_client)
                print(f"[PASS] {name}")
                passed += 1
            except Exception as e:
                print(f"[FAIL] {name}: {e}")

        print("-" * 70)
        print(f"Test Summary: {passed}/{len(tests)} tests passed successfully.")
        print("=" * 70)
        if passed != len(tests):
            sys.exit(1)

