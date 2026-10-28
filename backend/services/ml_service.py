import os
import time
import json
from typing import Dict, Any, Union, Optional
import numpy as np
import cv2
from PIL import Image


class MLClassificationService:
    """
    Dedicated ML Inference Service for DefectVision.
    Loads the MobileNetV2 model and class mapping ONCE at application startup,
    keeping weights resident in memory to serve low-latency inference requests.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(backend_dir)

        self.model_path = model_path or os.path.join(
            project_root, "ml", "artifacts", "model", "defectvision_mobilenet_v2.keras"
        )
        self.config_path = config_path or os.path.join(
            project_root, "ml", "artifacts", "model", "config.json"
        )

        self.model = None
        self.config = {}
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.short_codes = {}
        self.target_size = (224, 224)
        self._is_loaded = False
        self._load_error = None

    def load_model(self):
        """
        Loads the model architecture, pretrained weights, and configuration into memory.
        Executed once during FastAPI application startup.
        """
        print(f"[ML Service] Initializing DefectVision classification service...")
        start_time = time.time()

        # 1. Load Configuration
        if not os.path.exists(self.config_path):
            self._load_error = f"Config file not found at: {self.config_path}"
            print(f"[ML Service ERROR] {self._load_error}")
            return False

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            self.classes = self.config.get("classes", [])
            self.class_to_idx = self.config.get("class_to_idx", {})
            self.idx_to_class = {int(k): v for k, v in self.config.get("idx_to_class", {}).items()}
            self.short_codes = self.config.get("class_short_names", {})
            h = self.config.get("input_dimensions", {}).get("height", 224)
            w = self.config.get("input_dimensions", {}).get("width", 224)
            self.target_size = (w, h)
        except Exception as e:
            self._load_error = f"Failed to parse config.json: {str(e)}"
            print(f"[ML Service ERROR] {self._load_error}")
            return False

        # 2. Load Keras Model
        if not os.path.exists(self.model_path):
            self._load_error = f"Model checkpoint not found at: {self.model_path}"
            print(f"[ML Service ERROR] {self._load_error}")
            return False

        try:
            import tensorflow as tf
            # Load model into memory without compiling (compile=False optimizes memory for inference)
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
            # Warm-up forward pass with a dummy tensor to initialize execution graph/allocations
            dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            
            self._is_loaded = True
            elapsed = (time.time() - start_time) * 1000
            print(f"[ML Service] MobileNetV2 model loaded successfully in {elapsed:.1f}ms. Target classes: {self.classes}")
            return True
        except Exception as e:
            self._load_error = f"Failed to load Keras model: {str(e)}"
            print(f"[ML Service ERROR] {self._load_error}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._is_loaded and self.model is not None

    def get_status(self) -> Dict[str, Any]:
        """Returns service status for healthcheck endpoint."""
        return {
            "name": "MobileNetV2 Defect Classifier",
            "status": "ready" if self.is_ready else "offline",
            "model_path": os.path.basename(self.model_path),
            "classes": self.classes,
            "error": self._load_error if not self.is_ready else None,
        }

    def preprocess(self, image_input: Union[np.ndarray, bytes, str]) -> np.ndarray:
        """
        Preprocesses OpenCV BGR array, raw image bytes, or file path to MobileNetV2 input:
          1. Ensures 3-channel RGB matrix
          2. Bilinear resize to 224 x 224
          3. Min-Max normalization to [-1.0, 1.0]: (x / 127.5) - 1.0
          4. Expands to batch tensor: (1, 224, 224, 3)
        """
        # A. If OpenCV BGR / Grayscale numpy array (from preprocess_image in main.py)
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 3:
                rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                rgb_arr = image_input
            resized = cv2.resize(rgb_arr, self.target_size, interpolation=cv2.INTER_LINEAR)
        # B. If raw bytes
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_arr, self.target_size, interpolation=cv2.INTER_LINEAR)
        # C. If file path
        elif isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            rgb_arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_arr, self.target_size, interpolation=cv2.INTER_LINEAR)
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        # Normalize to [-1.0, 1.0]
        float_img = resized.astype(np.float32)
        normalized = (float_img / 127.5) - 1.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, image_input: Union[np.ndarray, bytes, str]) -> Dict[str, Any]:
        """
        Runs forward pass on resident model in memory and formats response.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"ML Classification Service is not ready. Error: {self._load_error or 'Model not loaded'}"
            )

        start_time = time.time()
        batch_input = self.preprocess(image_input)

        # Forward pass on resident model
        probs_batch = self.model.predict(batch_input, verbose=0)
        probs = probs_batch[0]

        top_idx = int(np.argmax(probs))
        predicted_class = self.idx_to_class.get(top_idx, self.classes[top_idx])
        confidence = float(probs[top_idx])

        # Softmax validation
        assert 0.0 <= confidence <= 1.0001, f"Invalid confidence: {confidence}"

        probabilities = {
            self.idx_to_class.get(i, self.classes[i]): round(float(probs[i]), 5)
            for i in range(len(probs))
        }

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "pipeline": "mobilenet_v2",
            "predicted_class": predicted_class,
            "short_code": self.short_codes.get(predicted_class, ""),
            "confidence": round(confidence, 5),
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "probabilities": probabilities,
            "processing_time": elapsed_ms,
        }


# Global service instance
ml_service = MLClassificationService()
