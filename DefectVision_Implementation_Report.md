# DefectVision: Comprehensive Engineering & Implementation Report

---

## 1. Executive Summary & Project Identification

**DefectVision** is an industrial-grade, multi-paradigm computer vision and deep learning system engineered for the automated detection, localized segmentation, and taxonomic classification of surface defects on manufactured materials (specifically hot-rolled steel plates from the benchmark **NEU Surface Defect Database**).

The system integrates **classical digital image processing (DIP)** with **modern deep convolutional neural networks (CNNs)** into a unified, high-performance web-accessible platform. Rather than relying on a single detection heuristic or treating deep learning as a monolithic black box, DefectVision provides a **hybrid inspection framework**:
1. **Four Independent Classical Vision Pipelines** isolating distinct physical defect signatures: intensity anomalies, structural edges, spectral frequencies, and directional textures.
2. **A Pixel-Wise Consensus & Voting Engine** that fuses classical anomaly masks to eliminate single-pipeline false positives and generate statistically grounded defect boundaries.
3. **A Decoupled MobileNetV2 Deep Learning Classifier** trained to categorize surface anomalies into six industrial defect categories with full softmax probability distributions.
4. **An Asynchronous REST Backend (FastAPI)** with resident model in-memory lifecycle caching.
5. **An Interactive Web Dashboard (React 19 + TypeScript + Vite)** delivering real-time pipeline execution, parameter inspection, and side-by-side visual comparisons.

---

## 2. Problem Statement & Industry Motivation

### The Industrial Inspection Challenge
In steel manufacturing and metallurgy, surface defects such as cracks (*crazing*), slag entrapment (*inclusions*), surface depressions (*pitted surfaces*), oxidized scale pressed into the plate (*rolled-in scale*), local abrasions (*scratches*), and discolored areas (*patches*) severely compromise tensile strength, fatigue resistance, and corrosion immunity.

Traditionally, surface quality assurance suffers from three critical bottlenecks:
* **Manual Human Inspection Fatigue**: Human inspectors operating along high-speed rolling lines cannot sustain continuous high-accuracy defect capture due to eye fatigue, subjective judgment, and rapid conveyor speeds.
* **Failure Modes of Single-Paradigm Vision Algorithms**:
  * Pure thresholding fails when illumination is non-uniform or surface oil creates false intensity gradients.
  * Edge detectors fail on faint, wide-area discolored patches.
  * Frequency-domain filters struggle with isolated localized abrasions.
  * Texture filters are computationally heavy and may misinterpret uniform rolling striations as flaws.
* **The "Black Box" Interpretability Gap of Pure Deep Learning**: Pure deep learning classifiers provide a class label and confidence score, but often lack interpretable spatial localization or verifiable mathematical metrics (such as defect area, aspect ratio, circularity, spectral energy ratio, and directional filter responses) required for root-cause manufacturing diagnostics.

### How DefectVision Solves This Problem
DefectVision addresses these challenges through a **complementary dual-layer architecture**:
* **Granular Classical Isolation**: Evaluates raw steel imagery across 4 distinct mathematical domains (Spatial, Edge/Geometry, Frequency, Texture) simultaneously.
* **High-Confidence Localization via Voting Consensus**: Requires at least $K \ge 2$ independent analytical algorithms to agree on a defective pixel before confirming an anomaly region, minimizing false alarms.
* **Fast, Mobile-Optimized Taxonomic Classification**: Employs transfer learning with MobileNetV2 to deliver low-latency classification across the 6 canonical defect classes.
* **Full Diagnostic Transparency**: Yields intermediate filter maps, binary segmentation masks, contour overlays, geometric properties, spectral ratios, and texture statistics alongside deep learning predictions.

---

## 3. High-Level System Architecture & Workflow

```text
                                  [ Input Image (JPG/PNG/BMP) ]
                                                │
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │          Common Preprocessing Engine                │
                     │  - Grayscale conversion                             │
                     │  - Min-Max contrast normalization [0, 255]          │
                     │  - 3x3 Gaussian noise reduction                     │
                     └───────────────┬─────────────────────┬───────────────┘
                                     │                     │
                     ┌───────────────┘                     └───────────────┐
                     ▼                                                     ▼
    ┌───────────────────────────────────┐               ┌──────────────────────────────────────┐
    │ Classical Multi-Domain Pipelines   │               │ Deep Learning Classification Engine  │
    ├───────────────────────────────────┤               ├──────────────────────────────────────┤
    │ 1. Spatial Pipeline:              │               │ Architecture: MobileNetV2            │
    │    CLAHE + Otsu + Morph Opening   │               │ Preprocessing:                       │
    │ 2. Edge / Shape Pipeline:         │               │  - RGB 3-channel conversion          │
    │    Canny + Otsu + Contour Metrics │               │  - Bilinear Resize: 224 x 224        │
    │ 3. Frequency Pipeline (2D-FFT):   │               │  - Scale: [-1.0, 1.0]                │
    │    FFT2 + High-Pass Gaussian + IFFT│              │ Inference:                           │
    │ 4. Texture Pipeline (Gabor):      │               │  - Resident weights in memory        │
    │    4 Orientations x 3 Scales      │               │  - 6-Class Softmax Output            │
    └─────────────────┬─────────────────┘               └──────────────────┬───────────────────┘
                      │                                                    │
                      ▼                                                    │
    ┌───────────────────────────────────┐                                  │
    │ Pixel-Wise Consensus Engine       │                                  │
    │ (Voting Threshold: K >= 2 masks)  │                                  │
    └─────────────────┬─────────────────┘                                  │
                      │                                                    │
                      └──────────────────────┬─────────────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │  FastAPI Backend (8000)   │
                               │  - JSON response payloads │
                               │  - Base64 PNG visuals     │
                               └─────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │  React 19 / Vite Frontend │
                               │  - Interactive Dashboard  │
                               │  - Inspection Comparators │
                               │  - Probabilities & Stats  │
                               └───────────────────────────┘
```

---

## 4. Detailed Component-by-Component Deep Dive

### 4.1. Image Preprocessing Subsystem
* **Source Implementation**: `backend/preprocessing/preprocess.py`
* **API Route**: `POST /api/preprocess`

Every image entering DefectVision undergoes deterministic standardization:
1. **Byte Stream Decoding**: Validates image format (PNG, JPEG, BMP) and decodes memory buffers into an 8-bit OpenCV BGR array (`cv2.imdecode`).
2. **Grayscale Reduction**: Converts 3-channel BGR images to single-channel 8-bit luminance using standard ITU-R 601-2 luma weights:
   $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
3. **Contrast Normalization**: Performs Min-Max dynamic range stretching (`cv2.normalize` with `NORM_MINMAX`) over the full dynamic range $[0, 255]$:
   $$I_{\text{norm}}(x, y) = \frac{I(x, y) - \min(I)}{\max(I) - \min(I)} \times 255$$
4. **Gaussian Denoising**: Smooths high-frequency sensor noise using a $3 \times 3$ kernel with standard deviation $\sigma = 0$, suppressing pixel-level acquisition artifacts while preserving macro boundary gradients.
5. **Base64 Serialization**: Generates standardized data URIs (`data:image/png;base64,...`) for web transmission.

---

### 4.2. Classical Pipeline 1: Spatial & Intensity Domain Analysis
* **Source Implementation**: `backend/pipelines/spatial.py`
* **API Route**: `POST /api/pipelines/spatial`
* **Target Defects**: Discolored surface areas, dark inclusions, scale patches, and regional reflectivity shifts.

```text
Grayscale ──> CLAHE (clip=2.5, grid=8x8) ──> Gaussian Blur (5x5) ──> Otsu Inv Threshold ──> Morph Opening/Closing (5x5) ──> Connected Components (Area >= 15px) ──> Anomaly Mask & Red Overlay
```

#### Algorithmic Steps:
1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**:
   * Standard histogram equalization over-amplifies noise in homogeneous steel backgrounds.
   * DefectVision applies CLAHE with `clipLimit=2.5` over an $8 \times 8$ contextual tile grid, redistributing local histogram bins above the clip limit uniformly across tiles and interpolating bilinear boundaries.
2. **Adaptive Otsu Binarization**:
   * Applies Gaussian smoothing ($5 \times 5$, $\sigma=0$) followed by inverted Otsu thresholding (`THRESH_BINARY_INV + THRESH_OTSU`). Otsu automatically computes the optimal global threshold $t^*$ maximizing inter-class variance:
     $$\sigma_B^2(t) = \omega_0(t)\omega_1(t)[\mu_0(t) - \mu_1(t)]^2$$
3. **Morphological Filtering**:
   * An elliptical structuring element ($5 \times 5$) performs morphological **Opening** ($\text{Erosion} \to \text{Dilation}$) to eliminate isolated pixel salt noise, followed by **Closing** ($\text{Dilation} \to \text{Erosion}$) to bridge narrow fissures in fragmented defects.
4. **Connected Components & Size Gating**:
   * Evaluates 8-connectivity labeling (`cv2.connectedComponentsWithStats`). Regions smaller than a 15-pixel threshold ($A < 15\text{ px}$) are filtered out as optical noise.
5. **Output Metrics**:
   * Total anomaly area in pixels, anomaly area percentage ($\frac{A_{\text{defect}}}{W \times H} \times 100\%$), discrete region count, and a weighted red alpha overlay ($\alpha=0.7, \beta=0.3$).

---

### 4.3. Classical Pipeline 2: Edge & Geometric Shape Analysis
* **Source Implementation**: `backend/pipelines/edge_shape.py`
* **API Route**: `POST /api/pipelines/edge-shape`
* **Target Defects**: Scratches, cracks, severe fissures, and sharp structural boundary faults.

```text
Grayscale ──> Gaussian Smoothing (5x5, σ=1.4) ──> Otsu-Guided Canny ──> Contour Extraction ──> Morphometric Analysis (Area, Perimeter, Aspect Ratio, Circularity) ──> Cyan/Yellow Vector Overlay
```

#### Algorithmic Steps:
1. **Gaussian Pre-Filtering**: Applies Gaussian filtering with a kernel size of $5 \times 5$ and $\sigma = 1.4$ to suppress high-frequency micro-textures.
2. **Otsu-Guided Canny Edge Detector**:
   * Computes an intensity histogram threshold $T_{\text{otsu}}$ on the smoothed image.
   * Sets dynamic hysteresis thresholds: High threshold $T_{\text{high}} = $T_{\text{otsu}}$, Low threshold $T_{\text{low}} = 0.5 \cdot T_{\text{high}}$.
   * Evaluates Sobel directional gradient magnitudes $G = \sqrt{G_x^2 + G_y^2}$ and angles $\theta = \arctan(G_y / G_x)$, performs non-maximum suppression along gradient normals, and applies hysteresis tracking.
3. **Contour Extraction & Feature Engineering**:
   * Extracts external topological contours (`cv2.findContours` with `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`).
   * For every candidate contour with area $A \ge 15.0\text{ px}^2$, DefectVision computes rigorous geometric morphometrics:
     * **Area ($A$)**: Green's Theorem contour integration (`cv2.contourArea`).
     * **Perimeter ($P$)**: Arc length computation (`cv2.arcLength(c, True)`).
     * **Bounding Rect ($x, y, w, h$)**: Minimal axis-aligned bounding box.
     * **Aspect Ratio**: $AR = \frac{w}{h}$ (distinguishes longitudinal scratches from circular pits).
     * **Circularity / Compactness**:
       $$C = \frac{4 \pi \cdot A}{P^2}$$
       ($C \approx 1.0$ indicates circular pits/inclusions, while $C \ll 0.2$ indicates elongated linear scratches).
4. **Output Visualizations**:
   * Binary edge map, vector bounding box visualization (cyan contours, yellow bounding boxes), and filled anomaly mask.

---

### 4.4. Classical Pipeline 3: Frequency Domain (2D-FFT) Analysis
* **Source Implementation**: `backend/pipelines/frequency.py`
* **API Route**: `POST /api/pipelines/frequency`
* **Target Defects**: Periodic roller marks, repetitive striations, micro-crazing networks, and surface roughness deviations.

```text
Grayscale ──> 2D-FFT (np.fft.fft2) ──> FFT Shift (DC to Center) ──> Log Magnitude Spectrum ──> Energy Partitioning ──> Gaussian High-Pass Filter ──> Inverse 2D-FFT ──> Otsu Threshold ──> Magenta Overlay
```

#### Algorithmic Steps:
1. **Discrete 2D Fourier Transformation**:
   * Maps spatial luminance $f(x, y)$ of size $M \times N$ to complex frequency space $F(u, v)$:
     $$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
2. **Frequency Centering & Magnitude Spectrum**:
   * Quadrant shifting (`np.fft.fftshift`) centers zero-frequency DC components $(u=0, v=0)$ at $\left(\frac{M}{2}, \frac{N}{2}\right)$.
   * Computes dynamic range-compressed logarithmic magnitude spectra:
     $$S(u, v) = 20 \cdot \log(|F_{\text{shift}}(u, v)| + 1.0)$$
3. **Spectral Energy Partitioning**:
   * Establishes a radial low-frequency cutoff radius $R_{\text{low}} = 0.12 \cdot \min(M, N)$.
   * Defines Euclidean distance from center: $D(u, v) = \sqrt{(u - u_c)^2 + (v - v_c)^2}$.
   * Calculates Total Spectral Energy, Low-Frequency Energy ($D \le R_{\text{low}}$), and High-Frequency Energy ($D > R_{\text{low}}$).
   * Computes the **High-to-Low Energy Ratio ($R_{H/L}$)**:
     $$R_{H/L} = \frac{E_{\text{high}}}{E_{\text{low}} + 10^{-8}}$$
     (Higher ratios indicate dense structural defects such as crazing or fine scratches).
4. **Gaussian High-Pass Filtering (GHPF)**:
   * Smoothly attenuates dominant low-frequency background rolling trends:
     $$H(u, v) = 1.0 - \exp\left(-\frac{D^2(u, v)}{2 R_{\text{low}}^2}\right)$$
5. **Spatial Reconstruction & Anomaly Extraction**:
   * Evaluates Inverse FFT: $f_{\text{reconstructed}} = |\mathcal{F}^{-1}\{\text{ifftshift}(F_{\text{shift}} \cdot H)\}|$.
   * Normalizes to $[0, 255]$, applies Otsu thresholding, filters connected components $\ge 15\text{ px}$, and renders a magenta alpha overlay.

---

### 4.5. Classical Pipeline 4: Texture Domain (Gabor Filter Bank) Analysis
* **Source Implementation**: `backend/pipelines/texture.py`
* **API Route**: `POST /api/pipelines/texture`
* **Target Defects**: Irregular rolled-in scale, inclusions, pitting, and non-directional surface disturbances.

```text
Grayscale ──> Gabor Filter Bank (4 Orientations: 0°, 45°, 90°, 135° x 3 Wavelengths: 4, 8, 12) ──> Scale Max-Pooling ──> Response Fusion ──> Shannon Entropy & GLCM Stats ──> Otsu Segmentation ──> Emerald Overlay
```

#### Algorithmic Steps:
1. **Filter Bank Parameterization**:
   * DefectVision constructs a multi-scale, multi-directional 2D Gabor filter bank:
     * **Orientations ($\theta$)**: $0^\circ, 45^\circ, 90^\circ, 135^\circ$ ($\theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}\}$).
     * **Wavelengths ($\lambda$)**: $4.0, 8.0, 12.0\text{ pixels}$.
     * **Gaussian Envelope Standard Deviation ($\sigma$)**: $3.0$.
     * **Spatial Aspect Ratio ($\gamma$)**: $0.5$.
     * **Phase Offset ($\psi$)**: $0$.
     * **Kernel Size**: $21 \times 21\text{ pixels}$.
   * Continuous spatial representation:
     $$g(x, y; \lambda, \theta, \psi, \sigma, \gamma) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi \frac{x'}{\lambda} + \psi\right)$$
     where $x' = x\cos\theta + y\sin\theta$ and $y' = -x\sin\theta + y\cos\theta$.
2. **Response Fusion**:
   * Convolves grayscale input with all 12 kernels.
   * For each orientation $\theta$, performs maximum response pooling across the 3 scales.
   * Computes an omnidirectional combined response map:
     $$R_{\text{combined}}(x, y) = \max_{\theta} R_{\theta}(x, y)$$
3. **Statistical Texture Characterization**:
   * Computes 5 descriptive statistical metrics:
     * **Mean Response**: $\mu_R = \frac{1}{MN} \sum R(x, y)$
     * **Variance**: $\sigma_R^2 = \frac{1}{MN} \sum (R(x, y) - \mu_R)^2$
     * **Energy**: $E = \frac{1}{MN} \sum R(x, y)^2$
     * **Contrast**: Standard deviation of response magnitudes.
     * **Shannon Entropy**:
       $$H = -\sum_{i=1}^{32} P(i) \log_2(P(i))$$
       derived from 32-bin normalized response histograms. High entropy indicates complex, irregular defect textures.
4. **Segmentation & Overlay**:
   * Otsu thresholding on $R_{\text{combined}}$, morphological cleaning, and emerald green overlay ($\text{BGR: } [0, 255, 128]$).

---

### 4.6. Multi-Pipeline Consensus & Fusion Engine
* **Source Implementation**: `backend/pipelines/consensus.py`
* **API Route**: `POST /api/pipelines/consensus`

#### Motivation & Mathematical Voting Logic
Individual computer vision pipelines exhibit specific sensitivities and noise vulnerabilities:
* Spatial thresholding can flag harmless illumination variations.
* Edge detection can pick up benign manufacturing grain boundaries.
* Frequency filtering can respond to plate edges.

To overcome these limitations, DefectVision implements an ensemble voting accumulator.

```text
[ Spatial Mask ]      ──┐
[ Edge/Shape Mask ]   ──┼──> Stacking & Alignment (H x W x K) ──> Pixel-Wise Sum (0..K) ──> Voting Filter (Sum >= 2) ──> Multi-Tier Color Overlay
[ Frequency Mask ]    ──┤
[ Texture Mask ]      ──┘
```

#### Consensus Algorithm:
1. **Input Alignment**: Receives $K$ binary segmentation masks ($2 \le K \le 4$) from the executed classical pipelines. Resizes masks to the source image dimensions $(H, W)$ using nearest-neighbor interpolation if dimensions diverge.
2. **Tensor Stacking & Pixel Voting Accumulator**:
   * Converts masks to boolean indicators $M_k(x, y) \in \{0, 1\}$.
   * Computes the discrete pixel vote sum across the stack:
     $$V(x, y) = \sum_{k=1}^{K} M_k(x, y), \quad V(x, y) \in [0, K]$$
3. **Majority Decision Rule**:
   * Generates the binary consensus defect mask $C(x, y)$ using a parameterized threshold (default $K_{\min} = 2$):
     $$C(x, y) = \begin{cases} 255, & \text{if } V(x, y) \ge K_{\min} \\ 0, & \text{otherwise} \end{cases}$$
4. **Multi-Level Confidence Color Coding**:
   * Pixels meeting the consensus threshold are rendered on the output visualization with distinct colors reflecting agreement depth:
     * **2 Supporting Pipelines (Moderate Confidence)**: Violet / Royal Blue (`RGB: #0080FF`)
     * **3 Supporting Pipelines (High Confidence)**: Cyan (`RGB: #00FFFF`)
     * **4 Supporting Pipelines (Unanimous Confidence)**: Emerald Green (`RGB: #00FF80`)
5. **Output**: Consensus area, agreement summary string (e.g., *"3 of 4 executed pipelines support this region"*), processing time, and base64-encoded visual overlays.

---

### 4.7. Deep Learning Subsystem (MobileNetV2 Classifier)
* **Source Implementations**:
  * Architecture & Transfer Learning: `ml/models/mobilenet_tf.py`
  * Artifact Export & Config: `ml/models/save_artifacts.py`
  * Model Configuration JSON: `ml/artifacts/model/config.json`
  * Standalone ML Inference Engine: `ml/inference.py`
  * Two-Stage Training Routines: `ml/training/train_stage1_tf.py`, `ml/training/train_finetune_tf.py`
  * Backend ML Microservice: `backend/services/ml_service.py`
* **API Route**: `POST /api/ml/classify` (and alias `POST /api/pipelines/classify`)

```text
Input Image (OpenCV BGR / Bytes)
       │
       ▼
[ Preprocessing Layer ]
  ├── Color Conversion: BGR -> RGB (3 channels)
  ├── Bilinear Resample: 224 x 224 pixels
  ├── Min-Max Scaling: (pixel / 127.5) - 1.0 -> Range [-1.0, 1.0]
  └── Batch Expansion: (1, 224, 224, 3)
       │
       ▼
[ Pretrained MobileNetV2 Backbone ]
  ├── 53 Inverted Residual Blocks (Depthwise Separable Convolutions)
  └── Final Feature Tensor: (1, 7, 7, 1280)
       │
       ▼
[ GlobalAveragePooling2D ] ──> Vector: (1, 1280)
       │
       ▼
[ Dropout Regularization (p = 0.2) ]
       │
       ▼
[ Dense Softmax Classification Head ] ──> 6 Neurons
       │
       ▼
Softmax Probabilities:
  ├── crazing (Cr)
  ├── inclusion (In)
  ├── patches (Pa)
  ├── pitted_surface (PS)
  ├── rolled-in_scale (RS)
  └── scratches (Sc)
```

#### Model Architecture Details:
* **Base Architecture**: MobileNetV2 initialized with ImageNet pretrained weights. Depthwise separable convolutions reduce parameter counts and floating-point operations (FLOPs) by ~85% relative to standard 2D convolutions of equivalent receptive field.
* **Feature Neck**: `GlobalAveragePooling2D` collapses the $(7 \times 7 \times 1280)$ spatial tensor into a 1280-element 1D feature representation.
* **Regularization**: `Dropout(rate=0.2)` randomly sets 20% of activations to 0 during training to prevent co-adaptation of weights.
* **Output Head**: `Dense(units=6, activation='softmax')` outputs mathematically normalized class likelihoods:
  $$P(y = c \mid \mathbf{x}) = \frac{\exp(z_c)}{\sum_{j=1}^{6} \exp(z_j)}, \quad \sum_{c=1}^6 P(y = c \mid \mathbf{x}) = 1.0$$

#### In-Memory Model Lifecycle Management:
* Rather than reloading the 9.6 MB `.keras` model artifact from disk on every HTTP request, the FastAPI backend loads the model **once at startup** using an `@asynccontextmanager` lifespan handler.
* A synthetic warm-up tensor `(1, 224, 224, 3)` is passed through the graph at boot, pre-allocating execution graphs and caching CUDA/CPU kernels.
* Subsequent inference requests achieve pure forward-pass execution in **15–35 milliseconds**.

---

### 4.8. Backend API & Service Layer
* **Source Implementation**: `backend/main.py`
* **Framework**: FastAPI (Asynchronous Python 3.10) with Uvicorn ASGI server.
* **CORS**: Fully configured for cross-origin access across modern local development ports.

#### Key API Endpoint Specifications:
| Method | Endpoint Path | Functionality | Request Type | Key Response Fields |
|---|---|---|---|---|
| `GET` | `/` | Service root check | None | `status: "online"`, docs route |
| `GET` | `/api/health` | Diagnostic & pipeline status | None | `technologies`, `pipelines` readiness |
| `POST` | `/api/preprocess` | Common image preparation | `multipart/form-data` | Base64 original, grayscale, normalized images |
| `POST` | `/api/pipelines/spatial` | Isolated spatial domain analysis | `multipart/form-data` | `enhanced_image`, `threshold_image`, `final_mask`, `overlay`, `anomaly_area` |
| `POST` | `/api/pipelines/edge-shape`| Isolated edge & contour analysis | `multipart/form-data` | `edge_map`, `contour_vis`, `anomaly_mask`, `shape_features` |
| `POST` | `/api/pipelines/frequency` | Isolated 2D-FFT spectral analysis | `multipart/form-data` | `fft_spectrum`, `reconstructed_image`, `high_to_low_frequency_ratio` |
| `POST` | `/api/pipelines/texture` | Isolated Gabor filter analysis | `multipart/form-data` | 4 Gabor orientation maps, `combined_response`, `texture_statistics` |
| `POST` | `/api/pipelines/consensus` | Pixel-wise voting fusion | Form data (`file` + $\ge 2$ masks) | `consensus_mask`, `consensus_overlay`, `supporting_pipeline_count` |
| `POST` | `/api/ml/classify` | Deep learning defect classifier | `multipart/form-data` | `predicted_class`, `confidence`, `scores` (all 6 classes), `inference_runtime_ms` |

---

### 4.9. Frontend User Interface & Visualization Engine
* **Source Implementation**: `frontend/src/App.tsx`, `frontend/src/index.css`
* **Stack**: React 19, TypeScript, Vite, Lucide Icons, Vanilla Modern CSS.

#### Key User Interface Modules:
1. **Header & System Health Telemetry**: Live pinging of `/api/health` displaying connection state, runtime library versions (OpenCV, NumPy, Scikit-Image, FastAPI), and pipeline readiness badges.
2. **Interactive Drag-and-Drop Uploader**:
   * Accepts JPG, PNG, and BMP files.
   * Displays instant file metadata (dimensions, byte size, format).
   * Includes a default sample loader to test inspection without manual file uploads.
3. **Execution Control Bar**:
   * Provides independent execution triggers for all four classical pipelines (`Spatial`, `Edge/Shape`, `Frequency`, `Texture`).
   * Features a `"Run All Pipelines"` batch execution button and a standalone `"Classify with MobileNetV2"` deep learning trigger.
4. **Deep Learning Classification Panel**:
   * Displays the top predicted defect class with its industrial short code (e.g., `PS` for *Pitted Surface*).
   * Shows a color-coded confidence score badge with millisecond inference execution latency.
   * Renders a 6-class horizontal probability distribution bar chart visualizing Softmax distribution across all categories.
5. **Individual Pipeline Inspection Cards**:
   * Interactive cards for each completed pipeline featuring side-by-side tabs for intermediate steps.
   * Displays pipeline-specific diagnostic metric grids (e.g., Circularity, High/Low Frequency Energy Ratio, Shannon Entropy, Defect Area Percentage).
6. **Side-by-Side Comparative Matrix & Consensus Card**:
   * Aligns overlays from all 4 classical pipelines in a comparative grid for visual cross-examination.
   * Generates the multi-colored Consensus Anomaly Mask with accompanying agreement metrics.

---

## 5. Target Defect Taxonomy & Dataset Specifications

DefectVision targets the six canonical surface defect classes established in the benchmark **NEU Surface Defect Database**:

| Defect Class | Short Code | Physical Manufacturing Mechanism | Typical Visual & Mathematical Signature | Best Detection Domain |
|---|---|---|---|---|
| **Crazing** | `Cr` | High-temperature thermal stress cracking during cooling | Fine, dense network of interconnected micro-cracks | **Frequency (FFT)** ($R_{H/L} \gg 1.0$) & **Edge** |
| **Inclusion** | `In` | Non-metallic chemical impurities/slag trapped in steel matrix | Small, dark or high-contrast discrete irregular spots | **Spatial** (intensity drop) & **Texture** |
| **Patches** | `Pa` | Oxidation or surface treatment irregularities | Broad, localized discolored areas with fuzzy boundaries | **Spatial (CLAHE)** & **Texture (Gabor)** |
| **Pitted Surface** | `PS` | Acid cleaning over-etching or localized chemical pitting | Clustered rough depressions with high circularity ($C \to 1.0$) | **Texture** (high entropy) & **Spatial** |
| **Rolled-in Scale**| `RS` | Iron oxide scale embedded by heavy rollers during rolling | Dark, elongated, or fish-scale textured bands | **Edge** ($AR > 2.0$) & **Texture** |
| **Scratches** | `Sc` | Mechanical abrasion from rollers or handling guides | Long, continuous linear striations with high aspect ratios ($AR \gg 3.0$) | **Edge/Shape** (Canny + aspect ratio) |

---

## 6. Resilience, Error Handling & System Validation

### 6.1. Defensive API Handling
DefectVision implements exhaustive edge-case validation across all endpoints:
* **Invalid MIME Type & File Extensions**: Uploading non-image payloads (e.g., `.txt`, `.pdf`) immediately aborts with `HTTP 400 Bad Request`.
* **Corrupted Image Payloads**: If uploaded byte streams cannot be decoded into an 8-bit image matrix by OpenCV or PIL, the system catches the null pointer and returns `HTTP 400 Bad Request: "Could not decode image bytes"`.
* **Empty / Zero-Byte Uploads**: Explicit buffer length checks reject empty files with `HTTP 400`.
* **Model Artifact Unavailability**: If `defectvision_mobilenet_v2.keras` is missing or corrupted, the service returns `HTTP 503 Service Unavailable` with diagnostic instructions rather than crashing the process.
* **Consensus Degeneracy Prevention**: Attempting to invoke the Consensus engine with fewer than 2 valid pipeline masks returns `HTTP 400 Bad Request`.
* **Mathematical Integrity Safeguards**: All inference outputs are validated by assertions ensuring:
  $$\forall i, \; 0.0 \le P_i \le 1.0 \quad \text{and} \quad \left| \sum_{i=1}^6 P_i - 1.0 \right| < 10^{-2}$$

### 6.2. Strict Pipeline Isolation Guarantee
The system maintains complete architectural decoupling:
* Invoking `POST /api/ml/classify` runs **only** the MobileNetV2 inference engine. None of the OpenCV classical routines (`run_spatial_pipeline`, `run_edge_shape_pipeline`, `run_frequency_pipeline`, `run_texture_pipeline`) are triggered during classification.
* Similarly, invoking any of the four classical endpoints runs strictly within its isolated mathematical domain without triggering neural network forward passes.

---

## 7. Deployment & Execution Guide

### 7.1. Prerequisites
* **Python**: Version 3.10 or 3.11 with `pip`.
* **Node.js**: Version 18+ with `npm`.
* Modern web browser (Chrome, Edge, Firefox).

### 7.2. Starting the Backend Server
```cmd
cd backend
pip install -r requirements.txt
pip install -r ../ml/requirements-ml.txt
uvicorn main:app --reload --port 8000
```
* **Backend URL**: `http://127.0.0.1:8000`
* **Interactive OpenAPI Swagger Docs**: `http://127.0.0.1:8000/docs`

### 7.3. Starting the Frontend Dashboard
```cmd
cd frontend
npm install
npm run dev
```
* **Frontend Web Dashboard**: `http://localhost:5173`

### 7.4. Executing Automated System Tests
```cmd
cd backend
python tests/test_full_system.py
```
