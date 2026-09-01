from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import cv2
import numpy as np
import skimage
import sys

from preprocessing import preprocess_image
from pipelines.spatial import run_spatial_pipeline, spatial_pipeline_stub
from pipelines.edge_shape import run_edge_shape_pipeline, edge_shape_pipeline_stub
from pipelines.frequency import run_frequency_pipeline, frequency_pipeline_stub
from pipelines.texture import run_texture_pipeline, texture_pipeline_stub

app = FastAPI(
    title="DefectVision API",
    description="Industrial Defect Detection Computer Vision Service",
    version="0.3.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_and_preprocess_upload(file: UploadFile, contents: bytes):
    """Helper method to validate image file and run common preprocessing."""
    if not (file.content_type.startswith("image/") or file.filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
        raise HTTPException(status_code=400, detail="Invalid file type. Uploaded file must be a valid image (JPG/PNG).")
    try:
        return preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preprocess uploaded image: {str(e)}")

@app.get("/")
def read_root():
    return {
        "project": "DefectVision API",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DefectVision Backend",
        "technologies": {
            "fastapi": "0.100+",
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scikit_image": skimage.__version__
        },
        "pipelines": {
            "spatial": spatial_pipeline_stub(),
            "edge_shape": edge_shape_pipeline_stub(),
            "frequency": frequency_pipeline_stub(),
            "texture": texture_pipeline_stub()
        }
    }

@app.post("/api/preprocess")
async def preprocess_endpoint(file: UploadFile = File(...)):
    """
    Exposes reusable preprocessing functionality.
    Returns original image, grayscale image, normalized image, width, and height.
    """
    contents = await file.read()
    results = validate_and_preprocess_upload(file, contents)

    return {
        "status": "success",
        "filename": file.filename,
        "width": results["width"],
        "height": results["height"],
        "original_image": results["original_base64"],
        "grayscale_image": results["grayscale_base64"],
        "normalized_image": results["normalized_base64"]
    }

# -----------------------------------------------------------------------------
# INDEPENDENT PIPELINE ENDPOINTS
# -----------------------------------------------------------------------------

@app.post("/api/pipelines/spatial")
async def spatial_pipeline_endpoint(file: UploadFile = File(...)):
    """
    Independent Endpoint for Spatial Domain Analysis.
    Validates, preprocesses, and executes ONLY Spatial Domain pipeline.
    """
    contents = await file.read()
    preprocessed = validate_and_preprocess_upload(file, contents)
    result = run_spatial_pipeline(preprocessed)
    return {
        "status": "success",
        "filename": file.filename,
        "result": result
    }

@app.post("/api/pipelines/edge-shape")
async def edge_shape_pipeline_endpoint(file: UploadFile = File(...)):
    """
    Independent Endpoint for Edge & Shape Analysis.
    Validates, preprocesses, and executes ONLY Edge & Shape pipeline.
    """
    contents = await file.read()
    preprocessed = validate_and_preprocess_upload(file, contents)
    result = run_edge_shape_pipeline(preprocessed)
    return {
        "status": "success",
        "filename": file.filename,
        "result": result
    }

@app.post("/api/pipelines/frequency")
async def frequency_pipeline_endpoint(file: UploadFile = File(...)):
    """
    Independent Endpoint for Frequency Domain (FFT) Analysis.
    Validates, preprocesses, and executes ONLY Frequency Domain pipeline.
    """
    contents = await file.read()
    preprocessed = validate_and_preprocess_upload(file, contents)
    result = run_frequency_pipeline(preprocessed)
    return {
        "status": "success",
        "filename": file.filename,
        "result": result
    }

@app.post("/api/pipelines/texture")
async def texture_pipeline_endpoint(file: UploadFile = File(...)):
    """
    Independent Endpoint for Texture Analysis (Gabor/LBP).
    Validates, preprocesses, and executes ONLY Texture Analysis pipeline.
    """
    contents = await file.read()
    preprocessed = validate_and_preprocess_upload(file, contents)
    result = run_texture_pipeline(preprocessed)
    return {
        "status": "success",
        "filename": file.filename,
        "result": result
    }

@app.post("/api/pipelines/consensus")
async def consensus_endpoint(
    file: UploadFile = File(...),
    spatial_mask: Optional[str] = Form(None),
    edge_shape_mask: Optional[str] = Form(None),
    frequency_mask: Optional[str] = Form(None),
    texture_mask: Optional[str] = Form(None)
):
    """
    Consensus Endpoint.
    Computes a pixel-wise 2+ voting consensus from executed pipeline anomaly masks.
    """
    contents = await file.read()
    preprocessed = validate_and_preprocess_upload(file, contents)
    original_bgr = preprocessed["original_bgr"]
    
    import base64
    def decode_mask(b64_str: str) -> np.ndarray:
        if ',' in b64_str:
            b64_str = b64_str.split(',', 1)[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
    masks_dict = {}
    if spatial_mask:
        masks_dict["spatial"] = decode_mask(spatial_mask)
    if edge_shape_mask:
        masks_dict["edge_shape"] = decode_mask(edge_shape_mask)
    if frequency_mask:
        masks_dict["frequency"] = decode_mask(frequency_mask)
    if texture_mask:
        masks_dict["texture"] = decode_mask(texture_mask)
        
    if len(masks_dict) < 2:
        raise HTTPException(status_code=400, detail="Consensus calculation requires at least 2 executed pipeline masks.")
        
    from pipelines.consensus import run_consensus_pipeline
    result = run_consensus_pipeline(original_bgr, masks_dict, min_agreement=2)
    
    return {
        "status": "success",
        "filename": file.filename,
        "result": result
    }

