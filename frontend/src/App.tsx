import { useState, useEffect, useRef, type ChangeEvent, type DragEvent } from 'react';
import { 
  Activity, 
  Cpu, 
  Layers, 
  RefreshCw, 
  Sliders, 
  Maximize2, 
  Grid, 
  Sparkles,
  CheckCircle2,
  AlertCircle,
  UploadCloud,
  FileImage,
  Trash2,
  Play,
  Columns,
  Eye,
  Info,
  Zap,
  ShieldCheck,
  ShieldAlert,
  Compass,
  Activity as ActivityIcon,
  Maximize
} from 'lucide-react';

interface BackendHealth {
  status: string;
  service: string;
  technologies: {
    fastapi: string;
    python: string;
    opencv: string;
    numpy: string;
    scikit_image: string;
  };
  pipelines: Record<string, { name: string; status: string; description: string }>;
}

interface ImageMetadata {
  name: string;
  size: string;
  width: number;
  height: number;
  type: string;
}

interface SpatialResult {
  pipeline: string;
  status: string;
  enhanced_image: string;
  threshold_image: string;
  final_mask: string;
  overlay: string;
  anomaly_detected: boolean;
  anomaly_area: number;
  anomaly_area_percentage: number;
  region_count: number;
  processing_time: number;
}

interface ShapeFeature {
  region_id: number;
  area: number;
  perimeter: number;
  width: number;
  height: number;
  aspect_ratio: number;
  circularity: number;
}

interface EdgeShapeResult {
  pipeline: string;
  status: string;
  edge_map: string;
  contour_visualization: string;
  anomaly_mask: string;
  overlay: string;
  shape_features: ShapeFeature[];
  anomaly_area: number;
  region_count: number;
  processing_time: number;
}

interface FrequencyResult {
  pipeline: string;
  status: string;
  fft_spectrum: string;
  filtered_frequency_representation: string;
  reconstructed_image: string;
  anomaly_mask: string;
  overlay: string;
  total_frequency_energy: number;
  low_frequency_energy: number;
  high_frequency_energy: number;
  high_to_low_frequency_ratio: number;
  anomaly_area: number;
  region_count: number;
  processing_time: number;
}

interface TextureStatistics {
  mean_response: number;
  variance: number;
  energy: number;
  contrast: number;
  entropy: number;
}

interface TextureResult {
  pipeline: string;
  status: string;
  gabor_0_response: string;
  gabor_45_response: string;
  gabor_90_response: string;
  gabor_135_response: string;
  combined_response: string;
  anomaly_mask: string;
  overlay: string;
  texture_statistics: TextureStatistics;
  anomaly_area: number;
  region_count: number;
  processing_time: number;
}

interface ConsensusResult {
  consensus_mask: string;
  consensus_overlay: string;
  supporting_pipeline_count: number;
  consensus_area: number;
  agreement_summary: string;
  processing_time: number;
}

interface MLClassificationResult {
  status: string;
  filename: string;
  predicted_class: string;
  short_code: string;
  confidence: number;
  confidence_percentage: string;
  scores: Record<string, number>;
  inference_runtime_ms: number;
}

export function App() {
  const [healthData, setHealthData] = useState<BackendHealth | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [_error, setError] = useState<string | null>(null);

  // Uploaded Image state
  const [rawFile, setRawFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageMeta, setImageMeta] = useState<ImageMetadata | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Independent Pipeline States (Four Classical Pipelines)
  const [activePipelineLoading, setActivePipelineLoading] = useState<string | null>(null);
  const [spatialResult, setSpatialResult] = useState<SpatialResult | null>(null);
  const [edgeShapeResult, setEdgeShapeResult] = useState<EdgeShapeResult | null>(null);
  const [frequencyResult, setFrequencyResult] = useState<FrequencyResult | null>(null);
  const [textureResult, setTextureResult] = useState<TextureResult | null>(null);
  const [_pipelineResults, setPipelineResults] = useState<Record<string, any>>({});
  const [activeMessage, setActiveMessage] = useState<string | null>(null);

  // Separate Deep Learning (MobileNetV2) States
  const [mlResult, setMlResult] = useState<MLClassificationResult | null>(null);
  const [mlLoading, setMlLoading] = useState<boolean>(false);
  const [mlError, setMlError] = useState<string | null>(null);
  const mlResultRef = useRef<HTMLDivElement>(null);

  // Compare All & Consensus States & Refs
  const [showComparison, setShowComparison] = useState<boolean>(false);
  const [consensusResult, setConsensusResult] = useState<ConsensusResult | null>(null);
  const [consensusLoading, setConsensusLoading] = useState<boolean>(false);

  const comparisonRef = useRef<HTMLDivElement>(null);
  const consensusRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchBackendHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/api/health');
      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }
      const data: BackendHealth = await response.json();
      setHealthData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to backend server');
      setHealthData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBackendHealth();
  }, []);

  // Format file size nicely
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // File Processor
  const processFile = (file: File) => {
    setUploadError(null);
    setActiveMessage(null);

    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const extension = file.name.split('.').pop()?.toLowerCase();
    
    if (!validTypes.includes(file.type) && !['jpg', 'jpeg', 'png'].includes(extension || '')) {
      setUploadError('Invalid file type. Please select a JPG, JPEG, or PNG image.');
      return;
    }

    setRawFile(file);
    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.src = objectUrl;

    img.onload = () => {
      setPreviewUrl(objectUrl);
      setImageMeta({
        name: file.name,
        size: formatFileSize(file.size),
        width: img.naturalWidth,
        height: img.naturalHeight,
        type: file.type || `image/${extension}`
      });
    };

    img.onerror = () => {
      setUploadError('Failed to load image file. Please try another file.');
    };
  };

  // Helper for demo sample image
  const loadSampleImage = async () => {
    setUploadError(null);
    setActiveMessage(null);
    try {
      const response = await fetch('/sample_defect.png');
      const blob = await response.blob();
      const sampleFile = new File([blob], 'sample_defect_surface.png', { type: 'image/png' });
      processFile(sampleFile);
    } catch {
      setUploadError('Unable to load sample defect image.');
    }
  };

  // File Input Change
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  // Remove uploaded image
  const handleRemoveImage = () => {
    if (previewUrl && !previewUrl.startsWith('/')) {
      URL.revokeObjectURL(previewUrl);
    }
    setRawFile(null);
    setPreviewUrl(null);
    setImageMeta(null);
    setUploadError(null);
    setActiveMessage(null);
    setSpatialResult(null);
    setEdgeShapeResult(null);
    setFrequencyResult(null);
    setTextureResult(null);
    setConsensusResult(null);
    setMlResult(null);
    setMlError(null);
    setShowComparison(false);
    setPipelineResults({});
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // INDEPENDENT PIPELINE EXECUTIONS
  const executeSinglePipeline = async (pipelineEndpoint: string, label: string) => {
    if (!rawFile && !previewUrl) return;

    setActivePipelineLoading(label);
    setActiveMessage(null);

    try {
      let fileToUpload = rawFile;

      if (!fileToUpload && previewUrl) {
        const resp = await fetch(previewUrl);
        const blob = await resp.blob();
        fileToUpload = new File([blob], imageMeta?.name || 'target_image.png', { type: blob.type || 'image/png' });
      }

      const formData = new FormData();
      formData.append('file', fileToUpload!);

      const res = await fetch(`http://localhost:8000/api/pipelines/${pipelineEndpoint}`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP status ${res.status}`);
      }

      const data = await res.json();

      if (pipelineEndpoint === 'spatial') {
        setSpatialResult(data.result as SpatialResult);
      } else if (pipelineEndpoint === 'edge-shape') {
        setEdgeShapeResult(data.result as EdgeShapeResult);
      } else if (pipelineEndpoint === 'frequency') {
        setFrequencyResult(data.result as FrequencyResult);
      } else if (pipelineEndpoint === 'texture') {
        setTextureResult(data.result as TextureResult);
      }

      setPipelineResults(prev => ({
        ...prev,
        [pipelineEndpoint]: data.result
      }));

      setActiveMessage(`Successfully executed independent endpoint: POST /api/pipelines/${pipelineEndpoint}`);
    } catch (err) {
      setActiveMessage(`Execution error for ${label}: ${err instanceof Error ? err.message : 'Failed endpoint call'}`);
    } finally {
      setActivePipelineLoading(null);
    }
  };

  // INDEPENDENT DEEP LEARNING (MOBILENETV2) EXECUTION
  const executeDeepLearning = async () => {
    if (!rawFile && !previewUrl) return;

    setMlLoading(true);
    setMlError(null);
    setActiveMessage('Executing MobileNetV2 Deep Learning classification (POST /api/ml/classify)...');

    try {
      let fileToUpload = rawFile;

      if (!fileToUpload && previewUrl) {
        const resp = await fetch(previewUrl);
        const blob = await resp.blob();
        fileToUpload = new File([blob], imageMeta?.name || 'target_image.png', { type: blob.type || 'image/png' });
      }

      const formData = new FormData();
      formData.append('file', fileToUpload!);

      const res = await fetch('http://localhost:8000/api/ml/classify', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned HTTP status ${res.status}`);
      }

      const data: MLClassificationResult = await res.json();
      setMlResult(data);
      setActiveMessage(`Deep Learning classification complete: ${data.predicted_class.toUpperCase()} (${data.confidence_percentage}) in ${data.inference_runtime_ms} ms.`);

      setTimeout(() => {
        mlResultRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Deep learning inference failed';
      setMlError(msg);
      setActiveMessage(`Deep Learning Error: ${msg}`);
    } finally {
      setMlLoading(false);
    }
  };

  // Compare All Handler: DOES NOT RUN PIPELINES, only compares existing state!
  const handleCompareAllClick = () => {
    setShowComparison(true);
    setActiveMessage('Comparing pipeline execution results currently stored in frontend state.');
    setTimeout(() => {
      comparisonRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  // Consensus Calculation Handler (Requires at least 2 executed pipelines)
  const executedCount = [spatialResult, edgeShapeResult, frequencyResult, textureResult].filter(Boolean).length;
  const canGenerateConsensus = executedCount >= 2;

  const handleGenerateConsensus = async () => {
    if (executedCount < 2) return;
    setConsensusLoading(true);
    setActiveMessage('Generating 2+ Pipeline Voting Consensus...');

    try {
      let fileToUpload = rawFile;
      if (!fileToUpload && previewUrl) {
        const resp = await fetch(previewUrl);
        const blob = await resp.blob();
        fileToUpload = new File([blob], imageMeta?.name || 'target_image.png', { type: blob.type || 'image/png' });
      }

      const formData = new FormData();
      formData.append('file', fileToUpload!);

      if (spatialResult) formData.append('spatial_mask', spatialResult.final_mask);
      if (edgeShapeResult) formData.append('edge_shape_mask', edgeShapeResult.anomaly_mask);
      if (frequencyResult) formData.append('frequency_mask', frequencyResult.anomaly_mask);
      if (textureResult) formData.append('texture_mask', textureResult.anomaly_mask);

      const res = await fetch('http://localhost:8000/api/pipelines/consensus', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: Failed to generate consensus`);
      }

      const data = await res.json();
      setConsensusResult(data.result as ConsensusResult);
      setActiveMessage(`Successfully generated 2+ Pipeline Consensus across ${executedCount} executed pipelines.`);

      setTimeout(() => {
        consensusRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err) {
      setActiveMessage(`Consensus generation failed: ${err instanceof Error ? err.message : 'API error'}`);
    } finally {
      setConsensusLoading(false);
    }
  };

  const isImageUploaded = Boolean(previewUrl && imageMeta);

  return (
    <div className="dashboard-container">
      {/* Header Bar */}
      <header className="header-bar">
        <div className="brand-title">
          <div className="brand-icon">
            <Activity className="text-white" size={24} />
          </div>
          <div>
            <h1>DefectVision</h1>
            <p className="brand-subtitle">Industrial Quality Control & Computer Vision System</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button 
            onClick={fetchBackendHealth} 
            className="btn-secondary"
            disabled={loading}
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh Status
          </button>
          
          <div className={`status-badge ${healthData ? 'online' : loading ? 'checking' : 'offline'}`}>
            <span className="pulse-dot"></span>
            {healthData ? 'Backend Connected' : loading ? 'Checking Connection...' : 'Backend Disconnected'}
          </div>
        </div>
      </header>

      {/* IMAGE UPLOADER SECTION */}
      <section className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <h2 className="section-heading" style={{ margin: 0 }}>
            <FileImage size={22} style={{ color: 'var(--primary)' }} />
            Defect Inspection Target Image
          </h2>
          {!isImageUploaded && (
            <button onClick={loadSampleImage} className="btn-secondary" style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}>
              <Eye size={14} />
              Load Sample Image
            </button>
          )}
        </div>

        <input 
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".jpg,.jpeg,.png,image/jpeg,image/png"
          style={{ display: 'none' }}
        />

        {uploadError && (
          <div style={{
            background: 'rgba(244, 63, 94, 0.12)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: '8px',
            padding: '0.75rem 1rem',
            color: '#fb7185',
            fontSize: '0.875rem',
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <AlertCircle size={16} />
            {uploadError}
          </div>
        )}

        {!isImageUploaded ? (
          /* Drag and Drop Zone */
          <div 
            className={`upload-dropzone ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-icon-wrapper">
              <UploadCloud size={28} />
            </div>
            <h3 className="upload-title">Drag & drop defect surface image here</h3>
            <p className="upload-subtitle">or click to browse your computer files</p>
            <span className="file-types-badge">Supported Formats: JPG, JPEG, PNG</span>
          </div>
        ) : (
          /* Preview and Metadata Section */
          <div className="preview-container">
            <div className="preview-image-wrapper">
              <img src={previewUrl!} alt="Uploaded surface preview" />
            </div>

            <div className="preview-details">
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f9fafb', wordBreak: 'break-all' }}>
                      {imageMeta?.name}
                    </h3>
                    <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                      Surface Inspection Target Ready
                    </p>
                  </div>
                  <button onClick={handleRemoveImage} className="btn-danger">
                    <Trash2 size={14} />
                    Remove Image
                  </button>
                </div>

                <div className="meta-grid">
                  <div className="meta-card">
                    <div className="meta-label">Dimensions</div>
                    <div className="meta-value">{imageMeta?.width} × {imageMeta?.height} px</div>
                  </div>
                  <div className="meta-card">
                    <div className="meta-label">File Size</div>
                    <div className="meta-value">{imageMeta?.size}</div>
                  </div>
                  <div className="meta-card">
                    <div className="meta-label">Format</div>
                    <div className="meta-value" style={{ color: 'var(--accent-cyan)' }}>
                      {imageMeta?.name.split('.').pop()?.toUpperCase()}
                    </div>
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-emerald)', fontSize: '0.85rem' }}>
                <CheckCircle2 size={16} />
                Image uploaded & validated for computer vision pipeline processing
              </div>
            </div>
          </div>
        )}

        {/* TOP ACTION BAR */}
        <div className="pipeline-actions-bar">
          <button 
            disabled={!isImageUploaded || activePipelineLoading !== null}
            onClick={() => executeSinglePipeline('spatial', 'Spatial')}
            className="btn-action"
          >
            <Play size={14} className={activePipelineLoading === 'Spatial' ? 'animate-spin' : ''} style={{ color: 'var(--primary)' }} />
            {activePipelineLoading === 'Spatial' ? 'Running Spatial...' : 'Run Spatial'}
          </button>

          <button 
            disabled={!isImageUploaded || activePipelineLoading !== null}
            onClick={() => executeSinglePipeline('edge-shape', 'Edge/Shape')}
            className="btn-action"
          >
            <Sliders size={14} className={activePipelineLoading === 'Edge/Shape' ? 'animate-spin' : ''} style={{ color: 'var(--accent-cyan)' }} />
            {activePipelineLoading === 'Edge/Shape' ? 'Running Edge/Shape...' : 'Run Edge/Shape'}
          </button>

          <button 
            disabled={!isImageUploaded || activePipelineLoading !== null}
            onClick={() => executeSinglePipeline('frequency', 'FFT')}
            className="btn-action"
          >
            <Grid size={14} className={activePipelineLoading === 'FFT' ? 'animate-spin' : ''} style={{ color: 'var(--accent-amber)' }} />
            {activePipelineLoading === 'FFT' ? 'Running FFT...' : 'Run FFT'}
          </button>

          <button 
            disabled={!isImageUploaded || activePipelineLoading !== null}
            onClick={() => executeSinglePipeline('texture', 'Texture/Gabor')}
            className="btn-action"
          >
            <Maximize2 size={14} className={activePipelineLoading === 'Texture/Gabor' ? 'animate-spin' : ''} style={{ color: 'var(--accent-emerald)' }} />
            {activePipelineLoading === 'Texture/Gabor' ? 'Running Texture...' : 'Run Texture/Gabor'}
          </button>

          {/* DEDICATED DEEP LEARNING BUTTON */}
          <button 
            disabled={!isImageUploaded || mlLoading}
            onClick={executeDeepLearning}
            className="btn-action"
            style={{
              borderColor: isImageUploaded ? '#818cf8' : 'rgba(255, 255, 255, 0.1)',
              background: mlLoading ? 'rgba(99, 102, 241, 0.2)' : 'rgba(99, 102, 241, 0.08)',
              color: isImageUploaded ? '#c7d2fe' : 'var(--text-dim)',
              fontWeight: 600
            }}
          >
            <Cpu size={14} className={mlLoading ? 'animate-spin' : ''} style={{ color: isImageUploaded ? '#818cf8' : 'var(--text-dim)' }} />
            {mlLoading ? 'Classifying Defect...' : '[Run Deep Learning]'}
          </button>

          <button 
            disabled={!isImageUploaded || activePipelineLoading !== null}
            onClick={handleCompareAllClick}
            className="btn-compare"
          >
            <Columns size={16} />
            Compare All
          </button>

          <button 
            disabled={!canGenerateConsensus || consensusLoading}
            onClick={handleGenerateConsensus}
            className="btn-action"
            style={{
              borderColor: canGenerateConsensus ? 'var(--accent-purple)' : 'rgba(255, 255, 255, 0.1)',
              color: canGenerateConsensus ? '#a78bfa' : 'var(--text-dim)',
              cursor: canGenerateConsensus ? 'pointer' : 'not-allowed'
            }}
          >
            <Sparkles size={14} className={consensusLoading ? 'animate-spin' : ''} style={{ color: canGenerateConsensus ? '#a78bfa' : 'var(--text-dim)' }} />
            {consensusLoading ? 'Generating Consensus...' : '[Generate Consensus]'}
          </button>
        </div>

        {activeMessage && (
          <div style={{
            marginTop: '1rem',
            padding: '0.75rem 1rem',
            background: 'rgba(59, 130, 246, 0.12)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '8px',
            color: '#93c5fd',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <Info size={16} />
            {activeMessage}
          </div>
        )}
      </section>

      {/* 2+ PIPELINE CONSENSUS SECTION */}
      {consensusResult && (
        <section ref={consensusRef} className="glass-card" style={{ border: '1px solid rgba(168, 85, 247, 0.4)', marginBottom: '2.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
            <div>
              <h2 className="section-heading" style={{ margin: 0, color: '#c084fc' }}>
                <Sparkles size={22} />
                2+ Pipeline Consensus
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
                Pixel-wise voting agreement calculated across {consensusResult.supporting_pipeline_count} executed pipelines
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span className="status-badge online" style={{ background: 'rgba(168, 85, 247, 0.15)', color: '#e9d5ff', borderColor: 'rgba(168, 85, 247, 0.3)' }}>
                "{consensusResult.agreement_summary}"
              </span>
              <span style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', fontFamily: 'monospace' }}>
                {consensusResult.processing_time} ms
              </span>
            </div>
          </div>

          <div className="meta-grid" style={{ marginBottom: '1.5rem' }}>
            <div className="meta-card">
              <div className="meta-label">Supporting Pipeline Count</div>
              <div className="meta-value" style={{ color: '#c084fc' }}>
                {consensusResult.supporting_pipeline_count} Executed Pipelines
              </div>
            </div>
            <div className="meta-card">
              <div className="meta-label">Consensus Area (≥ 2 Pipelines)</div>
              <div className="meta-value">{consensusResult.consensus_area.toLocaleString()} px</div>
            </div>
            <div className="meta-card">
              <div className="meta-label">Agreement Summary</div>
              <div className="meta-value" style={{ fontSize: '0.9rem', color: 'var(--accent-emerald)' }}>
                "{consensusResult.agreement_summary}"
              </div>
            </div>
            <div className="meta-card">
              <div className="meta-label">Runtime</div>
              <div className="meta-value" style={{ color: 'var(--accent-amber)' }}>{consensusResult.processing_time} ms</div>
            </div>
          </div>

          {/* Consensus Mask & Consensus Overlay Images */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
            <div className="meta-card" style={{ padding: '0.85rem' }}>
              <div className="meta-label" style={{ marginBottom: '0.5rem', fontWeight: 600, color: '#f3f4f6' }}>
                Consensus Mask
              </div>
              <img src={consensusResult.consensus_mask} alt="Consensus Mask" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
            </div>

            <div className="meta-card" style={{ padding: '0.85rem' }}>
              <div className="meta-label" style={{ marginBottom: '0.5rem', fontWeight: 600, color: '#f3f4f6' }}>
                Consensus Overlay
              </div>
              <img src={consensusResult.consensus_overlay} alt="Consensus Overlay" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
            </div>
          </div>

          <div style={{ marginTop: '1.25rem', padding: '0.75rem 1rem', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            ⚠️ <em>Consensus reflects multi-algorithm voting overlap across executed pipelines; it is not treated as ground truth nor assigned confidence scores.</em>
          </div>
        </section>
      )}

      {/* COMPARISON MATRIX & VISUAL COMPARISON SECTION */}
      {showComparison && (
        <section ref={comparisonRef} className="glass-card" style={{ border: '1px solid rgba(139, 92, 246, 0.4)', marginBottom: '2.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
            <div>
              <h2 className="section-heading" style={{ margin: 0, color: '#a78bfa' }}>
                <Columns size={22} />
                Pipeline Comparison Matrix
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
                Side-by-side behavioral comparison on the same target image
              </p>
            </div>

            <button 
              onClick={() => setShowComparison(false)}
              className="btn-secondary"
              style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}
            >
              Close Comparison
            </button>
          </div>

          {/* VISUAL COMPARISON GRID (Only when all 4 pipelines executed) */}
          {spatialResult && edgeShapeResult && frequencyResult && textureResult ? (
            <div style={{ marginBottom: '2rem', background: 'rgba(0, 0, 0, 0.25)', padding: '1.25rem', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f3f4f6', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Eye size={16} style={{ color: '#a78bfa' }} />
                Side-by-Side Visual Inspection Grid
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: '100px repeat(4, 1fr)', gap: '0.75rem', alignItems: 'center' }}>
                {/* Column Headers */}
                <div></div>
                <div style={{ textAlign: 'center', fontWeight: 700, color: '#60a5fa', fontSize: '0.85rem', letterSpacing: '0.05em' }}>SPATIAL</div>
                <div style={{ textAlign: 'center', fontWeight: 700, color: 'var(--accent-cyan)', fontSize: '0.85rem', letterSpacing: '0.05em' }}>EDGE</div>
                <div style={{ textAlign: 'center', fontWeight: 700, color: 'var(--accent-amber)', fontSize: '0.85rem', letterSpacing: '0.05em' }}>FFT</div>
                <div style={{ textAlign: 'center', fontWeight: 700, color: 'var(--accent-emerald)', fontSize: '0.85rem', letterSpacing: '0.05em' }}>GABOR</div>

                {/* Row 1: Original */}
                <div style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.85rem' }}>Original</div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={previewUrl!} alt="Spatial Original" style={{ width: '100%', borderRadius: '4px' }} />
                </div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={previewUrl!} alt="Edge Original" style={{ width: '100%', borderRadius: '4px' }} />
                </div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={previewUrl!} alt="FFT Original" style={{ width: '100%', borderRadius: '4px' }} />
                </div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={previewUrl!} alt="Gabor Original" style={{ width: '100%', borderRadius: '4px' }} />
                </div>

                {/* Row 2: Mask */}
                <div style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.85rem' }}>Mask</div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={spatialResult.final_mask} alt="Spatial Mask" style={{ width: '100%', borderRadius: '4px' }} />
                </div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={edgeShapeResult.anomaly_mask} alt="Edge Mask" style={{ width: '100%', borderRadius: '4px' }} />
                </div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={frequencyResult.anomaly_mask} alt="FFT Mask" style={{ width: '100%', borderRadius: '4px' }} />
                </div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={textureResult.anomaly_mask} alt="Gabor Mask" style={{ width: '100%', borderRadius: '4px' }} />
                </div>

                {/* Row 3: Overlay */}
                <div style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.85rem' }}>Overlay</div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={spatialResult.overlay} alt="Spatial Overlay" style={{ width: '100%', borderRadius: '4px' }} />
                </div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={edgeShapeResult.overlay} alt="Edge Overlay" style={{ width: '100%', borderRadius: '4px' }} />
                </div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={frequencyResult.overlay} alt="FFT Overlay" style={{ width: '100%', borderRadius: '4px' }} />
                </div>
                <div className="meta-card" style={{ padding: '0.35rem' }}>
                  <img src={textureResult.overlay} alt="Gabor Overlay" style={{ width: '100%', borderRadius: '4px' }} />
                </div>
              </div>
            </div>
          ) : (
            <div style={{
              padding: '1rem 1.25rem',
              marginBottom: '1.5rem',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px dashed rgba(255, 255, 255, 0.12)',
              borderRadius: '8px',
              fontSize: '0.85rem',
              color: 'var(--text-muted)'
            }}>
              💡 <strong>Visual Grid Note:</strong> Execute all four pipelines ([Run Spatial], [Run Edge/Shape], [Run FFT], [Run Texture/Gabor]) to view the complete 4-column visual alignment grid.
            </div>
          )}

          {/* NUMERICAL COMPARISON TABLE */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
              <thead>
                <tr style={{ background: 'rgba(255, 255, 255, 0.06)', borderBottom: '1px solid rgba(255, 255, 255, 0.12)' }}>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'left' }}>Pipeline</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'left' }}>Status</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'left' }}>Anomaly</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'left' }}>Area %</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'left' }}>Regions</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'left' }}>Runtime</th>
                </tr>
              </thead>
              <tbody>
                {/* Spatial Row */}
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#60a5fa' }}>Spatial</td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span style={{
                      padding: '0.25rem 0.6rem',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      fontFamily: 'monospace',
                      background: spatialResult ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.06)',
                      color: spatialResult ? '#34d399' : 'var(--text-dim)',
                      border: `1px solid ${spatialResult ? 'rgba(16, 185, 129, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`
                    }}>
                      {spatialResult ? 'Executed' : 'Not Run'}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: spatialResult ? (spatialResult.anomaly_detected ? '#fb7185' : '#34d399') : 'var(--text-dim)' }}>
                    {spatialResult ? (spatialResult.anomaly_detected ? 'Detected' : 'Clean') : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: spatialResult ? '#f3f4f6' : 'var(--text-dim)' }}>
                    {spatialResult ? `${spatialResult.anomaly_area_percentage}%` : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: spatialResult ? '#f3f4f6' : 'var(--text-dim)' }}>
                    {spatialResult ? spatialResult.region_count : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: spatialResult ? 'var(--accent-amber)' : 'var(--text-dim)' }}>
                    {spatialResult ? `${spatialResult.processing_time} ms` : 'Not Run'}
                  </td>
                </tr>

                {/* Edge/Shape Row */}
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>Edge/Shape</td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span style={{
                      padding: '0.25rem 0.6rem',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      fontFamily: 'monospace',
                      background: edgeShapeResult ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.06)',
                      color: edgeShapeResult ? '#34d399' : 'var(--text-dim)',
                      border: `1px solid ${edgeShapeResult ? 'rgba(16, 185, 129, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`
                    }}>
                      {edgeShapeResult ? 'Executed' : 'Not Run'}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: edgeShapeResult ? (edgeShapeResult.region_count > 0 ? '#fb7185' : '#34d399') : 'var(--text-dim)' }}>
                    {edgeShapeResult ? (edgeShapeResult.region_count > 0 ? 'Detected' : 'Clean') : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: edgeShapeResult ? '#f3f4f6' : 'var(--text-dim)' }}>
                    {edgeShapeResult && imageMeta ? `${((edgeShapeResult.anomaly_area / (imageMeta.width * imageMeta.height)) * 100).toFixed(2)}%` : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: edgeShapeResult ? '#f3f4f6' : 'var(--text-dim)' }}>
                    {edgeShapeResult ? edgeShapeResult.region_count : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: edgeShapeResult ? 'var(--accent-amber)' : 'var(--text-dim)' }}>
                    {edgeShapeResult ? `${edgeShapeResult.processing_time} ms` : 'Not Run'}
                  </td>
                </tr>

                {/* FFT Row */}
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--accent-amber)' }}>FFT</td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span style={{
                      padding: '0.25rem 0.6rem',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      fontFamily: 'monospace',
                      background: frequencyResult ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.06)',
                      color: frequencyResult ? '#34d399' : 'var(--text-dim)',
                      border: `1px solid ${frequencyResult ? 'rgba(16, 185, 129, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`
                    }}>
                      {frequencyResult ? 'Executed' : 'Not Run'}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: frequencyResult ? (frequencyResult.anomaly_area > 0 ? '#fb7185' : '#34d399') : 'var(--text-dim)' }}>
                    {frequencyResult ? (frequencyResult.anomaly_area > 0 ? 'Detected' : 'Clean') : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: frequencyResult ? '#f3f4f6' : 'var(--text-dim)' }}>
                    {frequencyResult && imageMeta ? `${((frequencyResult.anomaly_area / (imageMeta.width * imageMeta.height)) * 100).toFixed(2)}%` : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: frequencyResult ? '#f3f4f6' : 'var(--text-dim)' }}>
                    {frequencyResult ? frequencyResult.region_count : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: frequencyResult ? 'var(--accent-amber)' : 'var(--text-dim)' }}>
                    {frequencyResult ? `${frequencyResult.processing_time} ms` : 'Not Run'}
                  </td>
                </tr>

                {/* Texture/Gabor Row */}
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: 'var(--accent-emerald)' }}>Texture/Gabor</td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span style={{
                      padding: '0.25rem 0.6rem',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      fontFamily: 'monospace',
                      background: textureResult ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.06)',
                      color: textureResult ? '#34d399' : 'var(--text-dim)',
                      border: `1px solid ${textureResult ? 'rgba(16, 185, 129, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`
                    }}>
                      {textureResult ? 'Executed' : 'Not Run'}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: textureResult ? (textureResult.anomaly_area > 0 ? '#fb7185' : '#34d399') : 'var(--text-dim)' }}>
                    {textureResult ? (textureResult.anomaly_area > 0 ? 'Detected' : 'Clean') : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: textureResult ? '#f3f4f6' : 'var(--text-dim)' }}>
                    {textureResult && imageMeta ? `${((textureResult.anomaly_area / (imageMeta.width * imageMeta.height)) * 100).toFixed(2)}%` : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: textureResult ? '#f3f4f6' : 'var(--text-dim)' }}>
                    {textureResult ? textureResult.region_count : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: textureResult ? 'var(--accent-amber)' : 'var(--text-dim)' }}>
                    {textureResult ? `${textureResult.processing_time} ms` : 'Not Run'}
                  </td>
                </tr>

                {/* MobileNetV2 Deep Learning Row */}
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)', background: 'rgba(99, 102, 241, 0.05)' }}>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#a5b4fc', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    <Cpu size={15} style={{ color: '#818cf8' }} />
                    MobileNetV2 (Deep Learning)
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span style={{
                      padding: '0.25rem 0.6rem',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      fontFamily: 'monospace',
                      background: mlResult ? 'rgba(99, 102, 241, 0.15)' : 'rgba(255, 255, 255, 0.06)',
                      color: mlResult ? '#c7d2fe' : 'var(--text-dim)',
                      border: `1px solid ${mlResult ? 'rgba(99, 102, 241, 0.35)' : 'rgba(255, 255, 255, 0.1)'}`
                    }}>
                      {mlResult ? 'Executed' : 'Not Run'}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: mlResult ? '#34d399' : 'var(--text-dim)' }}>
                    {mlResult ? (
                      <span style={{ fontWeight: 600, color: '#a5b4fc' }}>
                        {mlResult.predicted_class.toUpperCase().replace('_', ' ')} ({mlResult.short_code}) — {mlResult.confidence_percentage}
                      </span>
                    ) : 'Not Run'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontSize: '0.8rem', color: 'var(--text-dim)', fontStyle: 'italic' }}>
                    N/A (No Pixel Mask)
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontSize: '0.8rem', color: 'var(--text-dim)', fontStyle: 'italic' }}>
                    N/A (Whole-Image)
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: mlResult ? 'var(--accent-amber)' : 'var(--text-dim)' }}>
                    {mlResult ? `${mlResult.inference_runtime_ms} ms` : 'Not Run'}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* HIGHLIGHTED NUMERICAL DIFFERENCES BREAKDOWN */}
          {imageMeta && (
            <div style={{ marginTop: '1.5rem', background: 'rgba(59, 130, 246, 0.06)', padding: '1rem 1.25rem', borderRadius: '8px', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
              <h4 style={{ fontSize: '0.875rem', fontWeight: 600, color: '#93c5fd', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Sparkles size={16} /> Numerical & Algorithmic Differences Analysis
              </h4>
              <ul style={{ fontSize: '0.825rem', color: 'var(--text-muted)', paddingLeft: '1.2rem', margin: 0, display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                {spatialResult && edgeShapeResult && frequencyResult && textureResult && (
                  <li>
                    <strong>Spatial Area Coverage Variance:</strong> Spatial ({spatialResult.anomaly_area_percentage}%), Edge ({((edgeShapeResult.anomaly_area / (imageMeta.width * imageMeta.height)) * 100).toFixed(2)}%), FFT ({((frequencyResult.anomaly_area / (imageMeta.width * imageMeta.height)) * 100).toFixed(2)}%), Gabor ({((textureResult.anomaly_area / (imageMeta.width * imageMeta.height)) * 100).toFixed(2)}%).
                  </li>
                )}
                <li>
                  <strong>Runtime Latency Comparison:</strong> MobileNetV2 ({mlResult ? `${mlResult.inference_runtime_ms} ms` : 'Not Run'}) vs Edge/Shape ({edgeShapeResult ? `${edgeShapeResult.processing_time} ms` : 'Not Run'}) vs Spatial ({spatialResult ? `${spatialResult.processing_time} ms` : 'Not Run'}) vs FFT ({frequencyResult ? `${frequencyResult.processing_time} ms` : 'Not Run'}) vs Gabor ({textureResult ? `${textureResult.processing_time} ms` : 'Not Run'}).
                </li>
                <li>
                  <strong>Localization vs. Classification Separation:</strong> Classical CV pipelines (Spatial, Edge, FFT, Texture) compute spatial anomaly masks from deterministic pixel transformations (intensity contrast, boundary edges, 2D Fourier high frequencies, Gabor wavelets). In contrast, MobileNetV2 performs hierarchical semantic feature extraction to produce 6-class softmax probabilities for the entire image without spatial localization.
                </li>
              </ul>
            </div>
          )}

          {/* GROUND-TRUTH EVALUATION METHODOLOGY & REQUIREMENTS */}
          <div style={{
            marginTop: '1.5rem',
            background: 'rgba(99, 102, 241, 0.04)',
            padding: '1.25rem',
            borderRadius: '10px',
            border: '1px solid rgba(129, 140, 248, 0.25)'
          }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#c7d2fe', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <ShieldCheck size={18} style={{ color: '#818cf8' }} />
              Ground-Truth Evaluation Requirements & Methodology
            </h4>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: '1.45' }}>
              DefectVision strictly avoids fabricating evaluation metrics (e.g. Accuracy, IoU, Dice, Precision, Recall) on live unannotated images. Quantitative metrics are scientifically valid only when evaluated against real ground truth:
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
              <div style={{ background: 'rgba(0, 0, 0, 0.25)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                <div style={{ fontWeight: 600, color: '#38bdf8', fontSize: '0.85rem', marginBottom: '0.4rem' }}>
                  1. Pixel-Level Segmentation (Classical Pipelines & Consensus)
                </div>
                <div style={{ fontSize: '0.8rem', color: '#e2e8f0', marginBottom: '0.4rem' }}>
                  <strong>Requires:</strong> Ground-Truth Pixel Binary Defect Masks (M_true ∈ {'{0, 1}'}^H×W)
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                  Evaluating segmentation quality (Intersection-over-Union IoU, Dice similarity coefficient, pixel-level true positive and false alarm rates) requires manually annotated defect boundary masks. Without pixel ground truth, DefectVision displays raw extracted anomaly area percentages and multi-algorithm consensus overlap.
                </div>
              </div>

              <div style={{ background: 'rgba(0, 0, 0, 0.25)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                <div style={{ fontWeight: 600, color: '#a5b4fc', fontSize: '0.85rem', marginBottom: '0.4rem' }}>
                  2. Image-Level Classification (MobileNetV2 Deep Learning)
                </div>
                <div style={{ fontSize: '0.8rem', color: '#e2e8f0', marginBottom: '0.4rem' }}>
                  <strong>Requires:</strong> Ground-Truth Defect Category Labels (y ∈ {'{crazing, inclusion, ...}'})
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                  Evaluating classification performance (Top-1 Accuracy, Macro Precision/Recall, Macro F1, Confusion Matrix) requires known class labels (as evaluated offline on the 360-image held-out test split). For live uploaded images without labels, the dashboard displays predicted softmax probabilities without fabricating accuracy.
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* DEEP LEARNING (MOBILENETV2) CLASSIFICATION MODULE */}
      <section ref={mlResultRef} className="glass-card" style={{ border: '1px solid rgba(129, 140, 248, 0.45)', marginBottom: '2.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <h2 className="section-heading" style={{ margin: 0, color: '#a5b4fc' }}>
              <Cpu size={22} style={{ color: '#818cf8' }} />
              Deep Learning Results Panel (MobileNetV2)
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
              Image-level defect taxonomy classification via fine-tuned MobileNetV2 transfer learning (POST /api/ml/classify)
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {mlResult && !mlLoading && (
              <span style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', fontFamily: 'monospace', background: 'rgba(6, 182, 212, 0.1)', padding: '0.3rem 0.6rem', borderRadius: '6px', border: '1px solid rgba(6, 182, 212, 0.25)' }}>
                Runtime: {mlResult.inference_runtime_ms} ms
              </span>
            )}
            <button 
              disabled={!isImageUploaded || mlLoading}
              onClick={executeDeepLearning}
              className="btn-action"
              style={{ 
                fontSize: '0.85rem', 
                borderColor: '#818cf8', 
                color: isImageUploaded ? '#c7d2fe' : 'var(--text-dim)',
                background: mlLoading ? 'rgba(99, 102, 241, 0.25)' : 'rgba(99, 102, 241, 0.12)'
              }}
            >
              <Cpu size={14} className={mlLoading ? 'animate-spin' : ''} style={{ color: isImageUploaded ? '#818cf8' : 'var(--text-dim)' }} />
              {mlLoading ? 'Classifying Defect...' : mlResult ? 'Re-Run Deep Learning' : '[Run Deep Learning]'}
            </button>
          </div>
        </div>

        {/* 1. LOADING STATE */}
        {mlLoading && (
          <div style={{
            padding: '2.5rem 1.5rem',
            textAlign: 'center',
            background: 'rgba(99, 102, 241, 0.05)',
            border: '1px solid rgba(129, 140, 248, 0.25)',
            borderRadius: '10px',
            marginBottom: '1rem'
          }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '56px', height: '56px', borderRadius: '50%', background: 'rgba(99, 102, 241, 0.15)', marginBottom: '1rem' }}>
              <Cpu size={28} className="animate-spin" style={{ color: '#818cf8' }} />
            </div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: '#e0e7ff', marginBottom: '0.4rem' }}>
              Running MobileNetV2 Defect Classification...
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '480px', margin: '0 auto 1.25rem auto', lineHeight: '1.4' }}>
              Streaming image bytes to <code>POST /api/ml/classify</code>, applying 224×224 3-channel RGB preprocessing, and computing 6-class softmax probabilities.
            </p>
            <div style={{ maxWidth: '320px', height: '4px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '2px', margin: '0 auto', overflow: 'hidden' }}>
              <div style={{ width: '60%', height: '100%', background: 'linear-gradient(90deg, #6366f1, #38bdf8)', borderRadius: '2px', animation: 'pulse 1.5s infinite' }} />
            </div>
          </div>
        )}

        {/* 2. ERROR & RETRY STATE */}
        {mlError && !mlLoading && (
          <div style={{
            background: 'rgba(244, 63, 94, 0.1)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: '10px',
            padding: '1.5rem',
            marginBottom: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            textAlign: 'center'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#fb7185', fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.5rem' }}>
              <AlertCircle size={20} />
              Deep Learning Classification Failed
            </div>
            <p style={{ color: '#fca5a5', fontSize: '0.85rem', maxWidth: '520px', margin: '0 0 1rem 0' }}>
              {mlError}
            </p>
            <button 
              onClick={executeDeepLearning}
              disabled={!isImageUploaded || mlLoading}
              className="btn-action"
              style={{
                borderColor: '#fb7185',
                background: 'rgba(244, 63, 94, 0.15)',
                color: '#fecdd3',
                fontSize: '0.825rem',
                padding: '0.45rem 1rem'
              }}
            >
              <RefreshCw size={14} className={mlLoading ? 'animate-spin' : ''} />
              Retry Classification
            </button>
          </div>
        )}

        {/* 3. SUCCESS / RESULTS STATE */}
        {mlResult && !mlLoading && (
          <div>
            {/* Top Category Highlight Header */}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              marginBottom: '1.25rem', 
              flexWrap: 'wrap', 
              gap: '0.75rem',
              padding: '0.85rem 1rem',
              background: 'rgba(16, 185, 129, 0.08)',
              border: '1px solid rgba(52, 211, 153, 0.25)',
              borderRadius: '8px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <CheckCircle2 size={20} style={{ color: '#34d399' }} />
                <div>
                  <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', display: 'block' }}>
                    Predicted Defect Category
                  </span>
                  <span style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f3f4f6' }}>
                    {mlResult.predicted_class.toUpperCase().replace('_', ' ')}
                  </span>
                  <span style={{ marginLeft: '0.5rem', fontSize: '0.85rem', color: '#a5b4fc', fontFamily: 'monospace', fontWeight: 600 }}>
                    ({mlResult.short_code})
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', display: 'block' }}>
                    Top Confidence
                  </span>
                  <span style={{ fontSize: '1.1rem', fontWeight: 700, color: '#34d399', fontFamily: 'monospace' }}>
                    {mlResult.confidence_percentage}
                  </span>
                </div>
              </div>
            </div>

            {/* Overview Key Metrics Grid */}
            <div className="meta-grid" style={{ marginBottom: '1.5rem' }}>
              <div className="meta-card">
                <div className="meta-label">Predicted Defect Category</div>
                <div className="meta-value" style={{ color: '#a5b4fc', textTransform: 'capitalize' }}>
                  {mlResult.predicted_class.replace('_', ' ')} ({mlResult.short_code})
                </div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Classification Confidence</div>
                <div className="meta-value" style={{ color: 'var(--accent-emerald)' }}>
                  {mlResult.confidence_percentage}
                </div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Inference Runtime</div>
                <div className="meta-value" style={{ color: 'var(--accent-amber)' }}>
                  {mlResult.inference_runtime_ms} ms
                </div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Output Format</div>
                <div className="meta-value" style={{ fontSize: '0.9rem', color: '#93c5fd' }}>
                  Whole-Image Categorical
                </div>
              </div>
            </div>

            {/* SIX-CLASS SOFTMAX PROBABILITY DISTRIBUTION BREAKDOWN */}
            <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '1.25rem', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.08)', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f3f4f6', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Activity size={18} style={{ color: '#818cf8' }} />
                  Six-Class Softmax Probability Distribution
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Sum of categorical scores = 100.0%
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                {Object.entries(mlResult.scores)
                  .sort(([, a], [, b]) => b - a)
                  .map(([className, probScore]) => {
                    const isTop = className === mlResult.predicted_class;
                    const pct = (probScore * 100).toFixed(2);
                    return (
                      <div key={className}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.825rem', marginBottom: '0.35rem' }}>
                          <span style={{ 
                            fontWeight: isTop ? 700 : 500, 
                            color: isTop ? '#c7d2fe' : 'var(--text-muted)',
                            textTransform: 'capitalize',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem'
                          }}>
                            {className.replace('_', ' ')}
                            {isTop && (
                              <span style={{ 
                                fontSize: '0.7rem', 
                                padding: '0.1rem 0.4rem', 
                                borderRadius: '4px', 
                                background: 'rgba(99, 102, 241, 0.25)', 
                                color: '#a5b4fc',
                                border: '1px solid rgba(99, 102, 241, 0.4)'
                              }}>
                                Top Match
                              </span>
                            )}
                          </span>
                          <span style={{ fontFamily: 'monospace', color: isTop ? '#34d399' : 'var(--text-dim)', fontSize: '0.85rem' }}>
                            {pct}% <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>({probScore.toFixed(4)})</span>
                          </span>
                        </div>
                        <div style={{ width: '100%', height: '8px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                          <div style={{
                            width: `${Math.max(Number(pct), 0.5)}%`,
                            height: '100%',
                            background: isTop 
                              ? 'linear-gradient(90deg, #6366f1, #34d399)' 
                              : 'rgba(148, 163, 184, 0.35)',
                            borderRadius: '4px',
                            transition: 'width 0.4s ease'
                          }} />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* EXPLICIT CLASSIFICATION SCOPE NOTICE: NOT A PIXEL MASK */}
            <div style={{ 
              padding: '0.85rem 1.1rem', 
              background: 'rgba(255, 255, 255, 0.03)', 
              borderRadius: '8px', 
              border: '1px solid rgba(255, 255, 255, 0.08)', 
              fontSize: '0.8rem', 
              color: 'var(--text-muted)',
              lineHeight: '1.45'
            }}>
              <strong style={{ color: '#c7d2fe' }}>ℹ️ Scope & Metric Clarification:</strong> Deep learning inference evaluates image-level defect likelihoods across the six NEU taxonomy classes. Unlike the classical CV pipelines below (Spatial, Edge/Shape, Frequency, Texture) and the 2+ Pipeline Consensus Engine, <strong>deep learning classification produces categorical confidence scores and does not generate pixel-level defect segmentation masks or boundary contour overlays.</strong>
            </div>
          </div>
        )}

        {/* 4. IDLE / UNEXECUTED STATE */}
        {!mlResult && !mlLoading && !mlError && (
          <div style={{
            padding: '2.5rem 1.5rem',
            textAlign: 'center',
            color: 'var(--text-dim)',
            fontSize: '0.9rem',
            border: '1px dashed rgba(255, 255, 255, 0.1)',
            borderRadius: '8px'
          }}>
            <Cpu size={32} style={{ color: 'var(--text-muted)', margin: '0 auto 0.75rem auto', opacity: 0.6 }} />
            <div style={{ color: '#e0e7ff', fontWeight: 500, marginBottom: '0.35rem' }}>
              Deep Learning Classifier Ready
            </div>
            <p style={{ maxWidth: '440px', margin: '0 auto', fontSize: '0.825rem', color: 'var(--text-muted)' }}>
              {isImageUploaded 
                ? 'Click [Run Deep Learning] above to identify the defect category and calculate softmax scores across all six classes.'
                : 'Upload or load a surface inspection target image to execute MobileNetV2 classification.'}
            </p>
          </div>
        )}
      </section>

      {/* FOUR INDEPENDENT PIPELINE SECTIONS */}
      <h2 className="section-heading" style={{ marginTop: '2.5rem', marginBottom: '1.25rem' }}>
        <Layers size={22} style={{ color: 'var(--primary)' }} />
        Defect Inspection Pipeline Modules
      </h2>

      {/* 1. SPATIAL / INTENSITY SECTION */}
      <section className="glass-card" style={{ border: '1px solid rgba(59, 130, 246, 0.35)', marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <div>
            <h2 className="section-heading" style={{ margin: 0, color: '#60a5fa' }}>
              <Zap size={20} />
              Spatial / Intensity
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
              Intensity thresholding, morphological filtering, and spatial component analysis
            </p>
          </div>

          <button 
            disabled={!isImageUploaded || activePipelineLoading !== null}
            onClick={() => executeSinglePipeline('spatial', 'Spatial')}
            className="btn-primary"
            style={{ fontSize: '0.85rem' }}
          >
            <Play size={14} className={activePipelineLoading === 'Spatial' ? 'animate-spin' : ''} />
            {activePipelineLoading === 'Spatial' ? 'Running Spatial...' : '[Run Spatial]'}
          </button>
        </div>

        {/* Spatial Result Container */}
        {spatialResult ? (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{
                background: spatialResult.anomaly_detected ? 'rgba(244, 63, 94, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                color: spatialResult.anomaly_detected ? '#fb7185' : '#34d399',
                border: `1px solid ${spatialResult.anomaly_detected ? 'rgba(244, 63, 94, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
                padding: '0.4rem 0.8rem',
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: 600,
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem'
              }}>
                {spatialResult.anomaly_detected ? <ShieldAlert size={16} /> : <ShieldCheck size={16} />}
                {spatialResult.anomaly_detected ? 'ANOMALY DETECTED' : 'CLEAN'}
              </div>

              <span style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', fontFamily: 'monospace' }}>
                Runtime: {spatialResult.processing_time} ms
              </span>
            </div>

            <div className="meta-grid" style={{ marginBottom: '1.25rem' }}>
              <div className="meta-card">
                <div className="meta-label">Anomaly Status</div>
                <div className="meta-value" style={{ color: spatialResult.anomaly_detected ? '#fb7185' : '#34d399' }}>
                  {spatialResult.anomaly_detected ? 'ANOMALY DETECTED' : 'CLEAN'}
                </div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Anomaly Area</div>
                <div className="meta-value">{spatialResult.anomaly_area.toLocaleString()} px</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Area Percentage</div>
                <div className="meta-value">{spatialResult.anomaly_area_percentage}%</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Region Count</div>
                <div className="meta-value">{spatialResult.region_count}</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Runtime</div>
                <div className="meta-value" style={{ color: 'var(--accent-amber)' }}>{spatialResult.processing_time} ms</div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Enhanced Image</div>
                <img src={spatialResult.enhanced_image} alt="Enhanced" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Threshold Image</div>
                <img src={spatialResult.threshold_image} alt="Threshold" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Final Mask</div>
                <img src={spatialResult.final_mask} alt="Final Mask" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Overlay</div>
                <img src={spatialResult.overlay} alt="Overlay" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
            </div>
          </div>
        ) : (
          <div style={{
            padding: '2rem',
            textAlign: 'center',
            color: 'var(--text-dim)',
            fontSize: '0.9rem',
            border: '1px dashed rgba(255, 255, 255, 0.1)',
            borderRadius: '8px'
          }}>
            Click [Run Spatial] above to execute intensity thresholding defect detection
          </div>
        )}
      </section>

      {/* 2. EDGE / SHAPE SECTION */}
      <section className="glass-card" style={{ border: '1px solid rgba(6, 182, 212, 0.35)', marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <div>
            <h2 className="section-heading" style={{ margin: 0, color: 'var(--accent-cyan)' }}>
              <Compass size={20} />
              Edge / Shape
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
              Canny edge extraction, boundary contour metrics, and geometric shape descriptors
            </p>
          </div>

          <button 
            disabled={!isImageUploaded || activePipelineLoading !== null}
            onClick={() => executeSinglePipeline('edge-shape', 'Edge/Shape')}
            className="btn-action"
            style={{ fontSize: '0.85rem', borderColor: 'var(--accent-cyan)' }}
          >
            <Sliders size={14} className={activePipelineLoading === 'Edge/Shape' ? 'animate-spin' : ''} style={{ color: 'var(--accent-cyan)' }} />
            {activePipelineLoading === 'Edge/Shape' ? 'Running Edge/Shape...' : '[Run Edge/Shape]'}
          </button>
        </div>

        {/* Edge / Shape Result Container */}
        {edgeShapeResult ? (
          <div>
            <div className="meta-grid" style={{ marginBottom: '1.25rem' }}>
              <div className="meta-card">
                <div className="meta-label">Area</div>
                <div className="meta-value">{edgeShapeResult.anomaly_area.toLocaleString()} px</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Region Count</div>
                <div className="meta-value">{edgeShapeResult.region_count}</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Runtime</div>
                <div className="meta-value" style={{ color: 'var(--accent-amber)' }}>{edgeShapeResult.processing_time} ms</div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Edge Map</div>
                <img src={edgeShapeResult.edge_map} alt="Edge Map" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Contour Visualization</div>
                <img src={edgeShapeResult.contour_visualization} alt="Contour Vis" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Anomaly Mask</div>
                <img src={edgeShapeResult.anomaly_mask} alt="Anomaly Mask" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Overlay</div>
                <img src={edgeShapeResult.overlay} alt="Overlay" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
            </div>

            {edgeShapeResult.shape_features.length > 0 && (
              <div>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f3f4f6', marginBottom: '0.5rem' }}>
                  Extracted Shape Features Table
                </h3>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                    <thead>
                      <tr style={{ background: 'rgba(255, 255, 255, 0.05)', borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
                        <th style={{ padding: '0.5rem', textAlign: 'left' }}>Region</th>
                        <th style={{ padding: '0.5rem', textAlign: 'left' }}>Area (px)</th>
                        <th style={{ padding: '0.5rem', textAlign: 'left' }}>Perimeter (px)</th>
                        <th style={{ padding: '0.5rem', textAlign: 'left' }}>W × H</th>
                        <th style={{ padding: '0.5rem', textAlign: 'left' }}>Aspect Ratio</th>
                        <th style={{ padding: '0.5rem', textAlign: 'left' }}>Circularity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {edgeShapeResult.shape_features.map((feat) => (
                        <tr key={feat.region_id} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                          <td style={{ padding: '0.5rem', color: 'var(--accent-cyan)', fontWeight: 600 }}>#{feat.region_id}</td>
                          <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{feat.area.toLocaleString()}</td>
                          <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{feat.perimeter.toLocaleString()}</td>
                          <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{feat.width}×{feat.height}</td>
                          <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{feat.aspect_ratio}</td>
                          <td style={{ padding: '0.5rem', fontFamily: 'monospace', color: 'var(--accent-emerald)' }}>{feat.circularity}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{
            padding: '2rem',
            textAlign: 'center',
            color: 'var(--text-dim)',
            fontSize: '0.9rem',
            border: '1px dashed rgba(255, 255, 255, 0.1)',
            borderRadius: '8px'
          }}>
            Click [Run Edge/Shape] above to execute Canny boundary contour extraction
          </div>
        )}
      </section>

      {/* 3. FREQUENCY / FFT SECTION */}
      <section className="glass-card" style={{ border: '1px solid rgba(245, 158, 11, 0.35)', marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <div>
            <h2 className="section-heading" style={{ margin: 0, color: 'var(--accent-amber)' }}>
              <ActivityIcon size={20} />
              Frequency / FFT
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
              2D Fourier Transform spectral energy decomposition and high-pass frequency filtering
            </p>
          </div>

          <button 
            disabled={!isImageUploaded || activePipelineLoading !== null}
            onClick={() => executeSinglePipeline('frequency', 'FFT')}
            className="btn-action"
            style={{ fontSize: '0.85rem', borderColor: 'var(--accent-amber)' }}
          >
            <Grid size={14} className={activePipelineLoading === 'FFT' ? 'animate-spin' : ''} style={{ color: 'var(--accent-amber)' }} />
            {activePipelineLoading === 'FFT' ? 'Running FFT...' : '[Run FFT]'}
          </button>
        </div>

        {/* Frequency Result Container */}
        {frequencyResult ? (
          <div>
            <div className="meta-grid" style={{ marginBottom: '1.25rem' }}>
              <div className="meta-card">
                <div className="meta-label">Total Frequency Energy</div>
                <div className="meta-value">{frequencyResult.total_frequency_energy.toExponential(2)}</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Low-Freq Energy</div>
                <div className="meta-value">{frequencyResult.low_frequency_energy.toExponential(2)}</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">High-Freq Energy</div>
                <div className="meta-value">{frequencyResult.high_frequency_energy.toExponential(2)}</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">High/Low Ratio</div>
                <div className="meta-value" style={{ color: 'var(--accent-cyan)' }}>
                  {frequencyResult.high_to_low_frequency_ratio}
                </div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Area</div>
                <div className="meta-value">{frequencyResult.anomaly_area.toLocaleString()} px</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Runtime</div>
                <div className="meta-value" style={{ color: 'var(--accent-amber)' }}>{frequencyResult.processing_time} ms</div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>FFT Spectrum</div>
                <img src={frequencyResult.fft_spectrum} alt="FFT Spectrum" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Filtered Frequency Rep</div>
                <img src={frequencyResult.filtered_frequency_representation} alt="Filtered Rep" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Reconstructed Image</div>
                <img src={frequencyResult.reconstructed_image} alt="Reconstructed" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Anomaly Mask</div>
                <img src={frequencyResult.anomaly_mask} alt="Anomaly Mask" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Overlay</div>
                <img src={frequencyResult.overlay} alt="Overlay" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
            </div>
          </div>
        ) : (
          <div style={{
            padding: '2rem',
            textAlign: 'center',
            color: 'var(--text-dim)',
            fontSize: '0.9rem',
            border: '1px dashed rgba(255, 255, 255, 0.1)',
            borderRadius: '8px'
          }}>
            Click [Run FFT] above to execute 2D Fourier spectral domain filtering
          </div>
        )}
      </section>

      {/* 4. TEXTURE / GABOR SECTION */}
      <section className="glass-card" style={{ border: '1px solid rgba(16, 185, 129, 0.35)', marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <div>
            <h2 className="section-heading" style={{ margin: 0, color: 'var(--accent-emerald)' }}>
              <Maximize size={20} />
              Texture / Gabor
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
              Multi-orientation Gabor filter bank (0°, 45°, 90°, 135°) and GLCM statistical response
            </p>
          </div>

          <button 
            disabled={!isImageUploaded || activePipelineLoading !== null}
            onClick={() => executeSinglePipeline('texture', 'Texture/Gabor')}
            className="btn-action"
            style={{ fontSize: '0.85rem', borderColor: 'var(--accent-emerald)' }}
          >
            <Maximize2 size={14} className={activePipelineLoading === 'Texture/Gabor' ? 'animate-spin' : ''} style={{ color: 'var(--accent-emerald)' }} />
            {activePipelineLoading === 'Texture/Gabor' ? 'Running Texture...' : '[Run Texture/Gabor]'}
          </button>
        </div>

        {/* Texture Result Container */}
        {textureResult ? (
          <div>
            <div className="meta-grid" style={{ marginBottom: '1.25rem' }}>
              <div className="meta-card">
                <div className="meta-label">Mean Response</div>
                <div className="meta-value">{textureResult.texture_statistics.mean_response}</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Variance</div>
                <div className="meta-value">{textureResult.texture_statistics.variance.toLocaleString()}</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Energy</div>
                <div className="meta-value">{textureResult.texture_statistics.energy.toLocaleString()}</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Contrast</div>
                <div className="meta-value">{textureResult.texture_statistics.contrast}</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Entropy</div>
                <div className="meta-value" style={{ color: 'var(--accent-emerald)' }}>{textureResult.texture_statistics.entropy}</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Area</div>
                <div className="meta-value">{textureResult.anomaly_area.toLocaleString()} px</div>
              </div>
              <div className="meta-card">
                <div className="meta-label">Runtime</div>
                <div className="meta-value" style={{ color: 'var(--accent-amber)' }}>{textureResult.processing_time} ms</div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Gabor 0°</div>
                <img src={textureResult.gabor_0_response} alt="Gabor 0 deg" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Gabor 45°</div>
                <img src={textureResult.gabor_45_response} alt="Gabor 45 deg" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Gabor 90°</div>
                <img src={textureResult.gabor_90_response} alt="Gabor 90 deg" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Gabor 135°</div>
                <img src={textureResult.gabor_135_response} alt="Gabor 135 deg" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Combined Response</div>
                <img src={textureResult.combined_response} alt="Combined Response" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Anomaly Mask</div>
                <img src={textureResult.anomaly_mask} alt="Anomaly Mask" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
              <div className="meta-card" style={{ padding: '0.75rem' }}>
                <div className="meta-label" style={{ marginBottom: '0.5rem' }}>Overlay</div>
                <img src={textureResult.overlay} alt="Overlay" style={{ width: '100%', borderRadius: '6px', border: '1px solid rgba(255, 255, 255, 0.1)' }} />
              </div>
            </div>
          </div>
        ) : (
          <div style={{
            padding: '2rem',
            textAlign: 'center',
            color: 'var(--text-dim)',
            fontSize: '0.9rem',
            border: '1px dashed rgba(255, 255, 255, 0.1)',
            borderRadius: '8px'
          }}>
            Click [Run Texture/Gabor] above to execute Gabor filter bank micro-texture extraction
          </div>
        )}
      </section>


    </div>
  );
}

export default App;
