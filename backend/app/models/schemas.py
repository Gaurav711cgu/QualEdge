from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class HardwareTarget(BaseModel):
    device: str
    runtime: Literal["qnn_context_binary", "precompiled_qnn_onnx", "onnx", "tflite", "qnn_dlc"]
    accelerator: Literal["cpu", "gpu", "hexagon_npu", "mixed"]

class BenchmarkResult(BaseModel):
    id: str
    source: Literal["demo", "measured"]
    modelName: str
    family: Literal["vision", "audio", "language"]
    precision: Literal["fp32", "w8a8", "w4a8"]
    metricName: Literal["top1_accuracy", "wer", "perplexity"]
    metricValue: Optional[float]
    modelSizeMb: Optional[float]
    latencyMs: Optional[float]
    target: HardwareTarget
    cpuFallbackOps: List[str]
    verifiedAt: Optional[str]

class CompressionStage(BaseModel):
    name: Literal["fp32", "bn_fold", "cle", "relu6_replace", "adaround", "onnx_export", "aihub_compile", "aihub_profile"]
    status: Literal["pending", "running", "passed", "failed", "skipped"]
    artifactPath: Optional[str] = None
    notes: Optional[str] = None

class AIHubJob(BaseModel):
    id: str
    modelName: str
    device: str
    runtime: Literal["qnn_context_binary", "precompiled_qnn_onnx", "onnx", "tflite", "qnn_dlc"]
    compileJobId: Optional[str]
    profileJobId: Optional[str]
    status: Literal["queued", "running", "success", "failed"]
    latencyMs: Optional[float]
    cpuFallbackOps: List[str]

class RouterRequest(BaseModel):
    query: str
    pathway: Literal["tfidf", "modernbert"] = "tfidf"
    forceDegrade: bool = False

class RouterResult(BaseModel):
    decision: Literal["on_device", "on_device_with_retry", "cloud"]
    complexityScore: float
    confidence: float
    routerLatencyMs: float
    estimatedCloudCostUsd: float
    estimatedOnDeviceEnergyJ: float
    source: Literal["demo", "measured"]
    text: str
    cpuFallbackOps: List[str]
    device: str

class ThresholdSweepPoint(BaseModel):
    threshold: float
    falseNegativeRate: float
    cloudRate: float
    onDeviceRate: float
    estimatedCostPerThousand: float
    p95LatencyMs: float

class OverviewStats(BaseModel):
    compressionRatio: float
    accuracyDelta: float
    aiHubJobsCount: int
    routerP95LatencyMs: float
    cloudAvoidanceRate: float
    costPerThousand: float
    driftPsi: float
    driftStatus: str
