export type DataSource = "demo" | "measured";

export type RouteDecision = "on_device" | "on_device_with_retry" | "cloud";

export type HardwareTarget = {
  device: string;
  runtime: "qnn_context_binary" | "precompiled_qnn_onnx" | "onnx" | "tflite" | "qnn_dlc";
  accelerator: "cpu" | "gpu" | "hexagon_npu" | "mixed";
};

export type BenchmarkResult = {
  id: string;
  source: DataSource;
  modelName: string;
  family: "vision" | "audio" | "language";
  precision: "fp32" | "w8a8" | "w4a8";
  metricName: "top1_accuracy" | "wer" | "perplexity";
  metricValue: number | null;
  modelSizeMb: number | null;
  latencyMs: number | null;
  target: HardwareTarget;
  cpuFallbackOps: string[];
  verifiedAt: string | null;
};

export type CompressionStage = {
  name: "fp32" | "bn_fold" | "cle" | "relu6_replace" | "adaround" | "onnx_export" | "aihub_compile" | "aihub_profile";
  status: "pending" | "running" | "passed" | "failed" | "skipped";
  artifactPath?: string;
  notes?: string;
};

export type AIHubJob = {
  id: string;
  modelName: string;
  device: string;
  runtime: HardwareTarget["runtime"];
  compileJobId: string | null;
  profileJobId: string | null;
  status: "queued" | "running" | "success" | "failed";
  latencyMs: number | null;
  cpuFallbackOps: string[];
};

export type RouterResult = {
  decision: RouteDecision;
  complexityScore: number;
  confidence: number;
  routerLatencyMs: number;
  estimatedCloudCostUsd: number;
  estimatedOnDeviceEnergyJ: number;
  source: DataSource;
  text: string;
  cpuFallbackOps: string[];
  device: string;
};

export type ThresholdSweepPoint = {
  threshold: number;
  falseNegativeRate: number;
  cloudRate: number;
  onDeviceRate: number;
  estimatedCostPerThousand: number;
  p95LatencyMs: number;
};

export type OverviewStats = {
  compressionRatio: number;
  accuracyDelta: number;
  aiHubJobsCount: number;
  routerP95LatencyMs: number;
  cloudAvoidanceRate: number;
  costPerThousand: number;
  driftPsi: number;
  driftStatus: string;
};
