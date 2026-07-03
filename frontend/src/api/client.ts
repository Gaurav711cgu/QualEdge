import type {
  AIHubJob,
  BenchmarkResult,
  RouterResult,
  ThresholdSweepPoint,
  CompressionStage,
  OverviewStats,
} from "../types/api";

const API_BASE_URL = (import.meta as any).env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export function fetchOverviewStats(): Promise<OverviewStats> {
  return getJson<OverviewStats>("/api/overview/stats");
}

export function fetchBenchmarks(): Promise<BenchmarkResult[]> {
  return getJson<BenchmarkResult[]>("/api/compression/benchmarks");
}

export function triggerCompressionRun(modelName: string, oodCalibration: boolean = false): Promise<{ run_id: string; status: string }> {
  return postJson<{ run_id: string; status: string }>(`/api/compression/run?model_name=${modelName}&ood_calibration=${oodCalibration}`, {});
}

export function fetchRunStages(runId: string): Promise<CompressionStage[]> {
  return getJson<CompressionStage[]>(`/api/compression/run/${runId}/stages`);
}

export function fetchAIHubJobs(): Promise<AIHubJob[]> {
  return getJson<AIHubJob[]>("/api/aihub/jobs");
}

export function routeQuery(query: string, pathway: "tfidf" | "modernbert" = "tfidf", forceDegrade: boolean = false): Promise<RouterResult> {
  return postJson<RouterResult>("/api/router/route", { query, pathway, forceDegrade });
}

export function fetchThresholdSweep(): Promise<ThresholdSweepPoint[]> {
  return getJson<ThresholdSweepPoint[]>("/api/router/sweep");
}

