# Frontend Architecture - Qualcomm-Aligned Edge AI Console

## Decision

Use **React + Vite + TypeScript** for the frontend.

This project should feel like an engineering console for Snapdragon edge AI deployment, not a marketing page and not a notebook dashboard. The UI should make the Qualcomm workflow visible:

1. Compress model with AIMET.
2. Export/package model for ONNX or AIMET encodings.
3. Compile for QNN/ONNX Runtime targets.
4. Profile on real devices.
5. Inspect latency, memory, CPU fallback, and accuracy deltas.
6. Tune the hybrid router threshold against latency, quality, and cost.

## Why This Fits Qualcomm

Qualcomm AI Hub Workbench documents compile, profile, inference, quantization, device, and job-management workflows as first-class concepts. The UI should mirror those operational concepts instead of hiding them behind a generic dashboard.

Qualcomm's AI Engine Direct SDK emphasizes targeting CPU, GPU, and Hexagon NPU through unified lower-level APIs, with workload delegation from ONNX Runtime or TensorFlow Lite. The frontend should therefore show accelerator-aware information: target runtime, device, NPU/HTP path, fallback ops, memory budget, and profile latency.

Snapdragon X Elite/X2 product material emphasizes on-device AI, efficiency, NPU TOPS, memory bandwidth, privacy, and chip-to-cloud experiences. The UI should make those tradeoffs explicit: what stays on-device, what goes cloud, what it costs, and why the router made that decision.

## Stack

```text
React + Vite + TypeScript
Tailwind CSS
shadcn/ui-style components
TanStack Query
Plotly.js
Lucide React
FastAPI backend
```

## Why Not Next.js

Next.js is excellent when SEO, server rendering, auth-heavy product flows, and hosted page routing are core requirements. This app is a technical console driven by authenticated/local APIs, polling, charts, and live ML job state. Vite is faster and simpler for that shape.

## Why TypeScript

TypeScript is valuable here because the UI consumes structured ML/backend objects:

- `BenchmarkResult`
- `AIHubJob`
- `RouteDecision`
- `ThresholdSweepPoint`
- `CompressionStage`
- `HardwareTarget`

Those contracts should be explicit. This avoids the common ML-demo failure mode where the frontend silently breaks because a Python response changed shape.

## Product Views

### 1. System Overview

Purpose: senior-level one-screen summary.

Shows:
- Latest verified benchmark run
- Compression ratio
- Accuracy/WER/perplexity delta
- AI Hub profile latency
- Cloud avoidance rate
- Router p95 latency
- Cost per 1,000 queries

### 2. AIMET Pipeline Explorer

Purpose: prove understanding of the Qualcomm quantization workflow.

Flow:

```text
FP32 -> BN Fold -> CLE -> ReLU6 Replacement -> AdaRound -> ONNX/AIMET Package -> AI Hub Compile -> AI Hub Profile
```

Each stage displays:
- Input/output artifact
- Metrics captured at that stage
- Failure modes
- Calibration notes

### 3. Compression Benchmark Matrix

Purpose: compare model families and precision choices.

Rows:
- MobileNetV2
- Whisper-tiny
- Phi-3-mini

Columns:
- FP32 metric
- W8/A8 metric
- W4/A8 metric
- size reduction
- latency
- target runtime
- CPU fallback count
- verification status

### 4. AI Hub Jobs

Purpose: expose real deployment operations.

Shows:
- compile job ID
- profile job ID
- device
- target runtime
- status
- output artifact
- latency
- fallback ops

### 5. Hybrid Router Playground

Purpose: live demo of on-device/cloud decisioning.

Input: user query.

Output:
- decision: `on_device`, `on_device_with_retry`, or `cloud`
- complexity score
- confidence
- router latency
- estimated energy/cost
- rationale based on thresholds

### 6. Threshold Lab

Purpose: show operating-point reasoning.

Controls:
- complexity threshold slider
- moderate/retry threshold slider
- cloud model selector
- on-device latency estimate

Charts:
- false negative rate on complex queries
- cloud rate
- cost per 1,000 queries
- quality proxy
- latency distribution

## UI Tone

Use a dense engineering UI:

- compact tables
- stage timelines
- status chips
- latency bars
- small multiples
- dark-on-light or light-on-dark neutral palette
- minimal decoration

Avoid:

- landing-page hero fluff
- oversized marketing cards
- generic AI gradients
- fake benchmark numbers
- hiding unverified data

## Frontend Data Contracts

Keep frontend API types in `frontend/src/types/api.ts`. Backend response schemas should match these models.

Important rule: all demo data must be labeled `source: "demo"`. Verified benchmark data should use `source: "measured"` and include the run ID/artifact path.

## Sources

- Qualcomm AI Hub Workbench overview: https://workbench.aihub.qualcomm.com/docs/hub/index.html
- Qualcomm AI Hub compile docs: https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html
- Qualcomm AI Hub profile docs: https://workbench.aihub.qualcomm.com/docs/hub/profile_examples.html
- Qualcomm AI Hub quantization docs: https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html
- Qualcomm AI Engine Direct SDK: https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
- Snapdragon X Elite product page: https://www.qualcomm.com/laptops/products/snapdragon-x-elite
- Vite guide: https://vite.dev/guide/
- TanStack Query React docs: https://tanstack.com/query/latest/docs/framework/react/overview
- shadcn/ui Vite docs: https://ui.shadcn.com/docs/installation/vite
- Plotly React docs: https://plotly.com/javascript/react/
