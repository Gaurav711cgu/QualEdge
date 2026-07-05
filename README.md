---
title: QualEgde
emoji: ⚡
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# QualEdge: Qualcomm Snapdragon Edge AI Optimization Console

QualEdge is a production-grade ML engineering demonstration showcasing edge-device model optimization and low-latency hybrid query routing. This project is specifically architected to align with **Qualcomm's Snapdragon X Elite (Hexagon NPU)** hardware constraints, using **Qualcomm's AIMET SDK** patterns and **Qualcomm AI Hub** profiling pipelines.

---

## ⚡ Qualcomm Hardware & Architectural Alignment

To deliver high-performance edge execution, QualEdge implements core hardware-aware principles that respect the limits of the **Hexagon Tensor Processor (HTP)**:

### 1. Hexagon NPU-Aware Post-Training Quantization (PTQ)
We implement a simulated quantization workflow modeling Qualcomm's **AIMET (AI Model Efficiency Toolkit)** surgeries:
* **Folding Batch Normalization:** Folds BN layers into preceding convolutions to eliminate runtime scaling layers.
* **Cross-Layer Equalization (CLE):** Rescales weights across channel groups to narrow the range variance in depthwise separable convolutions (like MobileNetV2), mitigating INT8 quantization error without training.
* **ReLU6 to ReLU Surgery:** Replaces bounded activation functions (`ReLU6` / `HardSigmoid`) with unbounded equivalents (`ReLU`) to preserve scaling factors prior to quantization.
* **AdaRound (Adaptive Rounding):** Implements the Qualcomm AI Research ICML 2020 standard, optimizing weight rounding decisions per-layer by minimizing output reconstruction error instead of using naive nearest rounding.

### 2. HTP Energy & Silicon Telemetry Model
We employ an HTP-specific silicon power model to represent real hardware draws:
* **HTP Native Tensor Operations (INT8/INT4):** Executes low-power matrix multiply blocks directly in hardware silicon ($\approx 0.08\text{ Joules}$ per query).
* **Kryo CPU Fallback Penalty:** Detects layers lacking HTP operator support (e.g., `LayerNorm` or custom activation kernels) and applies a power penalty ($\approx 2.10\text{ Joules}$ per query) to simulate CPU instruction decodes.

### 3. Hybrid Edge-Cloud LLM Routing
Not all queries warrant 70B parameter cloud execution. QualEdge splits workloads dynamically:
* **Complexity Classifier:** A sub-1ms TF-IDF and Logistic Regression router trained to classify query difficulty.
* **Local Engine Verification:** Analyzes local edge model responses (Qwen-0.5B-Chat) for output degradation (empty responses or $>40\%$ repetition index).
* **AutoMix Cascading Fallback:** If local output fails quality checks or if prompt complexity is high, it smoothly escalates to a cloud API fallback (Gemini/Claude).

---

## 🔬 Scientific Grounding & Literature

QualEdge builds upon established scientific literature in model efficiency:

| Area | Inspired by Paper | Core Implementation |
| :--- | :--- | :--- |
| **Adaptive Rounding** | *Up or Down? Adaptive Rounding for Post-Training Quantization* (Nagel et al., ICML 2020) | Optimizes rounding per-layer to prevent representation collapse in 4-bit weights. |
| **Weight Equalization** | *Data-Free Quantization Through Weight Equalization and Bias Correction* (Nagel et al., ICCV 2019) | Cross-layer scaling before quantization to suppress outliers. |
| **Query Routing** | *Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing* (Ding et al., ICLR 2024) | Dual-tier classification to maximize cloud avoidance while preserving performance. |
| **Output Verification** | *AutoMix: Mix-of-Granularity LLMs for Efficient Inference* (Yue et al., ArXiv 2024) | Real-time verification of local outputs to trigger fallback retries. |

---

## 🛠️ Project Structure & Architecture

The workspace is organized to separate model optimization libraries, client Serving/API layers, and front-end telemetry dashboards:

```text
edgeai-suite/
├── q1_compression_suite/    # Project Q1: AIMET Quantization & AI Hub Compile
│   ├── compression/          # BN Folding, CLE, and AdaRound simulations
│   ├── deployment/           # Qualcomm AI Hub profile submission client
│   └── evaluation/           # Top-1, WER, and perplexity metric evaluation
├── q2_hybrid_router/        # Project Q2: Edge/Cloud Router & Classifier
│   ├── router/               # Logistic Regression complexity classifier & PSI drift
│   ├── inference/            # Local Qwen engine and cloud client mocks
│   └── evaluation/           # Threshold sweeps and Pareto tradeoff analysis
├── backend/                  # FastAPI serving layer (CORS, router, and telemetry APIs)
├── frontend/                 # React 19 + TypeScript + Vite console dashboard
└── tests/                    # 100% Passing Pytest unit and integration tests
```

---

## 🚀 Quick Start & Verification

### Prerequisites
* Python 3.10+
* Node.js 18+

### 1. Install & Test Suite
Verify the backend, pipeline, and router logic:
```bash
# Clone and setup env
git clone https://github.com/Gaurav711cgu/QualEdge.git
cd edgeai-suite

# Install dependencies
pip install -r requirements.txt

# Run all 13 unit/integration tests
PYTHONPATH=. pytest tests/ -v --tb=short
```

### 2. Start the Backend API
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### 3. Start the Frontend Console
```bash
cd frontend
npm install
npm run dev
```

---

## 📈 Telemetry Dashboard
The React frontend (packaged as static files or deployable via Vercel) provides real-time dials for:
* **AIMET Pipeline Explorer:** Simulated step-by-step PTQ effects with dynamic OOD (Out-of-Distribution) accuracy collapse visualization.
* **Pareto sweep charts:** Dynamic interactive sweeps comparing latency, cost, and classification thresholds.
* **Live Router Playground:** Input real-time queries to see exact routing decisions, energy draws, and execution traces.
* **Academic Literature Board:** Direct references and link-outs to key Qualcomm AI Research papers.
