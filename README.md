---
title: QualEgde
emoji: ⚡
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# EdgeAI Suite

Qualcomm-focused AI/ML portfolio project with two systems:

- **Q1 Compression Suite:** AIMET post-training quantization pipeline, ONNX export, and Qualcomm AI Hub compile/profile submission.
- **Q2 Hybrid Router:** Sub-10ms query router that chooses on-device, on-device-with-retry, or cloud inference paths.

The dashboard is a real web app: **React + Vite TypeScript frontend** backed by a **FastAPI Python API**. Streamlit is intentionally not used.

## Quick Start

```bash
cd edgeai-suite
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

Run the backend:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL and make sure `VITE_API_BASE_URL` points at `http://localhost:8000`.

## Frontend Stack

- React + Vite + TypeScript
- TanStack Query for polling benchmark and AI Hub job state
- shadcn/ui-style component primitives with Tailwind CSS
- Plotly.js for engineering/scientific charts
- Lucide React for tool icons

## Architecture

```text
React TypeScript frontend
  |
  | HTTP JSON
  v
FastAPI backend
  |-- q1_compression_suite: AIMET pipeline, ONNX export, AI Hub jobs
  |-- q2_hybrid_router: TF-IDF router, cost model, threshold sweep
```

## Benchmark Reporting Rule

Any number in a result table must come from a real run. Placeholder/sample data may appear only in demo endpoints and must be labeled as demo data.

## Results

No verified benchmark runs have been recorded yet.
