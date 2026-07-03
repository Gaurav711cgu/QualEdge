# AGENTS.md - edgeai-suite

## What This Repo Does

Two production-grade ML systems:
- Q1: AIMET quantization benchmark suite for MobileNetV2, Whisper-tiny, and Phi-3-mini.
- Q2: Hybrid on-device/cloud LLM router with latency, quality, and cost analysis.

## Directory Map

```text
edgeai-suite/
├── q1_compression_suite/
│   ├── compression/
│   ├── deployment/
│   ├── evaluation/
│   └── config/
├── q2_hybrid_router/
│   ├── router/
│   ├── inference/
│   ├── evaluation/
│   └── config/
├── backend/                # FastAPI app for dashboard/demo APIs
├── frontend/               # React + Vite TypeScript frontend
├── tests/
├── requirements.txt
└── README.md
```

## Hard Rules

1. Never touch `notebooks/`; they are manual experiment logs.
2. Never report a benchmark number you did not compute.
3. Never use `torch.cuda.is_available()` as a guard that silently falls back to CPU; raise a clear error instead.
4. Never call `model.train()` during evaluation; use `model.eval()` and `torch.no_grad()`.
5. No `import *`.
6. No fake CI badges or hardcoded passing status.
7. Keep dashboard/product UI in React + TypeScript. Do not add Streamlit.

## Test Command

```bash
pytest tests/ -v --tb=short
```

## Frontend/Backend Choice

Use React + Vite TypeScript for the frontend and FastAPI for the backend. FastAPI keeps the serving layer close to AIMET, PyTorch, ONNX Runtime, MLflow, and Qualcomm AI Hub, while React provides a proper product dashboard and live demo experience.
