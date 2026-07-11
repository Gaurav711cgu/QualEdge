from fastapi import APIRouter, HTTPException
from typing import List
from backend.app.models.schemas import BenchmarkResult, CompressionStage
from backend.app.core.state import comp_service

router = APIRouter(prefix="/compression", tags=["compression"])

@router.get("/benchmarks", response_model=List[BenchmarkResult])
def get_benchmarks():
    try:
        return comp_service.get_benchmarks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch benchmarks: {str(e)}")

@router.post("/run")
def trigger_compression_run(model_name: str, ood_calibration: bool = False):
    try:
        run_id = comp_service.trigger_run(model_name, ood_calibration)
        return {"run_id": run_id, "status": "started"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger compression pipeline: {str(e)}")

@router.get("/run/{run_id}/stages", response_model=List[CompressionStage])
def get_run_stages(run_id: str):
    stages = comp_service.get_run_stages(run_id)
    if not stages:
        raise HTTPException(status_code=404, detail="Compression run not found or has not started stages.")
    return stages
