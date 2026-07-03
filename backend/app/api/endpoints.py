from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any

from backend.app.models.schemas import (
    OverviewStats, BenchmarkResult, CompressionStage,
    AIHubJob, RouterRequest, RouterResult, ThresholdSweepPoint
)
from backend.app.services.compression_service import CompressionService
from backend.app.services.routing_service import RoutingService

# Router instance
router = APIRouter(prefix="/api")

# Singleton service instances (mock container setup)
_comp_service = CompressionService()
_routing_service = RoutingService(_comp_service)

def get_comp_service() -> CompressionService:
    return _comp_service

def get_routing_service() -> RoutingService:
    return _routing_service

@router.get("/overview/stats", response_model=OverviewStats)
def get_overview_stats(routing_service: RoutingService = Depends(get_routing_service)):
    try:
        stats = routing_service.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load overview statistics: {str(e)}")

@router.get("/compression/benchmarks", response_model=List[BenchmarkResult])
def get_benchmarks(comp_service: CompressionService = Depends(get_comp_service)):
    try:
        return comp_service.get_benchmarks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch benchmarks: {str(e)}")

@router.post("/compression/run")
def trigger_compression_run(
    model_name: str, 
    ood_calibration: bool = False,
    comp_service: CompressionService = Depends(get_comp_service)
):
    try:
        run_id = comp_service.trigger_run(model_name, ood_calibration)
        return {"run_id": run_id, "status": "started"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger compression pipeline: {str(e)}")

@router.get("/compression/run/{run_id}/stages", response_model=List[CompressionStage])
def get_run_stages(run_id: str, comp_service: CompressionService = Depends(get_comp_service)):
    stages = comp_service.get_run_stages(run_id)
    if not stages:
        raise HTTPException(status_code=404, detail="Compression run not found or has not started stages.")
    return stages

@router.get("/aihub/jobs", response_model=List[AIHubJob])
def get_aihub_jobs(comp_service: CompressionService = Depends(get_comp_service)):
    try:
        return comp_service.get_aihub_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load AI Hub job records: {str(e)}")

@router.post("/router/route", response_model=RouterResult)
def route_query(
    payload: RouterRequest,
    routing_service: RoutingService = Depends(get_routing_service)
):
    try:
        result = routing_service.route_query(
            query=payload.query, 
            pathway=payload.pathway, 
            force_degrade=payload.forceDegrade
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Router decision crash: {str(e)}")

@router.get("/router/sweep", response_model=List[ThresholdSweepPoint])
def get_threshold_sweep(routing_service: RoutingService = Depends(get_routing_service)):
    try:
        return routing_service.get_sweep_points()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate threshold sweep matrix: {str(e)}")
