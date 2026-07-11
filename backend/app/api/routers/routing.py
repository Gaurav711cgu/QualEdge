from fastapi import APIRouter, HTTPException
from typing import List
from backend.app.models.schemas import RouterRequest, RouterResult, ThresholdSweepPoint
from backend.app.core.state import routing_service

router = APIRouter(prefix="/router", tags=["router"])

@router.post("/route", response_model=RouterResult)
def route_query(payload: RouterRequest):
    try:
        result = routing_service.route_query(
            query=payload.query, 
            pathway=payload.pathway, 
            force_degrade=payload.forceDegrade
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Router decision crash: {str(e)}")

@router.get("/sweep", response_model=List[ThresholdSweepPoint])
def get_threshold_sweep():
    try:
        return routing_service.get_sweep_points()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate threshold sweep matrix: {str(e)}")
