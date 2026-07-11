from fastapi import APIRouter, HTTPException
from backend.app.models.schemas import OverviewStats
from backend.app.core.state import routing_service

router = APIRouter(prefix="/overview", tags=["overview"])

@router.get("/stats", response_model=OverviewStats)
def get_overview_stats():
    try:
        stats = routing_service.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load overview statistics: {str(e)}")
