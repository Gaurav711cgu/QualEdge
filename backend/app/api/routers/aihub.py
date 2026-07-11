from fastapi import APIRouter, HTTPException
from typing import List
from backend.app.models.schemas import AIHubJob
from backend.app.core.state import comp_service

router = APIRouter(prefix="/aihub", tags=["aihub"])

@router.get("/jobs", response_model=List[AIHubJob])
def get_aihub_jobs():
    try:
        return comp_service.get_aihub_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load AI Hub job records: {str(e)}")
