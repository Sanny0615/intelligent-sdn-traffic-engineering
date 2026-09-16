"""
Health check route handler.
"""

from fastapi import APIRouter
from src.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Check API and Simulation Status")
def get_health():
    """Returns status ok indicating API and backend simulation service are operational."""
    return HealthResponse(status="ok", version="1.0.0", simulation="active")
