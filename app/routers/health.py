from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", version="0.1.0")
