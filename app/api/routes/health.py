"""
Health Check & Telemetry Endpoints for PCMC Sarathi AI.
"""

from pathlib import Path
from fastapi import APIRouter
from app.config.settings import settings
from app.agents.image_agent import image_agent

router = APIRouter(tags=["System Health"])


@router.get("/health")
def health_check():
    """
    Returns system status, active ML weights status, and provider telemetry.
    """
    custom_exists = Path(settings.YOLO_MODEL_PATH).exists() and Path(settings.YOLO_MODEL_PATH).stat().st_size > 50000

    return {
        "status": "ok",
        "service": "Ward Mitra Orchestrator",
        "version": settings.VERSION,
        "vision_model": {
            "mode": image_agent.model_mode,
            "custom_best_pt_exists": custom_exists,
            "using_pretrained": not custom_exists,
            "pretrained_model": "yolo11n-cls.pt"
        },
        "llm_providers": {
            "tier_2_ollama": settings.OLLAMA_BASE_URL,
            "tier_3_openai_configured": bool(settings.OPENAI_API_KEY)
        }
    }
