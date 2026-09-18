"""
Schemas for Image Classifier & Evidence Verification results.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ImagePredictionResult(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    raw_label: Optional[str] = None
    model_mode: str
    is_pretrained: bool = True
    is_confident: bool = True
    evidence_valid: bool = True
    verification_message: Optional[str] = None
