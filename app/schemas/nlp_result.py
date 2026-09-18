"""
Schemas for Multilingual NLP Analysis, Language Detection & Clarification.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class NLPAnalysisResult(BaseModel):
    intent: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    language: str = "en"  # "en", "mr", "unclear"
    is_ambiguous: bool = False
    clarification_type: Optional[str] = None
    clarification_question: Optional[str] = None
    source: str = "keyword_heuristic"
