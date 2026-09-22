"""
Pydantic Schemas for Video Understanding and Ingestion Pipeline.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FrameAnalysisResult(BaseModel):
    frame_index: int
    timestamp_sec: float
    category: str
    confidence: float
    raw_label: Optional[str] = None
    is_valid: bool = True
    rejection_reason: Optional[str] = None


class VideoIngestionResult(BaseModel):
    category: str = "other"
    confidence: float = 0.0
    issue: str = "unknown"
    visual_status: Optional[str] = None  # "on", "off", or None
    confidence_level: str = "high"  # "high", "medium", "low"
    has_multiple_issues: bool = False
    candidate_issues: List[str] = Field(default_factory=list)
    transcript: str = ""
    detected_language: str = "mr"
    evidence_frame_path: Optional[str] = None
    video_path: Optional[str] = None
    frame_count: int = 0
    duration_sec: float = 0.0
    frames_analyzed: int = 0
    frames_rejected_blur: int = 0
    frames_rejected_noncivic: int = 0
    combined_scores: Dict[str, float] = Field(default_factory=dict)
    frame_details: List[FrameAnalysisResult] = Field(default_factory=list)
