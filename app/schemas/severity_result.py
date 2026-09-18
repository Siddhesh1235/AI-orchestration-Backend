"""
Schemas for Severity, Emergency, and Priority evaluation results.
"""

from typing import Optional, List
from pydantic import BaseModel


class EmergencyResult(BaseModel):
    is_emergency: bool
    emergency_level: str = "NONE"  # "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    reason: Optional[str] = None


class SeverityResult(BaseModel):
    severity: str = "MEDIUM"  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    reason: Optional[str] = None
    safety_impact: str = "NORMAL"


class PriorityResult(BaseModel):
    priority: str = "P3"  # "P1" | "P2" | "P3" | "P4" (also maps to LOW/MEDIUM/HIGH for legacy)
    legacy_priority: str = "MEDIUM"
    reason: str
    is_automated: bool = True
