"""
Pydantic Schemas for Complaint Closure, Feedback, and Audit History.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ComplaintResolveRequest(BaseModel):
    officer_name: Optional[str] = None
    officer_contact: Optional[str] = None
    remarks: Optional[str] = None
    resolution_photo_path: Optional[str] = None


class CitizenConfirmResolutionRequest(BaseModel):
    confirmed: bool = Field(..., description="True if problem resolved, False if problem still exists")
    reason: Optional[str] = Field(None, description="Reason if citizen indicates problem is not resolved")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Citizen rating from 1 to 5 stars")
    comments: Optional[str] = Field(None, description="Citizen feedback comments")


class ComplaintFeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Star rating from 1 to 5")
    comments: Optional[str] = None
    confirmed_resolved: bool = True


class FeedbackResponse(BaseModel):
    ticket_id: str
    status: str
    rating: Optional[int] = None
    comments: Optional[str] = None
    message: str


class ComplaintHistoryItem(BaseModel):
    id: int
    complaint_id: int
    ticket_id: str
    old_status: Optional[str] = None
    new_status: str
    changed_by: Optional[str] = None
    reason: Optional[str] = None
    extra_metadata: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ComplaintHistoryResponse(BaseModel):
    ticket_id: str
    current_status: str
    reopen_count: int = 0
    history: List[ComplaintHistoryItem] = []
