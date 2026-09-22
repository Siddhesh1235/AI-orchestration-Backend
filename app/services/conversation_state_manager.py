"""
Conversation State Manager for WardMitra AI.
Provides deterministic, context-aware, multi-turn state management:
1. Tracks complaint_category, issue_type, location, description, visual_status.
2. Tracks image evidence and visual observations without topic derailment.
3. Tracks questions_asked, questions_answered, current_question, missing_fields.
4. Ensures an answered question is NEVER asked again.
5. Protects current_complaint_category against noisy model overrides.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.config.category_registry import normalize_category_key, get_category_info

logger = logging.getLogger("pcms.state_manager")


class ImageEvidenceState(BaseModel):
    uploaded: bool = False
    relevant: Optional[bool] = None
    detected_category: Optional[str] = None
    confidence: Optional[float] = None
    visual_status: Optional[str] = None
    has_multiple_issues: bool = False
    candidate_issues: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    raw_analysis: Dict[str, Any] = Field(default_factory=dict)


class VideoEvidenceState(BaseModel):
    uploaded: bool = False
    video_analyzed: bool = False
    detected_category: Optional[str] = None
    detected_issue: Optional[str] = None
    confidence: float = 0.0
    confidence_level: str = "high"  # "high", "medium", "low"
    visual_status: Optional[str] = None
    has_multiple_issues: bool = False
    candidate_issues: List[str] = Field(default_factory=list)
    user_confirmed: bool = False
    evidence_frame_path: Optional[str] = None
    video_path: Optional[str] = None


class ConversationHistoryState(BaseModel):
    current_question: Optional[str] = None
    asked_questions: List[str] = Field(default_factory=list)
    answered_questions: List[Dict[str, Any]] = Field(default_factory=list)


class ConversationState(BaseModel):
    session_id: str
    citizen_phone: Optional[str] = None
    complaint_category: Optional[str] = None
    issue_type: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    visual_status: Optional[str] = None  # e.g., "off", "on", "flickering"
    language: str = "en"  # "en", "mr", "hi", "hi-en"
    input_type: str = "text"  # "text", "voice", "image", "video"
    user_confirmed: bool = False  # Confirmation gate before collecting information
    image_uploaded: bool = False
    image_analysis: Dict[str, Any] = Field(default_factory=dict)
    image: ImageEvidenceState = Field(default_factory=ImageEvidenceState)
    video: VideoEvidenceState = Field(default_factory=VideoEvidenceState)
    conversation: ConversationHistoryState = Field(default_factory=ConversationHistoryState)
    questions_asked: List[str] = Field(default_factory=list)
    questions_answered: List[Dict[str, Any]] = Field(default_factory=list)
    current_question: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)
    complaint_status: str = "collecting_information"  # "collecting_information", "awaiting_confirmation", "ready_for_submission", "submitted"
    category_locked: bool = False
    category_confirmed: bool = False
    confirmed_category: Optional[str] = None
    category_source: Optional[str] = None  # "text", "button", "image", "video", "correction"
    text_category: Optional[str] = None
    text_confidence: float = 0.0
    image_category: Optional[str] = None
    image_confidence: float = 0.0
    pending_field: Optional[str] = None  # "category", "description", "location", "image", "registration_confirmation", "resolution_confirmation", "feedback", "image_category_confirmation", "cancel_confirmation"
    candidate_categories: List[str] = Field(default_factory=list)
    pending_clarification: Optional[Dict[str, Any]] = None
    pending_cancel_ticket: Optional[str] = None
    last_updated: float = Field(default_factory=time.time)

    def reset(self):
        """Resets complaint fields for fresh complaint session."""
        self.complaint_category = None
        self.category_locked = False
        self.category_confirmed = False
        self.confirmed_category = None
        self.category_source = None
        self.text_category = None
        self.text_confidence = 0.0
        self.image_category = None
        self.image_confidence = 0.0
        self.pending_field = None
        self.candidate_categories = []
        self.pending_cancel_ticket = None
        self.issue_type = None
        self.location = None
        self.description = None
        self.visual_status = None
        self.input_type = "text"
        self.user_confirmed = False
        self.image_uploaded = False
        self.image_analysis = None
        self.image = ImageEvidenceState()
        self.video = VideoEvidenceState()
        self.questions_asked = []
        self.questions_answered = []
        self.current_question = None
        self.missing_fields = []
        self.complaint_status = "idle"
        self.conversation = ConversationHistoryState()
        self.last_updated = time.time()

    def reset_after_registration(self):
        """
        Clears complaint-specific state after successful registration,
        preserving session_id, citizen_phone, and language.
        """
        self.complaint_category = None
        self.category_locked = False
        self.category_confirmed = False
        self.confirmed_category = None
        self.category_source = None
        self.text_category = None
        self.text_confidence = 0.0
        self.image_category = None
        self.image_confidence = 0.0
        self.pending_field = None
        self.candidate_categories = []
        self.pending_cancel_ticket = None
        self.issue_type = None
        self.location = None
        self.description = None
        self.visual_status = None
        self.input_type = "text"
        self.user_confirmed = False
        self.image_uploaded = False
        self.image_analysis = None
        self.image = ImageEvidenceState()
        self.video = VideoEvidenceState()
        self.questions_asked = []
        self.questions_answered = []
        self.current_question = None
        self.missing_fields = []
        self.complaint_status = "completed"
        self.conversation = ConversationHistoryState()
        self.last_updated = time.time()

    def on_video_received(self, video_result: Any):
        """Initializes state from analyzed video evidence."""
        self.input_type = "video"
        self.video.uploaded = True
        self.video.video_analyzed = True
        self.video.detected_category = getattr(video_result, "category", None)
        self.video.detected_issue = getattr(video_result, "issue", None)
        self.video.confidence = getattr(video_result, "confidence", 0.0)
        self.video.confidence_level = getattr(video_result, "confidence_level", "high")
        self.video.visual_status = getattr(video_result, "visual_status", None)
        self.video.has_multiple_issues = getattr(video_result, "has_multiple_issues", False)
        self.video.candidate_issues = getattr(video_result, "candidate_issues", [])
        self.video.evidence_frame_path = getattr(video_result, "evidence_frame_path", None)
        self.video.video_path = getattr(video_result, "video_path", None)
        self.user_confirmed = False
        self.complaint_status = "awaiting_confirmation"
        cat = getattr(video_result, "category", None)
        if cat and cat not in ["other", "unrelated"]:
            self.complaint_category = cat
        iss = getattr(video_result, "issue", None)
        if iss and iss != "unknown":
            self.issue_type = iss
        vstat = getattr(video_result, "visual_status", None)
        if vstat:
            self.visual_status = vstat
        self.last_updated = time.time()

    def on_video_confirmed_yes(self):
        """Advances state after user explicitly confirms video complaint."""
        self.user_confirmed = True
        self.video.user_confirmed = True
        self.complaint_status = "collecting_information"
        self.category_locked = True
        self.category_confirmed = True
        self.confirmed_category = self.complaint_category
        self.category_source = "video"
        self.record_question_answered("category_confirmation", "YES")
        if self.complaint_category:
            self.record_question_answered("category", self.complaint_category)
            if self.complaint_category == "streetlight":
                self.record_question_answered("home_vs_streetlight", "streetlight")
        if self.visual_status:
            self.record_question_answered("visual_status", f"{self.visual_status} (from video)")
        self.update_missing_fields()
        self.last_updated = time.time()

    def on_video_confirmed_no(self):
        """Resets complaint state after user declines video complaint."""
        self.reset()

    def is_field_provided(self, field_name: str) -> bool:
        """Checks if a field has valid information."""
        if field_name == "category" or field_name == "complaint_category":
            return bool(self.complaint_category and self.complaint_category != "other")
        if field_name == "issue_type":
            return bool(self.issue_type)
        if field_name == "visual_status":
            return bool(self.visual_status)
        if field_name == "location":
            # Check length and non-generic
            return bool(self.location and len(self.location.strip()) >= 3)
        if field_name == "description":
            return bool(self.description and len(self.description.strip()) >= 3)
        if field_name == "image":
            return bool(self.image_uploaded)
        return False

    def is_question_answered(self, question_id: str) -> bool:
        """Checks if question has already been marked as answered."""
        for record in self.questions_answered:
            if record.get("question") == question_id:
                return True
        return False

    def record_question_asked(self, question_id: str, question_text: str = ""):
        """Records that a specific question has been asked to the citizen."""
        self.current_question = question_id
        if question_id not in self.questions_asked:
            self.questions_asked.append(question_id)
        if question_id not in self.conversation.asked_questions:
            self.conversation.asked_questions.append(question_id)
        self.conversation.current_question = question_id
        self.last_updated = time.time()

    def record_question_answered(self, question_id: str, answer_text: str):
        """Marks a question as answered with the user's response."""
        record = {
            "question": question_id,
            "answer": answer_text,
            "timestamp": time.time()
        }
        self.questions_answered.append(record)
        self.conversation.answered_questions.append(record)
        if self.current_question == question_id:
            self.current_question = None
            self.conversation.current_question = None
        self.last_updated = time.time()

    def update_missing_fields(self) -> List[str]:
        """
        Determines the list of required fields that are still missing.
        Rules:
        - All complaints require: category, location, description.
        - Streetlight complaints specifically benefit from visual_status (ON/OFF) if not already known.
        """
        missing = []
        if not self.is_field_provided("category"):
            missing.append("category")

        # For streetlight: check if visual_status or issue_type is known
        if self.complaint_category == "streetlight":
            if not self.is_field_provided("visual_status") and not self.is_question_answered("visual_status"):
                missing.append("visual_status")

        if not self.is_field_provided("location") and not self.is_question_answered("location"):
            missing.append("location")

        if not self.is_field_provided("description") and not self.is_question_answered("description"):
            # If description already exists from initial message or category clarification, do not mark missing
            if not (self.description and len(self.description.strip()) >= 5):
                missing.append("description")

        self.missing_fields = missing
        if not missing and self.complaint_status != "submitted":
            self.complaint_status = "ready_for_submission"
        return missing

    def get_next_missing_field(self) -> Optional[str]:
        """Returns the next single missing field to ask the citizen."""
        missing = self.update_missing_fields()
        for field in missing:
            # Don't re-ask a question that's already in questions_answered
            if not self.is_question_answered(field):
                return field
        return None

    def lock_category(self, category: str, source: str = "text", confidence: float = 0.9):
        """Locks complaint category to prevent accidental topic switching."""
        normalized = normalize_category_key(category)
        self.complaint_category = normalized
        self.confirmed_category = normalized
        self.category_confirmed = True
        self.category_locked = True
        self.category_source = source
        if source == "text":
            self.text_category = normalized
            self.text_confidence = confidence
        elif source in ["image", "video"]:
            self.image_category = normalized
            self.image_confidence = confidence
        self.last_updated = time.time()

    def correct_category(self, category: str, source: str = "correction", confidence: float = 0.95):
        """
        Explicitly corrects complaint category based on user instruction (BUG 5),
        even if the category was previously locked.
        """
        normalized = normalize_category_key(category)
        logger.info(f"[ConversationState] Correcting category from '{self.complaint_category}' to '{normalized}'")
        self.complaint_category = normalized
        self.confirmed_category = normalized
        self.category_confirmed = True
        self.category_locked = True
        self.category_source = source
        self.text_category = normalized
        self.text_confidence = confidence
        # Reset visual status if switching away from streetlight
        if normalized != "streetlight":
            self.visual_status = None
        self.update_missing_fields()
        self.last_updated = time.time()

    def set_pending_field(self, field: Optional[str]):
        """Sets the active pending field that the bot is waiting the citizen to answer."""
        self.pending_field = field
        self.current_question = field
        if field and field not in self.questions_asked:
            self.questions_asked.append(field)
        self.last_updated = time.time()

    def clear_pending_field(self):
        """Clears the pending field once an answer has been provided."""
        self.pending_field = None
        self.current_question = None
        self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Returns state as a structured dictionary matching requested format."""
        return {
            "input_type": self.input_type,
            "video_analyzed": self.video.video_analyzed,
            "detected_category": self.video.detected_category or self.complaint_category,
            "detected_issue": self.video.detected_issue or self.issue_type,
            "confidence": self.video.confidence,
            "user_confirmed": self.user_confirmed,
            "complaint_status": self.complaint_status,
            "complaint_category": self.complaint_category,
            "category": self.complaint_category,
            "confirmed_category": self.confirmed_category,
            "category_confirmed": self.category_confirmed,
            "category_locked": self.category_locked,
            "category_source": self.category_source,
            "text_category": self.text_category,
            "text_confidence": self.text_confidence,
            "image_category": self.image_category,
            "image_confidence": self.image_confidence,
            "pending_field": self.pending_field,
            "pending_cancel_ticket": self.pending_cancel_ticket,
            "candidate_categories": self.candidate_categories,
            "issue_type": self.issue_type,
            "location": self.location,
            "description": self.description,
            "visual_status": self.visual_status,
            "language": self.language,
            "image_uploaded": self.image_uploaded,
            "image_analysis": self.image_analysis,
            "image": {
                "uploaded": self.image_uploaded,
                "relevant": self.image.relevant,
                "detected_category": self.image.detected_category,
                "confidence": self.image.confidence,
                "visual_status": self.image.visual_status,
                "observations": self.image.observations
            },
            "video": {
                "uploaded": self.video.uploaded,
                "video_analyzed": self.video.video_analyzed,
                "detected_category": self.video.detected_category,
                "detected_issue": self.video.detected_issue,
                "confidence": self.video.confidence,
                "confidence_level": self.video.confidence_level,
                "visual_status": self.video.visual_status,
                "has_multiple_issues": self.video.has_multiple_issues,
                "candidate_issues": self.video.candidate_issues,
                "user_confirmed": self.video.user_confirmed
            },
            "conversation": {
                "current_question": self.current_question,
                "asked_questions": self.questions_asked,
                "answered_questions": self.questions_answered
            },
            "questions_asked": self.questions_asked,
            "questions_answered": self.questions_answered,
            "current_question": self.current_question,
            "missing_fields": self.missing_fields
        }


class ConversationStateManager:
    """
    Thread-safe conversation state store with in-memory persistence and session tracking.
    """
    def __init__(self, ttl_seconds: int = 7200):
        self._states: Dict[str, ConversationState] = {}
        self.ttl_seconds = ttl_seconds

    def _cleanup_expired(self):
        """Removes sessions older than TTL."""
        now = time.time()
        expired = [sid for sid, st in self._states.items() if (now - st.last_updated) > self.ttl_seconds]
        for sid in expired:
            del self._states[sid]

    def get_or_create(self, session_id: str, citizen_phone: Optional[str] = None) -> ConversationState:
        """Retrieves an existing conversation state or creates a fresh one."""
        self._cleanup_expired()
        key = session_id or citizen_phone or "default_session"
        if key not in self._states:
            self._states[key] = ConversationState(session_id=key, citizen_phone=citizen_phone)
            logger.info(f"[StateManager] Created new conversation state for session: {key}")
        return self._states[key]

    def get_state(self, session_id: str) -> Optional[ConversationState]:
        """Gets existing state if any."""
        return self._states.get(session_id)

    def reset_state(self, session_id: str) -> ConversationState:
        """Resets the state for a new complaint after successful submission."""
        if session_id in self._states:
            del self._states[session_id]
        fresh = ConversationState(session_id=session_id)
        self._states[session_id] = fresh
        return fresh

    def update_from_user_turn(
        self,
        session_id: str,
        user_text: str,
        extracted_category: Optional[str] = None,
        extracted_location: Optional[str] = None,
        extracted_issue: Optional[str] = None,
        detected_language: Optional[str] = None
    ) -> ConversationState:
        """
        Processes a user message turn:
        1. Checks if user answers the current_question.
        2. Updates complaint fields (category, location, visual_status, description).
        3. Recalculates missing fields.
        """
        state = self.get_or_create(session_id)
        text_clean = (user_text or "").strip()
        text_lower = text_clean.lower()

        if detected_language:
            state.language = detected_language

        # Check if user is answering the current question
        current_q = state.current_question

        # Case 1: Answering visual_status ("Is the light ON or OFF?")
        if current_q == "visual_status" or (state.complaint_category == "streetlight" and not state.visual_status):
            # Check for OFF answers
            if text_lower in ["off", "band", "band hai", "band aahe", "nahi", "not on", "dead", "blackout", "बंद", "बंद आहे", "लाईट बंद आहे"]:
                state.visual_status = "off"
                state.issue_type = "light_not_working"
                state.record_question_answered("visual_status", text_clean)
                state.update_missing_fields()
                return state
            # Check for ON answers
            elif text_lower in ["on", "chalu", "chalu hai", "chal raha hai", "suru", "चालू", "सुरु आहे", "दिन में चालू"]:
                state.visual_status = "on_daytime"
                state.issue_type = "burning_during_day"
                state.record_question_answered("visual_status", text_clean)
                state.update_missing_fields()
                return state
            # Check for flickering
            elif "flicker" in text_lower or "lukluk" in text_lower or "लुकलुक" in text_lower or "कम-ज्यादा" in text_lower:
                state.visual_status = "flickering"
                state.issue_type = "light_flickering"
                state.record_question_answered("visual_status", text_clean)
                state.update_missing_fields()
                return state

        # Case 2: Answering location
        if current_q == "location":
            if len(text_clean) >= 3:
                state.location = text_clean
                state.record_question_answered("location", text_clean)
                state.update_missing_fields()
                return state

        # Case 3: Answering description
        if current_q == "description":
            if len(text_clean) >= 3:
                state.description = text_clean
                state.record_question_answered("description", text_clean)
                state.update_missing_fields()
                return state

        # General Field Extraction
        if extracted_category and not state.category_locked and not state.category_confirmed:
            norm_cat = normalize_category_key(extracted_category)
            if norm_cat != "other":
                state.complaint_category = norm_cat
                state.confirmed_category = norm_cat
                state.category_confirmed = True
                state.category_locked = True
                if current_q == "category":
                    state.record_question_answered("category", norm_cat)

        if extracted_location and not state.location:
            state.location = extracted_location
            if current_q == "location":
                state.record_question_answered("location", extracted_location)

        if extracted_issue and not state.issue_type:
            state.issue_type = extracted_issue

        if not state.description and len(text_clean) >= 5:
            state.description = text_clean

        state.update_missing_fields()
        state.last_updated = time.time()
        return state


# Singleton Instance
conversation_state_manager = ConversationStateManager()
