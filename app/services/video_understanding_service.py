"""
Video Understanding & Ingestion Service for PCMC Sarathi AI.
Executes the Video Grievance Processing Pipeline:
1. Validates video format, duration (MAX_VIDEO_DURATION_SEC), and size (MAX_VIDEO_SIZE_MB).
2. Extracts 1 frame per second via OpenCV (capped at MAX_FRAMES_PER_VIDEO).
3. Reuses existing image_understanding_service checks (blur via Laplacian variance, non-civic keywords).
4. Reuses existing best.pt classifier (image_agent.classify_image) for civic defect prediction.
5. Aggregates frame predictions using frequency × confidence scoring; persists the highest-confidence
   frame as the primary evidence photo in uploads/.
6. Reuses bhashini_stt_service to extract and transcribe the audio track via FFmpeg + Bhashini ASR.
7. Merges visual aggregation and audio transcript into a structured VideoIngestionResult.
"""

import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from app.config.settings import settings
from app.schemas.video import VideoIngestionResult, FrameAnalysisResult
from app.agents.image_agent import image_agent
from app.services.image_understanding_service import image_understanding_service
from app.services.bhashini_stt_service import (
    bhashini_stt_service,
    extract_audio_from_video_file,
    normalize_language_code
)

logger = logging.getLogger("pcms.video_understanding")


class VideoUnderstandingService:
    def __init__(self):
        self.max_duration_sec: int = settings.MAX_VIDEO_DURATION_SEC
        self.max_frames: int = settings.MAX_FRAMES_PER_VIDEO
        self.max_size_mb: int = settings.MAX_VIDEO_SIZE_MB

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Reads video duration, total frame count, and fps using OpenCV."""
        if not os.path.exists(video_path):
            return {"valid": False, "error": "File not found", "duration_sec": 0.0, "frame_count": 0, "fps": 0.0}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"valid": False, "error": "Could not open video file", "duration_sec": 0.0, "frame_count": 0, "fps": 0.0}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        duration_sec = total_frames / fps if total_frames > 0 else 0.0
        return {
            "valid": True,
            "duration_sec": round(duration_sec, 2),
            "frame_count": total_frames,
            "fps": round(fps, 2)
        }

    async def process_video(
        self,
        video_path: str,
        language: str = "mr",
        target_fps_sample: int = 1
    ) -> VideoIngestionResult:
        """
        Executes the full video understanding pipeline:
        1. Temporal frame extraction (1 frame/sec up to max_frames).
        2. Blur & non-civic filtering via image_understanding_service.
        3. Neural classification via image_agent (best.pt).
        4. (Frequency × Confidence) voting aggregation.
        5. Audio extraction and Bhashini speech-to-text transcription.
        6. Preserves the best evidence frame.
        """
        lang = normalize_language_code(language)
        meta = self.get_video_metadata(video_path)
        duration_sec = meta.get("duration_sec", 0.0)
        total_video_frames = meta.get("frame_count", 0)
        fps = meta.get("fps", 25.0)

        # 1. Create a dedicated isolated temporary directory for extracted frames
        session_id = uuid.uuid4().hex
        temp_frames_dir = Path(settings.UPLOAD_DIR) / f"temp_frames_{session_id}"
        temp_frames_dir.mkdir(parents=True, exist_ok=True)

        frames_analyzed = 0
        frames_rejected_blur = 0
        frames_rejected_noncivic = 0
        extracted_frame_records: List[Dict[str, Any]] = []
        frame_details_pydantic: List[FrameAnalysisResult] = []

        try:
            # 2. Extract 1 frame per second using OpenCV CAP_PROP_FPS
            cap = cv2.VideoCapture(video_path)
            frame_interval = max(1, int(round(fps / max(1, target_fps_sample))))

            frame_idx = 0
            extracted_count = 0

            while cap.isOpened() and extracted_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    current_time_sec = round(frame_idx / fps, 2)
                    frame_name = f"frame_{extracted_count:03d}_{current_time_sec}s.jpg"
                    frame_path = temp_frames_dir / frame_name
                    cv2.imwrite(str(frame_path), frame)
                    extracted_count += 1

                    # 3. Validate frame using existing image_understanding_service
                    val_res = image_understanding_service.validate_image(str(frame_path))
                    if not val_res["is_valid"]:
                        err_type = val_res.get("error_type")
                        if err_type == "blurry":
                            frames_rejected_blur += 1
                            frame_details_pydantic.append(FrameAnalysisResult(
                                frame_index=extracted_count,
                                timestamp_sec=current_time_sec,
                                category="unknown",
                                confidence=0.0,
                                is_valid=False,
                                rejection_reason="blurry_image"
                            ))
                            continue
                        elif err_type in ("unrelated", "corrupted"):
                            frames_rejected_noncivic += 1
                            frame_details_pydantic.append(FrameAnalysisResult(
                                frame_index=extracted_count,
                                timestamp_sec=current_time_sec,
                                category="unrelated",
                                confidence=0.0,
                                is_valid=False,
                                rejection_reason="non_civic_image"
                            ))
                            continue

                    # 4. Neural classification via image_agent (running trained best.pt)
                    pred = image_agent.classify_image(str(frame_path))
                    cat = (pred.get("category") or "other").lower()
                    conf = float(pred.get("confidence", 0.0))
                    raw_label = pred.get("raw_label", "")

                    # Check non-civic label keywords
                    if image_understanding_service.is_non_civic(raw_label) or cat == "unrelated":
                        frames_rejected_noncivic += 1
                        frame_details_pydantic.append(FrameAnalysisResult(
                            frame_index=extracted_count,
                            timestamp_sec=current_time_sec,
                            category="unrelated",
                            confidence=conf,
                            raw_label=raw_label,
                            is_valid=False,
                            rejection_reason="non_civic_prediction"
                        ))
                        continue

                    # Frame successfully survived all filters
                    frames_analyzed += 1
                    extracted_frame_records.append({
                        "frame_index": extracted_count,
                        "timestamp_sec": current_time_sec,
                        "category": cat,
                        "confidence": conf,
                        "raw_label": raw_label,
                        "frame_path": str(frame_path)
                    })
                    frame_details_pydantic.append(FrameAnalysisResult(
                        frame_index=extracted_count,
                        timestamp_sec=current_time_sec,
                        category=cat,
                        confidence=conf,
                        raw_label=raw_label,
                        is_valid=True
                    ))

                frame_idx += 1

            cap.release()

            # 5. Aggregation Logic: Combined Score = (Frequency × Average Confidence) = Sum of Confidences
            final_category = "other"
            final_confidence = 0.0
            evidence_frame_path = None
            combined_scores: Dict[str, float] = {}

            if extracted_frame_records:
                category_confidences: Dict[str, List[float]] = {}
                category_frames: Dict[str, List[Dict[str, Any]]] = {}

                for r in extracted_frame_records:
                    c = r["category"]
                    category_confidences.setdefault(c, []).append(r["confidence"])
                    category_frames.setdefault(c, []).append(r)

                for c, conf_list in category_confidences.items():
                    # Combined score = count * (sum(conf_list) / count) = sum(conf_list)
                    score = sum(conf_list)
                    combined_scores[c] = round(score, 3)

                # Winning category is the one with highest combined score
                best_category = max(combined_scores.items(), key=lambda x: x[1])[0]
                best_frames = category_frames[best_category]

                final_category = best_category
                # Average confidence for the winning class
                final_confidence = round(float(np.mean(category_confidences[best_category])), 3)

                # Pick the highest-confidence frame within the winning category as primary evidence
                highest_frame_record = max(best_frames, key=lambda x: x["confidence"])

                # Persist evidence image to permanent uploads directory
                evidence_filename = f"EVIDENCE_VIDEO_{session_id}.jpg"
                perm_evidence_path = Path(settings.UPLOAD_DIR) / evidence_filename
                shutil.copyfile(highest_frame_record["frame_path"], str(perm_evidence_path))
                evidence_frame_path = str(perm_evidence_path)
                logger.info(f"[VideoUnderstanding] Selected top evidence frame: {evidence_frame_path} (conf={highest_frame_record['confidence']})")

            # 6. Audio Track Extraction and Bhashini Speech-to-Text Transcription
            audio_transcript = ""
            detected_audio_lang = lang

            try:
                audio_bytes = extract_audio_from_video_file(video_path)
                if audio_bytes and len(audio_bytes) > 44:
                    stt_res = await bhashini_stt_service.transcribe_audio(
                        audio_bytes=audio_bytes,
                        language=lang,
                        audio_format="wav"
                    )
                    if stt_res.get("success") and stt_res.get("transcript"):
                        audio_transcript = stt_res["transcript"].strip()
                        detected_audio_lang = stt_res.get("language") or lang
                        logger.info(f"[VideoUnderstanding] Transcribed audio track: '{audio_transcript}' ({detected_audio_lang})")
            except Exception as audio_err:
                logger.warning(f"[VideoUnderstanding] Audio transcription skipped or failed: {audio_err}")

            # 7. Deep Defect Analysis & Attribute Extraction on Evidence Frame
            final_issue = "unknown"
            visual_status = None
            has_multiple_issues = False
            candidate_issues: List[str] = []

            # Check if multiple distinct categories appeared with significant frame frequency
            if len(combined_scores) > 1 and final_category not in ["other", "unrelated"]:
                top_score = combined_scores.get(final_category, 0.0)
                distinct_candidates = [
                    cat for cat, sc in combined_scores.items()
                    if cat != final_category and cat not in ["other", "unrelated"] and sc >= 0.35 and (sc / max(0.1, top_score)) >= 0.30
                ]
                if distinct_candidates:
                    has_multiple_issues = True
                    candidate_issues = [final_category] + distinct_candidates

            if evidence_frame_path and os.path.exists(evidence_frame_path):
                try:
                    img_analysis = image_understanding_service.analyze_image(
                        evidence_frame_path,
                        user_message=audio_transcript,
                        context={"category": final_category}
                    )
                    if img_analysis:
                        final_issue = img_analysis.issue or final_issue
                        visual_status = img_analysis.visual_status or visual_status
                        if img_analysis.has_multiple_issues:
                            has_multiple_issues = True
                            for c in img_analysis.candidate_issues:
                                if c not in candidate_issues:
                                    candidate_issues.append(c)
                except Exception as frame_err:
                    logger.warning(f"[VideoUnderstanding] Frame defect analysis warning: {frame_err}")

            # 8. Determine Confidence Level
            if final_confidence >= 0.70 and final_category not in ["other", "unrelated"]:
                conf_level = "high"
            elif final_confidence >= 0.40 and final_category not in ["other", "unrelated"]:
                conf_level = "medium"
            else:
                conf_level = "low"

            return VideoIngestionResult(
                category=final_category,
                confidence=final_confidence,
                issue=final_issue,
                visual_status=visual_status,
                confidence_level=conf_level,
                has_multiple_issues=has_multiple_issues,
                candidate_issues=candidate_issues,
                transcript=audio_transcript,
                detected_language=detected_audio_lang,
                evidence_frame_path=evidence_frame_path,
                video_path=video_path,
                frame_count=extracted_count,
                duration_sec=duration_sec,
                frames_analyzed=frames_analyzed,
                frames_rejected_blur=frames_rejected_blur,
                frames_rejected_noncivic=frames_rejected_noncivic,
                combined_scores=combined_scores,
                frame_details=frame_details_pydantic
            )

        finally:
            # Clean up temporary frame extraction folder
            if temp_frames_dir.exists():
                try:
                    shutil.rmtree(temp_frames_dir, ignore_errors=True)
                except Exception as clean_err:
                    logger.debug(f"[VideoUnderstanding] Cleanup notice: {clean_err}")


video_understanding_service = VideoUnderstandingService()
