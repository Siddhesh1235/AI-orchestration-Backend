"""
Content Moderation Agent for PCMC Sarathi AI.
Fulfills Box 4.3 (Content Moderation & Safety Layer) of WardMitra AI Orchestrator Architecture.
Coordinates:
1. Multilingual Text Profanity & Abusive Language Filtering (via profanity_agent)
2. Image Safety & NSFW Moderation (Local CV heuristics with optional AWS Rekognition fallback)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import numpy as np

from app.config.settings import settings
from app.agents.profanity_agent import profanity_agent

logger = logging.getLogger("pcms.moderation_agent")


class ModerationAgent:
    """Central Content Moderation Agent protecting municipal systems and field staff."""

    def __init__(self):
        self.enabled = settings.ENABLE_CONTENT_MODERATION
        self.aws_enabled = settings.ENABLE_AWS_REKOGNITION and bool(settings.AWS_ACCESS_KEY_ID)
        logger.info(
            f"[ModerationAgent] Initialized (Active: {self.enabled}, AWS Rekognition: {self.aws_enabled})"
        )

    def evaluate_image_safety(self, image_path: Optional[str]) -> Dict[str, Any]:
        """
        Evaluates an uploaded photo for:
        - Image validity and integrity
        - Blank / solid color spam photos
        - Unsafe / explicit / NSFW skin-tone exposure
        - Optional AWS Rekognition moderation labels (if configured)
        """
        if not image_path or not os.path.exists(image_path):
            return {
                "is_safe": True,
                "is_nsfw": False,
                "reason": None,
                "source": "none"
            }

        # 1. Check Image Validity with Pillow
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size

                # Minimum dimension sanity check
                if width < 50 or height < 50:
                    return {
                        "is_safe": False,
                        "is_nsfw": False,
                        "reason": "प्रतिमा आकाराने फार लहान किंवा अस्पष्ट आहे (Image dimensions too small)",
                        "source": "dimension_check"
                    }

                # Resize for fast heuristic analysis
                img_small = img.resize((150, 150))
                arr = np.array(img_small, dtype=np.float32)

                # 2. Check for solid/blank image (zero variance)
                std_dev = np.std(arr)
                if std_dev < 8.0:
                    return {
                        "is_safe": False,
                        "is_nsfw": False,
                        "reason": "रिकामे किंवा एकरंगी बनावट छायाचित्र आढळले (Blank or single-color image detected)",
                        "source": "blank_image_check"
                    }

                # 3. Local NSFW / Explicit Heuristic (Skin-tone pixel coverage)
                # In standard RGB: R > 95, G > 40, B > 20, max(R,G,B) - min(R,G,B) > 15, |R - G| > 15, R > G, R > B
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                skin_mask = (
                    (r > 95) & (g > 40) & (b > 20) &
                    ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15) &
                    (np.abs(r - g) > 15) & (r > g) & (r > b)
                )
                skin_ratio = np.sum(skin_mask) / (150 * 150)

                # Excessive skin exposure (> 50% skin tone area) flags potential NSFW
                if skin_ratio > 0.52:
                    logger.warning(
                        f"[ModerationAgent] Potential NSFW content detected by skin ratio ({skin_ratio:.2f}) on {image_path}"
                    )
                    return {
                        "is_safe": False,
                        "is_nsfw": True,
                        "reason": "छायाचित्रात अयोग्य किंवा आक्षेपार्ह घटक आढळले (Inappropriate or explicit image detected)",
                        "source": "local_vision_nsfw"
                    }

        except Exception as e:
            logger.warning(f"[ModerationAgent] Image inspection exception on {image_path}: {e}")
            return {
                "is_safe": False,
                "is_nsfw": False,
                "reason": "छायाचित्र फाइल खराब किंवा अवैध आहे (Corrupted or invalid image file)",
                "source": "corrupt_file_check"
            }

        # 4. Optional: AWS Rekognition API call if configured
        if self.aws_enabled:
            aws_res = self._check_aws_rekognition(image_path)
            if not aws_res["is_safe"]:
                return aws_res

        return {
            "is_safe": True,
            "is_nsfw": False,
            "reason": None,
            "source": "local_vision_pass"
        }

    def _check_aws_rekognition(self, image_path: str) -> Dict[str, Any]:
        """Optional AWS Rekognition moderation labels invocation."""
        try:
            import boto3
            client = boto3.client(
                "rekognition",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            with open(image_path, "rb") as image_file:
                response = client.detect_moderation_labels(
                    Image={"Bytes": image_file.read()},
                    MinConfidence=60.0
                )
            labels = response.get("ModerationLabels", [])
            if labels:
                flagged_names = [l["Name"] for l in labels]
                logger.warning(f"[ModerationAgent] AWS Rekognition flagged image: {flagged_names}")
                return {
                    "is_safe": False,
                    "is_nsfw": True,
                    "reason": f"Amazon Rekognition ने अयोग्य घटक शोधले: {', '.join(flagged_names)}",
                    "source": "aws_rekognition"
                }
        except Exception as e:
            logger.warning(f"[ModerationAgent] AWS Rekognition call failed, falling back to local: {e}")
            
        return {"is_safe": True, "is_nsfw": False, "reason": None, "source": "aws_fallback"}

    def moderate_complaint(
        self,
        description: str,
        photo_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified multi-modal moderation pipeline:
        Checks both Text (Profanity/Abuse) and Image (NSFW/Spam).
        Returns:
            is_safe: bool
            action: "ALLOW" | "REJECT"
            reasons: List[str]
            details: Dict[str, Any]
        """
        if not self.enabled:
            return {
                "is_safe": True,
                "action": "ALLOW",
                "reasons": [],
                "details": {"status": "moderation_disabled"}
            }

        reasons: List[str] = []

        # 1. Text Profanity Evaluation
        text_eval = profanity_agent.evaluate_text(description)
        if text_eval["is_profane"]:
            reasons.append(text_eval["reason"])

        # 2. Image Safety Evaluation
        image_eval = self.evaluate_image_safety(photo_path)
        if not image_eval["is_safe"]:
            reasons.append(image_eval["reason"])

        is_safe = (len(reasons) == 0)
        action = "ALLOW" if is_safe else "REJECT"

        if not is_safe:
            logger.warning(f"[ModerationAgent] Moderation Violation: {reasons}")

        return {
            "is_safe": is_safe,
            "action": action,
            "reasons": reasons,
            "details": {
                "text_moderation": text_eval,
                "image_moderation": image_eval
            }
        }


moderation_agent = ModerationAgent()
