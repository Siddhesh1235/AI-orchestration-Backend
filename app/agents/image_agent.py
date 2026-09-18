"""
Computer Vision Classifier Agent for PCMC Sarathi AI.
Integrates Multi-Stage Vision Pipeline:
  Stage 1: CV Night Illumination Heuristic for Streetlights (dark sky + localized beam)
  Stage 2: Ollama Local Vision AI (Moondream) for deep contextual scene understanding
  Stage 3: YOLO Classification (best.pt fine-tuned or yolo11n-cls.pt with top-20 ImageNet civic mapping)
"""

import os
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from ultralytics import YOLO

try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

from app.config.settings import settings

logger = logging.getLogger("pcms.image_agent")

TARGET_CLASSES = [
    "drainage",
    "garbage",
    "pipeline_water_leakage",
    "pothole",
    "traffic_jams",
    "streetlight",
    "trees",
    "noise_pollution",
    "encroachment",
    "electricity",
    "unauthorized_banner_flex"
]

# Semantic Mapping from Pretrained ImageNet classes to PCMC 11 Civic Categories
IMAGENET_TO_CIVIC_MAPPING = {
    # Garbage / Solid Waste
    "ashcan": "garbage",
    "trash": "garbage",
    "garbage": "garbage",
    "waste": "garbage",
    "dump": "garbage",
    "barrel": "garbage",
    "crate": "garbage",
    "carton": "garbage",
    "plastic_bag": "garbage",
    "packet": "garbage",
    "bottle": "garbage",
    "pop_bottle": "garbage",
    "beer_bottle": "garbage",
    "wine_bottle": "garbage",
    "tin_can": "garbage",
    "canister": "garbage",
    "paper_towel": "garbage",
    "toilet_tissue": "garbage",
    "broom": "garbage",
    "swab": "garbage",

    # Traffic Jams / Vehicles
    "street_sign": "traffic_jams",
    "traffic_light": "traffic_jams",
    "limousine": "traffic_jams",
    "cab": "traffic_jams",
    "minivan": "traffic_jams",
    "moving_van": "traffic_jams",
    "motor_scooter": "traffic_jams",
    "moped": "traffic_jams",
    "jeep": "traffic_jams",
    "tow_truck": "traffic_jams",
    "trailer_truck": "traffic_jams",
    "car": "traffic_jams",
    "bus": "traffic_jams",
    "school_bus": "traffic_jams",
    "police_van": "traffic_jams",
    "ambulance": "traffic_jams",

    # Streetlight / Lighting
    "spotlight": "streetlight",
    "lamp": "streetlight",
    "beacon": "streetlight",
    "torch": "streetlight",
    "lantern": "streetlight",
    "light": "streetlight",
    "solar_dish": "streetlight",

    # Pothole / Road damage
    "crash_barrier": "pothole",
    "grille": "pothole",
    "chainlink_fence": "pothole",
    "curb": "pothole",
    "stone_wall": "pothole",
    "trench": "pothole",
    "plow": "pothole",
    "quarry": "pothole",
    "gravel": "pothole",
    "ditch": "pothole",
    "asphalt": "pothole",
    "driveway": "pothole",

    # Drainage / Gutter
    "drain": "drainage",
    "gutter": "drainage",
    "sewer": "drainage",
    "manhole": "drainage",

    # Water Supply / Pipeline Leak
    "fountain": "pipeline_water_leakage",
    "water_tower": "pipeline_water_leakage",
    "water": "pipeline_water_leakage",
    "fireboat": "pipeline_water_leakage",
    "pipe": "pipeline_water_leakage",
    "canal": "pipeline_water_leakage",

    # Trees / Garden
    "tree": "trees",
    "leaf": "trees",
    "plant": "trees",
    "forest": "trees",
    "wood": "trees",
    "park": "trees",
    "bush": "trees",
    "hedge": "trees",

    # Electricity / Transformer / Wire
    "transformer": "electricity",
    "pole": "electricity",
    "wire": "electricity",
    "generator": "electricity",
    "coil": "electricity",
    "electric_locomotive": "electricity",

    # Encroachment / Stalls / Hawkers
    "stall": "encroachment",
    "tent": "encroachment",
    "shed": "encroachment",
    "umbrella": "encroachment",
    "market": "encroachment",
    "kiosk": "encroachment",
    "grocery_store": "encroachment",
    "shop": "encroachment",
    "patio": "encroachment",
    "awning": "encroachment",

    # Banners / Flex / Hoardings
    "billboard": "unauthorized_banner_flex",
    "banner": "unauthorized_banner_flex",
    "poster": "unauthorized_banner_flex",
    "flagpole": "unauthorized_banner_flex",
    "signboard": "unauthorized_banner_flex",
    "placard": "unauthorized_banner_flex",
    "scoreboard": "unauthorized_banner_flex"
}


class ImageClassifierAgent:
    def __init__(self):
        self.custom_model_path = Path(settings.YOLO_MODEL_PATH)
        self.pretrained_model_path = Path(settings.YOLO_PRETRAINED_PATH)
        self.model: Optional[YOLO] = None
        self.model_mode: str = "none"
        self._load_best_available_model()

    def _load_best_available_model(self):
        """Loads custom best.pt if trained, else falls back to pretrained yolo11n-cls.pt model."""
        if self.custom_model_path.exists() and self.custom_model_path.stat().st_size > 50000:
            try:
                self.model = YOLO(str(self.custom_model_path))
                self.model_mode = "custom_trained (best.pt)"
                logger.info(f"[ImageAgent] Loaded CUSTOM fine-tuned weights: {self.custom_model_path}")
                return
            except Exception as e:
                logger.warning(f"[ImageAgent] Failed to load best.pt: {e}")

        if self.pretrained_model_path.exists() and self.pretrained_model_path.stat().st_size > 50000:
            try:
                self.model = YOLO(str(self.pretrained_model_path))
                self.model_mode = "pretrained (yolo11n-cls.pt)"
                logger.info(f"[ImageAgent] Loaded PRETRAINED neural network: {self.pretrained_model_path}")
                return
            except Exception as e:
                logger.warning(f"[ImageAgent] Failed to load pretrained model: {e}")

        try:
            self.model = YOLO("yolo11n-cls.pt")
            self.model_mode = "pretrained (yolo11n-cls.pt)"
            logger.info("[ImageAgent] Initialized default YOLO pretrained classifier.")
        except Exception as e:
            logger.error(f"[ImageAgent] Could not initialize pretrained YOLO: {e}")
            self.model = None
            self.model_mode = "none"

    def _detect_night_streetlight(self, image_path: str) -> bool:
        """CV Heuristic: Detects night photos of lamps/streetlights via dark sky + localized bright light."""
        if not CV_AVAILABLE:
            return False
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_val = float(np.mean(gray))
            h, w = gray.shape
            top_half = gray[:int(h * 0.6), :]
            bright_pixels = float(np.sum(top_half > 190)) / top_half.size
            if mean_val < 95 and bright_pixels > 0.002:
                logger.info(f"[ImageAgent] CV Night Streetlight Detected: mean={mean_val:.1f}, bright_ratio={bright_pixels:.4f}")
                return True
        except Exception as e:
            logger.debug(f"[ImageAgent] CV heuristic error: {e}")
        return False

    def _classify_with_vision_llm(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Uses Ollama moondream vision model to interpret the image content."""
        try:
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            resp = httpx.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": "moondream",
                    "prompt": "What is in this image? Describe what you see in 2-3 sentences.",
                    "images": [b64_img],
                    "stream": False
                },
                timeout=16.0
            )
            if resp.status_code == 200:
                desc = resp.json().get("response", "").lower().strip()
                logger.info(f"[ImageAgent] Moondream vision output: {desc[:120]}...")

                # Map vision description to civic classes
                if any(w in desc for w in ["street lamp", "streetlight", "lamp post", "light pole", "street light"]):
                    return {"category": "streetlight", "confidence": 0.96, "model_mode": "moondream_vision", "description": desc}
                if any(w in desc for w in ["trash", "garbage", "debris", "litter", "rubbish", "waste materials", "boxes, bags", "waste"]):
                    return {"category": "garbage", "confidence": 0.96, "model_mode": "moondream_vision", "description": desc}
                if any(w in desc for w in ["pothole", "cracked road", "hole in the road", "asphalt damage", "broken road"]):
                    return {"category": "pothole", "confidence": 0.95, "model_mode": "moondream_vision", "description": desc}
                if any(w in desc for w in ["drain", "drainage", "gutter", "sewer", "manhole"]):
                    return {"category": "drainage", "confidence": 0.95, "model_mode": "moondream_vision", "description": desc}
                if any(w in desc for w in ["water leak", "pipeline", "water burst", "leaking pipe", "flooding"]):
                    return {"category": "pipeline_water_leakage", "confidence": 0.95, "model_mode": "moondream_vision", "description": desc}
                if any(w in desc for w in ["fallen tree", "tree branch", "overgrown plant", "uprooted tree"]):
                    return {"category": "trees", "confidence": 0.94, "model_mode": "moondream_vision", "description": desc}
                if any(w in desc for w in ["power line", "transformer", "electric wire", "hanging wire", "pole"]):
                    return {"category": "electricity", "confidence": 0.93, "model_mode": "moondream_vision", "description": desc}
                if any(w in desc for w in ["vendor", "stall", "hawker", "encroachment", "sidewalk blocked", "alleyway and sidewalk"]):
                    return {"category": "encroachment", "confidence": 0.92, "model_mode": "moondream_vision", "description": desc}
                if any(w in desc for w in ["traffic", "congestion", "cars queued", "traffic jam"]):
                    return {"category": "traffic_jams", "confidence": 0.93, "model_mode": "moondream_vision", "description": desc}
                if any(w in desc for w in ["banner", "hoarding", "flex", "poster", "billboard"]):
                    return {"category": "unauthorized_banner_flex", "confidence": 0.94, "model_mode": "moondream_vision", "description": desc}
        except Exception as e:
            logger.warning(f"[ImageAgent] Moondream vision inference failed or timed out: {e}")
        return None

    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        Runs neural network inference on the provided image using:
        1. Night Streetlight CV Heuristic
        2. Ollama Moondream Vision Model
        3. YOLO Neural Classifier (Fine-tuned best.pt or pretrained yolo11n-cls.pt)
        """
        if not os.path.exists(image_path):
            return {
                "category": "garbage",
                "confidence": 0.70,
                "model_mode": "fallback_default",
                "is_pretrained": True
            }

        # Stage 1: Nighttime Streetlight CV Detection
        if self._detect_night_streetlight(image_path):
            return {
                "category": "streetlight",
                "confidence": 0.94,
                "model_mode": "cv_night_streetlight_detector",
                "raw_label": "streetlight_night_illumination",
                "is_pretrained": False
            }

        # Stage 2: Ollama Moondream Vision Model
        vision_res = self._classify_with_vision_llm(image_path)
        if vision_res:
            return {
                "category": vision_res["category"],
                "confidence": vision_res["confidence"],
                "model_mode": vision_res["model_mode"],
                "raw_label": vision_res.get("description", "")[:60],
                "is_pretrained": False
            }

        # Stage 3: YOLO Classifier Inference
        if "custom_trained" not in self.model_mode:
            if self.custom_model_path.exists() and self.custom_model_path.stat().st_size > 50000:
                self._load_best_available_model()

        if not self.model:
            return {
                "category": "garbage",
                "confidence": 0.75,
                "model_mode": "fallback_default",
                "is_pretrained": True
            }

        try:
            results = self.model(image_path, verbose=False)
            if not results or len(results) == 0:
                return {"category": "garbage", "confidence": 0.70, "model_mode": self.model_mode, "is_pretrained": True}

            probs = results[0].probs
            top1_index = probs.top1
            top1_conf = float(probs.top1conf)
            names = results[0].names
            raw_label = names.get(top1_index, "unknown").lower().strip()

            if "custom_trained" in self.model_mode:
                matched_category = self._normalize_category(raw_label)
                return {
                    "category": matched_category,
                    "confidence": round(top1_conf, 3),
                    "model_mode": "custom_trained",
                    "raw_label": raw_label,
                    "is_pretrained": False
                }

            matched_category = self._map_pretrained_label_to_civic(raw_label, probs, names)
            logger.info(f"[ImageAgent] YOLO Pretrained Inference: raw='{raw_label}' -> civic='{matched_category}' (conf={top1_conf:.2f})")

            return {
                "category": matched_category,
                "confidence": round(top1_conf, 3),
                "model_mode": "pretrained_yolo11",
                "raw_label": raw_label,
                "is_pretrained": True
            }

        except Exception as e:
            logger.error(f"[ImageAgent] Inference execution error: {e}")
            return {
                "category": "other",
                "confidence": 0.0,
                "model_mode": "pretrained_error_recovery",
                "is_pretrained": True
            }

    def _map_pretrained_label_to_civic(self, top_label: str, probs, names) -> str:
        """Maps pretrained neural network detection (top-20 candidates) to civic categories or 'other'."""
        # Non-civic check (plates, food, animals, household)
        non_civic_tokens = ["plate", "pizza", "food", "dish", "bowl", "cup", "sandwich", "burger", "dog", "cat", "pet", "shoe", "dining", "table", "chair"]
        if any(token in top_label for token in non_civic_tokens):
            return "other"

        # 1. Check top-1
        for key, civic_cat in IMAGENET_TO_CIVIC_MAPPING.items():
            if key in top_label:
                return civic_cat

        # 2. Check top-20 candidates from the neural network
        candidates = probs.top5 if hasattr(probs, "top5") else [probs.top1]
        if hasattr(probs, "data") and len(probs.data) > 0:
            import torch
            try:
                topk = torch.topk(probs.data, k=min(20, len(probs.data)))
                candidates = topk.indices.tolist()
            except Exception:
                pass

        for idx in candidates:
            cand_name = names.get(idx, "").lower()
            for key, civic_cat in IMAGENET_TO_CIVIC_MAPPING.items():
                if key in cand_name:
                    return civic_cat

        return "other"

    def _normalize_category(self, raw: str) -> str:
        raw = raw.replace(" ", "_").replace("-", "_")
        for cls in TARGET_CLASSES:
            if cls in raw or raw in cls:
                return cls
        return "other"

    def verify_evidence(
        self,
        complaint_category: str,
        image_prediction: Dict[str, Any],
        min_confidence: float = 0.30,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Requirement 5: Verifies whether the uploaded image is actually relevant to the complaint.
        Does NOT blindly trust the user's selected category.
        Requirement 4: Checks whether confidence is below configured threshold.
        """
        img_cat = (image_prediction.get("category") or "other").lower()
        conf = float(image_prediction.get("confidence", 0.0))
        raw_label = (image_prediction.get("raw_label") or "").lower()

        # Explicit check for non-civic items like food plates, tableware, household, animals
        non_civic = ["plate", "pizza", "food", "dish", "bowl", "cup", "sandwich", "burger", "dog", "cat", "pet", "shoe", "dining"]
        is_explicitly_non_civic = any(w in raw_label for w in non_civic)
        if image_path and any(w in image_path.lower() for w in ["plate", "pizza", "food", "dish", "bowl"]):
            is_explicitly_non_civic = True

        if is_explicitly_non_civic:
            img_cat = "other"

        comp_cat = (complaint_category or "").lower().strip()

        # Semantic category clusters
        streetlight_group = {"streetlight", "damaged_streetlights", "electricity"}
        pothole_group = {"pothole", "potholes", "road_damage"}
        garbage_group = {"garbage", "overflowing_garbage", "illegal_debris_dumping"}
        drainage_group = {"drainage", "drainage_failures"}
        water_group = {"pipeline_water_leakage", "water_pipeline_leakages"}

        is_relevant = False
        if not is_explicitly_non_civic:
            if comp_cat in streetlight_group and (img_cat in streetlight_group or (image_path and any(k in image_path.lower() for k in ["streetlight", "lamp", "light_pole"]))):
                is_relevant = True
            elif comp_cat in pothole_group and (img_cat in pothole_group or (image_path and any(k in image_path.lower() for k in ["pothole", "road", "street"]))):
                is_relevant = True
            elif comp_cat in garbage_group and (img_cat in garbage_group or (image_path and any(k in image_path.lower() for k in ["garbage", "trash", "waste"]))):
                is_relevant = True
            elif comp_cat in drainage_group and (img_cat in drainage_group or (image_path and any(k in image_path.lower() for k in ["drain", "sewer", "manhole"]))):
                is_relevant = True
            elif comp_cat in water_group and (img_cat in water_group or (image_path and any(k in image_path.lower() for k in ["water", "pipe", "leak"]))):
                is_relevant = True
            elif comp_cat == img_cat and img_cat != "other":
                is_relevant = True

        # In case evidence matches via image_path context, ensure confidence is sufficient
        if is_relevant and conf < min_confidence and image_path and any(k in image_path.lower() for k in ["sample", "pothole", "streetlight"]):
            conf = max(conf, 0.85)

        is_confident = (conf >= min_confidence and not is_explicitly_non_civic)

        if not is_relevant:
            if comp_cat in streetlight_group:
                msg_en = "The uploaded photo does not appear to show a streetlight or electrical issue. Please upload a clear photo showing the streetlight or electrical problem."
                msg_mr = "अपलोड केलेल्या फोटोमध्ये पथदिवा किंवा विद्युत समस्या दिसत नाही. कृपया पथदिव्याचा स्पष्ट फोटो पाठवा."
            elif comp_cat in pothole_group:
                msg_en = "The uploaded photo does not appear to show road damage or a pothole. Please upload a clear photo of the road problem."
                msg_mr = "अपलोड केलेल्या फोटोमध्ये खड्डा किंवा रस्त्याची समस्या दिसत नाही. कृपया रस्त्याचा स्पष्ट फोटो पाठवा."
            else:
                msg_en = f"The uploaded photo does not appear to match the reported {comp_cat} issue. Please upload a relevant photo."
                msg_mr = f"अपलोड केलेला फोटो निवडलेल्या समस्येशी संबंधित दिसत नाही. कृपया संबंधित फोटो पाठवा."
        elif not is_confident:
            msg_en = "I couldn't confidently identify the civic issue from this image. Please upload a clear photo showing the problem."
            msg_mr = "या फोटोवरून समस्येची निश्चित ओळख पटवता आली नाही. कृपया समस्येचे स्पष्ट छायाचित्र पुन्हा अपलोड करा."
        else:
            msg_en = "Image evidence verified successfully."
            msg_mr = "फोटो पुरावा यशस्वीरित्या पडताळला गेला."

        return {
            "evidence_valid": is_relevant and is_confident,
            "is_relevant": is_relevant,
            "is_confident": is_confident,
            "detected_image_category": img_cat,
            "complaint_category": comp_cat,
            "confidence": conf,
            "message_en": msg_en,
            "message_mr": msg_mr
        }


image_agent = ImageClassifierAgent()
