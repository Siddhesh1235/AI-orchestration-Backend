"""
Configuration Settings for PCMC Sarathi AI Orchestrator.
Supports environment variables (.env) with production-ready defaults.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Ward Mitra - Grievance Orchestrator"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./pcmc_sarathi.db"

    # Local ML Model Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    YOLO_MODEL_PATH: str = str(BASE_DIR / "models" / "image_classifier" / "best.pt")
    YOLO_PRETRAINED_PATH: str = str(BASE_DIR / "models" / "image_classifier" / "yolo11n-cls.pt")
    MODEL_CONFIG_PATH: str = str(BASE_DIR / "app" / "config" / "model_config.yaml")
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")

    # Tier 2: Free Local Ollama Provider (Primary)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT_SECONDS: float = 35.0

    # Tier 3: OpenAI Fallback (Secondary, Emergency Only)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TIMEOUT_SECONDS: float = 10.0

    # Geolocation bounds for PCMC area (Latitude 18.57 to 18.72, Longitude 73.74 to 73.92)
    PCMC_LAT_MIN: float = 18.5700
    PCMC_LAT_MAX: float = 18.7200
    PCMC_LNG_MIN: float = 73.7400
    PCMC_LNG_MAX: float = 73.9200

    # Content Moderation (Profanity & Image Safety)
    ENABLE_CONTENT_MODERATION: bool = True
    ENABLE_AWS_REKOGNITION: bool = False
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "ap-south-1"

    # Bhashini Indic STT / ASR & TTS Configuration (Optional Cloud Fallback)
    BHASHINI_ENABLED: bool = False
    BHASHINI_ENDPOINT: str = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    BHASHINI_USER_ID: Optional[str] = None
    BHASHINI_API_KEY: Optional[str] = None
    BHASHINI_INFERENCE_API_KEY: Optional[str] = None
    BHASHINI_ASR_SERVICE_ID: str = "ai4bharat/conformer-multilingual-dravidian-indoaryan-gpu--gpu"
    BHASHINI_TTS_SERVICE_ID: str = "ai4bharat/indic-tts-coqui-indoaryan-gpu--gpu"
    BHASHINI_TTS_GENDER: str = "female"
    BHASHINI_TTS_SAMPLING_RATE: int = 22050
    BHASHINI_TTS_PIPELINE_ID: Optional[str] = None

    # Local Speech-to-Text (Faster-Whisper STT - 100% Free, Local & Self-Hosted)
    WHISPER_MODEL: str = "small"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_DOWNLOAD_ROOT: Optional[str] = None
    WHISPER_BEAM_SIZE: int = 5

    # Local Text-to-Speech (Meta MMS-TTS VITS - 100% Free, Local & Self-Hosted)
    TTS_ENGINE: str = "mms"
    TTS_LANGUAGE: str = "mr"
    TTS_DEVICE: str = "cpu"
    TTS_MODELS_DIR: Optional[str] = None
    TTS_SAMPLING_RATE: int = 16000

    # Video Ingestion Pipeline Configuration
    MAX_VIDEO_DURATION_SEC: int = 60
    MAX_FRAMES_PER_VIDEO: int = 30
    MAX_VIDEO_SIZE_MB: int = 10
    VIDEO_PROCESSING_TIMEOUT_SEC: float = 45.0

    # Notification Layer Configuration
    NOTIFICATION_ENABLED: bool = True
    WHATSAPP_ENABLED: bool = True
    WHATSAPP_PROVIDER: str = "mock"  # mock, twilio, gupshup, meta
    WHATSAPP_API_URL: Optional[str] = None
    WHATSAPP_API_TOKEN: Optional[str] = None
    WHATSAPP_SENDER_NUMBER: Optional[str] = None

    SMS_ENABLED: bool = True
    SMS_PROVIDER: str = "mock"  # mock, fast2sms, twilio
    SMS_API_KEY: Optional[str] = None
    SMS_SENDER_ID: Optional[str] = "PCMCGOV"

    PUSH_ENABLED: bool = True
    PUSH_PROVIDER: str = "mock"  # mock, fcm
    FIREBASE_CREDENTIALS_PATH: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure uploads directory exists
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
