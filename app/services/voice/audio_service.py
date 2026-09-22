"""
Audio Normalization & Preprocessing Service for WardMitra Local Voice.
Supports: WebM, WAV, MP3, M4A, OGG, Opus.
Normalizes audio to 16kHz mono PCM WAV via bundled FFmpeg / imageio-ffmpeg.
Guarantees cleanup of temporary files.
"""

import os
import io
import shutil
import logging
import tempfile
import subprocess
from typing import Tuple, Optional
import soundfile as sf
import numpy as np

logger = logging.getLogger("pcms.audio_service")

# Supported audio extensions and MIME types
SUPPORTED_FORMATS = {"webm", "wav", "mp3", "m4a", "ogg", "opus", "flac", "aac", "oga"}


def get_ffmpeg_executable() -> Optional[str]:
    """Finds the ffmpeg executable from imageio-ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


class AudioService:
    """Handles audio format validation, conversion to 16kHz mono WAV, and safe cleanup."""

    @staticmethod
    def validate_audio_bytes(audio_bytes: bytes, min_bytes: int = 100) -> Tuple[bool, Optional[str]]:
        """
        Validates whether raw audio bytes are non-empty and sufficient for speech recognition.
        Returns (is_valid, error_message).
        """
        if not audio_bytes or len(audio_bytes) < min_bytes:
            return False, "Audio data is empty or too short."
        return True, None

    @staticmethod
    def convert_to_wav_16k_mono(audio_bytes: bytes, input_format: str = "webm") -> bytes:
        """
        Converts raw audio bytes (WebM, OGG, Opus, MP3, M4A, etc.) into 16kHz mono PCM WAV.
        Uses bundled FFmpeg executable for lossless resampling and channel downmixing.
        """
        if not audio_bytes or len(audio_bytes) < 10:
            return audio_bytes

        # If already standard 16kHz mono RIFF WAV, check format quickly
        if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]:
            try:
                data, samplerate = sf.read(io.BytesIO(audio_bytes))
                if samplerate == 16000 and (len(data.shape) == 1 or data.shape[1] == 1):
                    return audio_bytes
            except Exception:
                pass  # Re-convert if header is malformed

        ffmpeg_exe = get_ffmpeg_executable()
        if not ffmpeg_exe:
            logger.warning("[AudioService] FFmpeg executable not found. Proceeding with raw bytes.")
            return audio_bytes

        fmt = input_format.lower().lstrip(".")
        if fmt not in SUPPORTED_FORMATS:
            if audio_bytes.startswith(b"OggS"):
                fmt = "ogg"
            elif audio_bytes.startswith(b"\x1aE\xdf\xa3"):
                fmt = "webm"
            elif audio_bytes.startswith(b"RIFF"):
                fmt = "wav"
            elif audio_bytes.startswith(b"ID3") or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
                fmt = "mp3"
            else:
                fmt = "webm"

        in_file = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        in_path = in_file.name
        try:
            in_file.write(audio_bytes)
            in_file.close()

            out_path = in_path + ".converted.wav"
            cmd = [
                ffmpeg_exe, "-y",
                "-i", in_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                out_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 44:
                with open(out_path, "rb") as f:
                    wav_bytes = f.read()
                logger.info(f"[AudioService] Converted {fmt} ({len(audio_bytes)}B) -> 16kHz WAV ({len(wav_bytes)}B)")
                return wav_bytes
            else:
                err_msg = res.stderr.decode("utf-8", errors="ignore")
                logger.warning(f"[AudioService] FFmpeg conversion warning: {err_msg}")
                return audio_bytes
        except Exception as e:
            logger.error(f"[AudioService] Audio conversion error: {e}")
            return audio_bytes
        finally:
            AudioService.cleanup_temp_files(in_path, in_path + ".converted.wav")

    @staticmethod
    def get_audio_duration(audio_bytes: bytes) -> float:
        """Estimates duration of audio bytes in seconds."""
        try:
            # Try soundfile directly
            data, samplerate = sf.read(io.BytesIO(audio_bytes))
            return float(len(data)) / float(samplerate)
        except Exception:
            pass

        # Fallback: estimate PCM 16-bit 16kHz mono: 32,000 bytes/sec
        if audio_bytes.startswith(b"RIFF"):
            pcm_data_len = max(0, len(audio_bytes) - 44)
            return round(pcm_data_len / 32000.0, 2)

        return 0.0

    @staticmethod
    def cleanup_temp_files(*file_paths: Optional[str]):
        """Safely removes temporary files."""
        for p in file_paths:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception as e:
                    logger.debug(f"[AudioService] Cleanup error for {p}: {e}")


audio_service = AudioService()
