"""
Audio Stream Transcription using Faster-Whisper (with graceful fallback).
"""
from datetime import datetime
from typing import Optional, List
from app.ingestion.chunker import StreamChunk
from app.utils.logger import get_logger
from app.utils.time_utils import now_utc

logger = get_logger("WhisperTranscriber")


class AudioStreamTranscriber:
    """
    Transcribes incoming audio segments into timestamped StreamChunk objects.
    Uses faster-whisper when installed, with graceful fallback.
    """

    def __init__(self, model_size: str = "base.en", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None
        self._initialized = False

    def _lazy_init(self):
        if not self._initialized:
            try:
                from faster_whisper import WhisperModel  # type: ignore
                logger.info(f"Loading faster-whisper model '{self.model_size}' on {self.device}...")
                self._model = WhisperModel(self.model_size, device=self.device, compute_type="int8")
                self._initialized = True
                logger.info("Faster-whisper model loaded successfully.")
            except ImportError:
                logger.warning("faster-whisper is not installed. Audio transcription will run in simulated mode.")
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to load faster-whisper model: {e}. Audio transcription will run in fallback mode.")
                self._initialized = True

    def transcribe_segment(
        self, audio_data: bytes, source: str = "stream://live-mic", timestamp: Optional[datetime] = None
    ) -> List[StreamChunk]:
        """Transcribe an audio segment into timestamped text chunks."""
        self._lazy_init()
        ts = timestamp or now_utc()

        if self._model is not None:
            import io
            audio_io = io.BytesIO(audio_data)
            segments, _ = self._model.transcribe(audio_io, beam_size=1)
            results = []
            for seg in segments:
                results.append(
                    StreamChunk(
                        id=f"audio-{int(ts.timestamp())}-{int(seg.start)}",
                        text=seg.text.strip(),
                        timestamp=ts,
                        source=source,
                        metadata={"start_sec": seg.start, "end_sec": seg.end},
                    )
                )
            return results

        # Simulated fallback if no whisper available
        return [
            StreamChunk(
                id=f"audio-{int(ts.timestamp())}",
                text="[Transcribed audio segment: operational status confirmed]",
                timestamp=ts,
                source=source,
                metadata={"mode": "fallback_stub"},
            )
        ]
