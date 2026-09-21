"""CPU-only Silero VAD gate for a streaming 16 kHz microphone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # Silero's required 32 ms frame size at 16 kHz.
_CONTEXT_SAMPLES = 64


class VadModelNotFoundError(FileNotFoundError):
    """Raised when the local Silero ONNX model has not been provisioned."""


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """Inclusive-start/exclusive-end coordinates in a captured waveform."""

    start_sample: int
    end_sample: int


@dataclass(frozen=True, slots=True)
class GateUpdate:
    """Result of passing one microphone frame through the speech gate."""

    probability: float
    automatic_stop: bool
    segment: SpeechSegment | None


class SileroOnnxVad:
    """Minimal stateful wrapper around Silero's bundled ONNX VAD model."""

    def __init__(self, model_path: Path | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime is required for Silero VAD. Run `uv sync` "
                "after updating pyproject.toml."
            ) from error

        model_path = model_path or self._default_model_path()
        if not model_path.is_file():
            raise VadModelNotFoundError(
                f"Silero VAD model was not found at: {model_path}\n"
                "Place silero_vad.onnx in HexaFlow/models/."
            )

        options = ort.SessionOptions()
        # VAD is intentionally small and CPU-bound; do not compete with ASR later.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.reset()

    def reset(self) -> None:
        """Clear recurrent model state before a new utterance."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def speech_probability(self, frame: np.ndarray) -> float:
        """Return Silero's speech probability for one 512-sample frame."""
        frame = np.asarray(frame, dtype=np.float32)

        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(
                f"Silero requires exactly {FRAME_SAMPLES} mono samples; "
                f"received {frame.shape}."
            )

        model_audio = np.concatenate(
            (self._context, np.ascontiguousarray(frame).reshape(1, -1)),
            axis=1,
        )

        probability, next_state = self._session.run(
            None,
            {
                "input": model_audio,
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )

        self._state = np.asarray(next_state, dtype=np.float32)
        self._context = model_audio[:, -_CONTEXT_SAMPLES:].copy()
        return float(np.asarray(probability).squeeze())

    @staticmethod
    def _default_model_path() -> Path:
        if getattr(sys, "frozen", False):
            project_root = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        else:
            project_root = Path(__file__).resolve().parents[2]

        return project_root / "models" / "silero_vad.onnx"


class SileroSpeechGate:
    """
    Detects a valid utterance and ends it after one second of silence.

    Leading silence and final silence are removed, while short pauses inside
    speech are preserved.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        pause_seconds: float = 1.0,
        speech_padding_seconds: float = 0.10,
        minimum_speech_seconds: float = 0.25,
        model_path: Path | None = None,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if pause_seconds <= 0:
            raise ValueError("pause_seconds must be positive")

        self._vad = SileroOnnxVad(model_path)
        self._threshold = threshold
        self._negative_threshold = max(threshold - 0.15, 0.01)
        self._pause_samples = round(pause_seconds * SAMPLE_RATE)
        self._padding_samples = round(speech_padding_seconds * SAMPLE_RATE)
        self._minimum_speech_samples = round(
            minimum_speech_seconds * SAMPLE_RATE
        )

        self.reset()

    def reset(self) -> None:
        """Prepare the gate for a new recording session."""
        self._vad.reset()
        self._current_sample = 0
        self._active_start: int | None = None
        self._temporary_end: int | None = None
        self._triggered = False

    def process(self, frame: np.ndarray) -> GateUpdate:
        """
        Process one 32 ms audio frame.

        `automatic_stop` becomes true after a detected utterance has been
        followed by one second of non-speech.
        """
        probability = self._vad.speech_probability(frame)
        self._current_sample += FRAME_SAMPLES

        if probability >= self._threshold:
            self._temporary_end = None

            if not self._triggered:
                self._triggered = True
                self._active_start = max(
                    0,
                    self._current_sample
                    - self._padding_samples
                    - FRAME_SAMPLES,
                )

            return GateUpdate(probability, False, None)

        if probability < self._negative_threshold and self._triggered:
            if self._temporary_end is None:
                self._temporary_end = self._current_sample

            silence_samples = self._current_sample - self._temporary_end
            if silence_samples >= self._pause_samples:
                end_sample = (
                    self._temporary_end
                    + self._padding_samples
                    - FRAME_SAMPLES
                )
                return GateUpdate(
                    probability,
                    True,
                    self._complete_segment(end_sample),
                )

        return GateUpdate(probability, False, None)

    def finish(self) -> SpeechSegment | None:
        """
        Finish a manually released hotkey recording.

        If speech had already paused, trailing silence is trimmed here too.
        """
        if not self._triggered:
            return None

        end_sample = self._current_sample
        if self._temporary_end is not None:
            end_sample = (
                self._temporary_end
                + self._padding_samples
                - FRAME_SAMPLES
            )

        return self._complete_segment(end_sample)

    def _complete_segment(self, end_sample: int) -> SpeechSegment | None:
        start_sample = self._active_start

        self._triggered = False
        self._active_start = None
        self._temporary_end = None

        if start_sample is None:
            return None

        end_sample = max(
            start_sample,
            min(end_sample, self._current_sample),
        )

        if end_sample - start_sample < self._minimum_speech_samples:
            return None

        return SpeechSegment(start_sample, end_sample)