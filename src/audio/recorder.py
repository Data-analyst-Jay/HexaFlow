"""Thread-safe microphone capture and WAV export."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from queue import Empty, Queue
import threading
from typing import Callable, Protocol
import wave

import numpy as np

from src.audio.vad import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    SileroSpeechGate,
    SpeechSegment,
)


LOGGER = logging.getLogger(__name__)


class _InputStreamLike(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

RecordingCallback = Callable[["AudioRecording | None", bool], None]


@dataclass(frozen=True, slots=True)
class AudioRecording:
    """A cleaned, mono float32 utterance ready for future Whisper inference."""

    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


class AudioRecorder:
    """
    Captures microphone frames in a callback and runs VAD on a worker thread.

    The sounddevice callback only copies audio into a Queue, keeping the
    real-time audio thread fast and deterministic.
    """

    def __init__(
        self,
        on_recording_complete: RecordingCallback,
        *,
        device: int | str | None = None,
        queue_blocks: int = 250,
    ) -> None:
        if queue_blocks < 2:
            raise ValueError("queue_blocks must be at least 2")

        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError(
                "sounddevice is required for microphone capture. "
                "Run `uv sync` after updating pyproject.toml."
            ) from error

        self._sounddevice = sd
        self._on_recording_complete = on_recording_complete
        self._device = device
        self._gate = SileroSpeechGate()

        self._lock = threading.RLock()
        self._recording = False
        self._closed = False
        self._session_id = 0
        self._stream: _InputStreamLike | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frames: Queue[np.ndarray] = Queue(maxsize=queue_blocks)
        self._queue_blocks = queue_blocks
        self._queue_overflowed = False

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def start(self) -> None:
        """Start a 16 kHz, mono, float32 microphone stream."""
        stream: _InputStreamLike | None = None

        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("The recorder has been closed.")
                if self._recording:
                    raise RuntimeError("A recording is already in progress.")

                self._session_id += 1
                session_id = self._session_id
                self._recording = True
                self._queue_overflowed = False
                self._stop_event = threading.Event()
                self._frames = Queue(maxsize=self._queue_blocks)
                self._gate.reset()

                stream = self._sounddevice.InputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=FRAME_SAMPLES,
                    channels=1,
                    dtype="float32",
                    device=self._device,
                    callback=lambda data, frames, time_info, status: (
                        self._audio_callback(
                            session_id,
                            data,
                            frames,
                            time_info,
                            status,
                        )
                    ),
                )
                self._stream = stream
                stream.start()

                worker = threading.Thread(
                    target=self._consume_frames,
                    args=(session_id, self._frames, self._stop_event),
                    name="hexaflow-audio-vad",
                    daemon=True,
                )
                self._worker = worker
                worker.start()

        except Exception:
            with self._lock:
                self._recording = False
                self._stream = None
                self._stop_event.set()

            if stream is not None:
                stream.close()

            raise

    def stop(self) -> None:
        """Stop microphone input; queued frames are still VAD-processed."""
        with self._lock:
            session_id = self._session_id

        self._stop_input(session_id)

    def close(self) -> None:
        """Stop capture during application shutdown without saving a WAV."""
        with self._lock:
            self._closed = True
            worker = self._worker
            session_id = self._session_id

        self._stop_input(session_id)

        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)

    @staticmethod
    def save_wav(recording: AudioRecording, destination: Path | str) -> Path:
        """Save float32 mono audio as standard 16-bit PCM WAV."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        samples = np.asarray(recording.samples, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("A non-empty mono recording is required.")

        pcm16 = np.rint(
            np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max
        ).astype("<i2")

        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(recording.sample_rate)
            wav_file.writeframes(pcm16.tobytes())

        return destination

    def _audio_callback(
        self,
        session_id: int,
        input_data: np.ndarray,
        frames: int,
        _time_info: object,
        status: object,
    ) -> None:
        """Copy data from sounddevice's real-time callback into the Queue."""
        if status:
            LOGGER.warning("Microphone stream status: %s", status)

        if frames != FRAME_SAMPLES:
            LOGGER.error(
                "Unexpected microphone frame size: %d (expected %d).",
                frames,
                FRAME_SAMPLES,
            )
            return

        with self._lock:
            if not self._recording or session_id != self._session_id:
                return
            frame_queue = self._frames

        # sounddevice reuses its callback buffer, so this copy is required.
        frame = np.asarray(input_data[:, 0], dtype=np.float32).copy()

        try:
            frame_queue.put_nowait(frame)
        except Exception:
            with self._lock:
                self._queue_overflowed = True

            LOGGER.error(
                "Audio queue overflowed; this utterance will be discarded."
            )

    def _consume_frames(
        self,
        session_id: int,
        frame_queue: Queue[np.ndarray],
        stop_event: threading.Event,
    ) -> None:
        frames: list[np.ndarray] = []
        automatic_stop = False
        final_segment: SpeechSegment | None = None
        recording: AudioRecording | None = None

        try:
            while True:
                try:
                    frame = frame_queue.get(timeout=0.1)
                except Empty:
                    if stop_event.is_set():
                        break
                    continue

                frames.append(frame)
                update = self._gate.process(frame)

                if update.automatic_stop:
                    automatic_stop = True
                    final_segment = update.segment
                    self._stop_input(session_id)
                    break

            if not automatic_stop:
                final_segment = self._gate.finish()

            with self._lock:
                queue_overflowed = (
                    session_id == self._session_id
                    and self._queue_overflowed
                )

            if queue_overflowed:
                LOGGER.error(
                    "Discarding utterance because microphone data was dropped."
                )
            else:
                recording = self._build_recording(frames, final_segment)

        except Exception:
            LOGGER.exception("Audio capture or VAD processing failed.")
            self._stop_input(session_id)

        finally:
            self._finish_session(session_id, recording, automatic_stop)

    def _stop_input(self, session_id: int) -> None:
        """Stop PortAudio safely from either the hotkey or VAD worker thread."""
        with self._lock:
            if session_id != self._session_id or not self._recording:
                return

            self._recording = False
            self._stop_event.set()
            stream = self._stream
            self._stream = None

        if stream is None:
            return

        try:
            stream.stop()
        except Exception:
            LOGGER.warning("Could not stop microphone stream cleanly.", exc_info=True)
        finally:
            try:
                stream.close()
            except Exception:
                LOGGER.warning(
                    "Could not close microphone stream cleanly.",
                    exc_info=True,
                )

    def _finish_session(
        self,
        session_id: int,
        recording: AudioRecording | None,
        automatic_stop: bool,
    ) -> None:
        with self._lock:
            if session_id != self._session_id:
                return

            self._recording = False
            self._stream = None
            self._worker = None
            deliver_callback = not self._closed

        if not deliver_callback:
            return

        try:
            self._on_recording_complete(recording, automatic_stop)
        except Exception:
            LOGGER.exception("The recording-complete callback failed.")

    @staticmethod
    def _build_recording(
        frames: list[np.ndarray],
        segment: SpeechSegment | None,
    ) -> AudioRecording | None:
        if not frames or segment is None:
            return None

        waveform = np.concatenate(frames).astype(np.float32, copy=False)
        start = max(0, min(segment.start_sample, len(waveform)))
        end = max(start, min(segment.end_sample, len(waveform)))

        if end <= start:
            return None

        return AudioRecording(samples=waveform[start:end].copy())