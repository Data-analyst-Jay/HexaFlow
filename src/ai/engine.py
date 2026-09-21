"""Background orchestration from VAD-cleaned audio to text injection."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import time
from typing import Callable

from src.ai.asr import WhisperSmallQairt
from src.audio.recorder import AudioRecording
from src.os_integration.injector import TextInjector


LOGGER = logging.getLogger(__name__)

CompletionCallback = Callable[
    ["DictationResult | None", Exception | None],
    None,
]


@dataclass(frozen=True, slots=True)
class DictationResult:
    text: str
    audio_seconds: float
    processing_seconds: float


class DictationEngine:
    """
    Serializes QAIRT access on one worker thread.

    The audio/VAD worker stays responsive while the NPU handles ASR.
    """

    def __init__(self, injector: TextInjector) -> None:
        self._injector = injector
        self._asr: WhisperSmallQairt | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hexaflow-asr",
        )
        self._closed = False

    def warm_up(self) -> Future[WhisperSmallQairt]:
        """Load QAIRT contexts before the first dictation."""
        future = self._executor.submit(self._get_asr)
        future.add_done_callback(self._report_warm_up_failure)
        return future

    def submit(
        self,
        recording: AudioRecording,
        on_complete: CompletionCallback,
    ) -> Future[None]:
        if self._closed:
            raise RuntimeError("The dictation engine has been closed.")

        return self._executor.submit(
            self._transcribe_and_inject,
            recording,
            on_complete,
        )

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

        if self._asr is not None:
            self._asr.close()
            self._asr = None

    def _get_asr(self) -> WhisperSmallQairt:
        if self._asr is None:
            self._asr = WhisperSmallQairt()
        return self._asr

    def _transcribe_and_inject(
        self,
        recording: AudioRecording,
        on_complete: CompletionCallback,
    ) -> None:
        started_at = time.perf_counter()

        try:
            transcript = self._get_asr().transcribe(
                recording.samples,
                recording.sample_rate,
            )

            if transcript:
                self._injector.paste(transcript)

            result = DictationResult(
                text=transcript,
                audio_seconds=recording.duration_seconds,
                processing_seconds=time.perf_counter() - started_at,
            )
            on_complete(result, None)
        except Exception as error:
            LOGGER.exception("ASR or text injection failed.")
            on_complete(None, error)

    @staticmethod
    def _report_warm_up_failure(
        future: Future[WhisperSmallQairt],
    ) -> None:
        try:
            future.result()
        except Exception:
            LOGGER.exception("QAIRT ASR warm-up failed.")