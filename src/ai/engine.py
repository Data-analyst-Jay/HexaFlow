"""Background orchestration from VAD-cleaned audio to streamed text injection."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import time
from typing import Callable

from src.ai.asr import WhisperSmallQairt
from src.ai.slm import Gemma4E4BQairt, SlmModelError
from src.audio.recorder import AudioRecording
from src.os_integration.context import FocusedContext
from src.os_integration.injector import TextInjector


LOGGER = logging.getLogger(__name__)

CompletionCallback = Callable[
    ["DictationResult | None", Exception | None],
    None,
]


@dataclass(frozen=True, slots=True)
class DictationResult:
    text: str
    raw_text: str
    used_slm: bool
    audio_seconds: float
    processing_seconds: float


class DictationEngine:
    """
    Serializes ASR and SLM access on one worker thread.

    Whisper and Gemma each retain their own loaded NPU session. Running them
    sequentially avoids competing for the Hexagon NPU during one dictation.
    """

    def __init__(self, injector: TextInjector) -> None:
        self._injector = injector
        self._asr: WhisperSmallQairt | None = None
        self._slm: Gemma4E4BQairt | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hexaflow-inference",
        )
        self._closed = False

    def warm_up(self) -> Future[None]:
        """Load Whisper and the separate GenieX QAIRT SLM before dictation."""
        future = self._executor.submit(self._warm_up_models)
        future.add_done_callback(self._report_warm_up_failure)
        return future

    def submit(
        self,
        recording: AudioRecording,
        context: FocusedContext,
        on_complete: CompletionCallback,
    ) -> Future[None]:
        if self._closed:
            raise RuntimeError("The dictation engine has been closed.")

        return self._executor.submit(
            self._transcribe_format_and_inject,
            recording,
            context,
            on_complete,
        )

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

        if self._slm is not None:
            self._slm.close()
            self._slm = None

        if self._asr is not None:
            self._asr.close()
            self._asr = None

    def _warm_up_models(self) -> None:
        self._get_asr()
        self._get_slm()

    def _get_asr(self) -> WhisperSmallQairt:
        if self._asr is None:
            self._asr = WhisperSmallQairt()
        return self._asr

    def _get_slm(self) -> Gemma4E4BQairt:
        if self._slm is None:
            self._slm = Gemma4E4BQairt()
        return self._slm

    def _transcribe_format_and_inject(
        self,
        recording: AudioRecording,
        context: FocusedContext,
        on_complete: CompletionCallback,
    ) -> None:
        started_at = time.perf_counter()

        try:
            raw_transcript = self._get_asr().transcribe(
                recording.samples,
                recording.sample_rate,
            )

            final_text, used_slm = self._format_and_stream(
                raw_transcript,
                context,
            )

            result = DictationResult(
                text=final_text,
                raw_text=raw_transcript,
                used_slm=used_slm,
                audio_seconds=recording.duration_seconds,
                processing_seconds=time.perf_counter() - started_at,
            )
            on_complete(result, None)

        except Exception as error:
            LOGGER.exception("Dictation pipeline failed.")
            on_complete(None, error)

    def _format_and_stream(
        self,
        raw_transcript: str,
        context: FocusedContext,
    ) -> tuple[str, bool]:
        """Format dictation, validate the response, then insert it once."""
        if not raw_transcript:
            return "", False

        try:
            formatted_text = "".join(
                self._get_slm().stream_format(raw_transcript, context)
            ).strip()

        except SlmModelError:
            LOGGER.exception(
                "SLM failed; using raw ASR fallback."
            )
            self._injector.paste(raw_transcript)
            return raw_transcript, False

        # Reject an SLM response that is actually an echo of the context JSON.
        # The real dictation must never contain these internal context-field names.
        echo_markers = (
            '"active_window"',
            '"focused_control"',
            '"focused_field_tail"',
            '"raw_dictation"',
        )
        if not formatted_text or any(marker in formatted_text for marker in echo_markers):
            LOGGER.warning(
                "SLM returned empty text or echoed its context; using raw ASR fallback."
            )
            self._injector.paste(raw_transcript)
            return raw_transcript, False

        self._injector.paste(formatted_text)
        return formatted_text, True

    @staticmethod
    def _report_warm_up_failure(future: Future[None]) -> None:
        try:
            future.result()
        except Exception:
            LOGGER.exception("QAIRT ASR/SLM warm-up failed.")