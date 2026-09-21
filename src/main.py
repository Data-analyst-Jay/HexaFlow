"""Phase 1 shell plus Phase 2 microphone capture and speech gating."""

from __future__ import annotations

import logging
# from pathlib import Path
# import threading
from src.ai.engine import DictationEngine, DictationResult

from src.audio.recorder import AudioRecorder, AudioRecording
from src.os_integration.context import FocusedContext, capture_context
from src.os_integration.injector import TextInjector
from src.ui.hotkeys import PushToTalkHotkey
from src.ui.tray_app import SystemTrayApp, TrayState


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("hexaflow")

# CAPTURE_OUTPUT_PATH = Path(
#     r"F:\Jay Gehlot\QAI Labs\hexaflow_last_recording.wav"
# )


class HexaFlowPhaseOne:
    """Connects the working Phase 1 shell to Phase 2 audio capture."""

    def __init__(self) -> None:
        self._context: FocusedContext | None = None
        self._injector = TextInjector()  # Used by the later ASR phase.
        self._engine = DictationEngine(self._injector)

        self._tray = SystemTrayApp(on_exit=self.stop)
        self._recorder = AudioRecorder(
            on_recording_complete=self._recording_complete,
        )
        self._hotkeys = PushToTalkHotkey(
            on_recording_started=self._recording_started,
            on_recording_finished=self._recording_finished,
            on_error=self._hotkey_error,
        )

    def run(self) -> None:
        try:
            self._tray.run(on_ready=self._services_ready)
        finally:
            self.stop()

    def stop(self) -> None:
        self._hotkeys.stop()
        self._recorder.close()
        self._engine.close()

    def _services_ready(self) -> None:
        self._engine.warm_up()
        self._hotkeys.start()

    def _recording_started(self) -> None:
        # Capture context before future processing can change focused control.
        self._context = capture_context()
        self._tray.set_state(TrayState.LISTENING)

        try:
            self._recorder.start()
        except Exception:
            self._tray.set_state(TrayState.IDLE)
            raise

        LOGGER.info(
            "Recording started in %r with %d context characters.",
            self._context.active_window_name,
            len(self._context.focused_text_tail),
        )

    def _recording_finished(self, duration_seconds: float) -> None:
        # The worker drains queued audio before reporting the final utterance.
        self._hotkeys.set_enabled(False)
        LOGGER.info(
            "Hotkey released after %.2f seconds; finalizing audio.",
            duration_seconds,
        )
        self._recorder.stop()

    def _recording_complete(
        self,
        recording: AudioRecording | None,
        automatic_stop: bool,
    ) -> None:
        self._hotkeys.set_enabled(False)

        if automatic_stop:
            self._hotkeys.complete_recording()

        if recording is None:
            LOGGER.info("No valid speech was detected;  nothing to transcribe.")
            self._tray.set_state(TrayState.IDLE)
            self._hotkeys.set_enabled(True)
            return

        self._tray.set_state(TrayState.PROCESSING)

        try:
            self._engine.submit(recording, self.    _dictation_complete)
        except Exception:
            LOGGER.exception("Could not queue dictation.")
            self._tray.set_state(TrayState.IDLE)
            self._hotkeys.set_enabled(True)

    def _dictation_complete(
        self,
        result: DictationResult | None,
        error: Exception | None,
    ) -> None:
        try:
            if error is not None:
                LOGGER.error("Dictation failed: %s", error)
            elif result is None or not result.text:
                LOGGER.info("Whisper returned no transcript.    ")
            else:
                LOGGER.info(
                    "Injected raw transcript from %.2f  seconds of speech in %.2f seconds.",
                    result.audio_seconds,
                    result.processing_seconds,
                )
        finally:
            self._tray.set_state(TrayState.IDLE)
            self._hotkeys.set_enabled(True)

    @staticmethod
    def _hotkey_error(error: Exception) -> None:
        LOGGER.exception("Global hotkey failure.", exc_info=error)


def main() -> None:
    HexaFlowPhaseOne().run()


if __name__ == "__main__":
    main()