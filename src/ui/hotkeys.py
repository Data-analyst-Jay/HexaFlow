"""Global push-to-talk hotkey handling."""

from __future__ import annotations

import threading
import time
from typing import Callable

from pynput import keyboard


class PushToTalkHotkey:
    """Starts recording while Ctrl and Shift are both held."""

    _CONTROL_KEYS = {
        keyboard.Key.ctrl,
        keyboard.Key.ctrl_l,
        keyboard.Key.ctrl_r,
    }
    _SHIFT_KEYS = {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r}

    def __init__(
        self,
        on_recording_started: Callable[[], None],
        on_recording_finished: Callable[[float], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._on_recording_started = on_recording_started
        self._on_recording_finished = on_recording_finished
        self._on_error = on_error

        self._lock = threading.Lock()
        self._held_modifiers: set[str] = set()
        self._enabled = True
        self._recording = False
        self._awaiting_hotkey_release = False
        self._started_at = 0.0
        self._listener: keyboard.Listener | None = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def set_enabled(self, enabled: bool) -> None:
        """Disable new recordings while an utterance is processing."""
        with self._lock:
            self._enabled = enabled

    def complete_recording(self) -> None:
        """
        Finish an automatically VAD-stopped recording without invoking the
        release callback a second time when Ctrl/Space are released.
        """
        with self._lock:
            if self._recording:
                self._recording = False
                self._started_at = 0.0
                self._awaiting_hotkey_release = True

    def start(self) -> None:
        if self._listener is not None and self._listener.is_alive():
            return

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self) -> None:
        with self._lock:
            self._recording = False
            self._awaiting_hotkey_release = False
            self._held_modifiers.clear()

        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(
        self,
        key: keyboard.Key | keyboard.KeyCode | None,
    ) -> None:
        modifier = self._modifier_name(key)
        if modifier is None:
            return

        should_start = False
        with self._lock:
            self._held_modifiers.add(modifier)

            if (
                self._enabled
                and not self._recording
                and not self._awaiting_hotkey_release
                and {"ctrl", "shift"}.issubset(self._held_modifiers)
            ):
                self._recording = True
                self._started_at = time.monotonic()
                should_start = True

        if should_start:
            try:
                self._on_recording_started()
            except Exception as error:
                with self._lock:
                    self._recording = False
                self._report_error(error)

    def _on_release(
        self,
        key: keyboard.Key | keyboard.KeyCode | None,
    ) -> None:
        modifier = self._modifier_name(key)
        if modifier is None:
            return

        duration = 0.0
        should_finish = False

        with self._lock:
            self._held_modifiers.discard(modifier)
            hotkey_is_held = {"ctrl", "shift"}.issubset(
                self._held_modifiers
            )

            if self._awaiting_hotkey_release and not hotkey_is_held:
                self._awaiting_hotkey_release = False

            if self._recording and not hotkey_is_held:
                self._recording = False
                duration = max(0.0, time.monotonic() - self._started_at)
                should_finish = True

        if should_finish:
            try:
                self._on_recording_finished(duration)
            except Exception as error:
                self._report_error(error)

    def _modifier_name(
        self,
        key: keyboard.Key | keyboard.KeyCode | None,
    ) -> str | None:
        if key in self._CONTROL_KEYS:
            return "ctrl"
        if key in self._SHIFT_KEYS:
            return "shift"
        return None

    def _report_error(self, error: Exception) -> None:
        if self._on_error is not None:
            self._on_error(error)