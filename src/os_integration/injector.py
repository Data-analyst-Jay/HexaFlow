"""Unicode-safe text injection into the currently focused Windows field."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import platform
import threading
import time

import pyautogui


CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


class ClipboardError(RuntimeError):
    """Raised when another application keeps the Windows clipboard locked."""


if platform.system() == "Windows":
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _user32.OpenClipboard.argtypes = (wintypes.HWND,)
    _user32.OpenClipboard.restype = wintypes.BOOL
    _user32.CloseClipboard.argtypes = ()
    _user32.EmptyClipboard.argtypes = ()
    _user32.EmptyClipboard.restype = wintypes.BOOL
    _user32.IsClipboardFormatAvailable.argtypes = (wintypes.UINT,)
    _user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    _user32.GetClipboardData.argtypes = (wintypes.UINT,)
    _user32.GetClipboardData.restype = ctypes.c_void_p
    _user32.SetClipboardData.argtypes = (wintypes.UINT, ctypes.c_void_p)
    _user32.SetClipboardData.restype = ctypes.c_void_p

    _kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
    _kernel32.GlobalAlloc.restype = ctypes.c_void_p
    _kernel32.GlobalFree.argtypes = (ctypes.c_void_p,)
    _kernel32.GlobalLock.argtypes = (ctypes.c_void_p,)
    _kernel32.GlobalLock.restype = ctypes.c_void_p
    _kernel32.GlobalUnlock.argtypes = (ctypes.c_void_p,)
else:
    _user32 = None
    _kernel32 = None


class WindowsClipboard:
    """Minimal clipboard implementation without an additional dependency."""

    @staticmethod
    def get_unicode_text() -> str | None:
        _ensure_windows()
        _open_clipboard()

        try:
            assert _user32 is not None
            if not _user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
                return None

            assert _user32 is not None
            handle = _user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return None

            assert _kernel32 is not None
            pointer = _kernel32.GlobalLock(handle)
            if not pointer:
                return None

            try:
                return ctypes.wstring_at(pointer)
            finally:
                assert _kernel32 is not None
                _kernel32.GlobalUnlock(handle)
        finally:
            assert _user32 is not None
            _user32.CloseClipboard()

    @staticmethod
    def set_unicode_text(text: str) -> None:
        _ensure_windows()
        payload = text.encode("utf-16-le") + b"\x00\x00"
        memory = None

        _open_clipboard()
        try:
            assert _user32 is not None
            if not _user32.EmptyClipboard():
                _raise_clipboard_error("EmptyClipboard")

            assert _kernel32 is not None
            memory = _kernel32.GlobalAlloc(GMEM_MOVEABLE, len(payload))
            if not memory:
                _raise_clipboard_error("GlobalAlloc")

            assert _kernel32 is not None
            pointer = _kernel32.GlobalLock(memory)
            if not pointer:
                _raise_clipboard_error("GlobalLock")

            try:
                ctypes.memmove(pointer, payload, len(payload))
            finally:
                assert _kernel32 is not None
                _kernel32.GlobalUnlock(memory)

            # Windows owns this memory after SetClipboardData succeeds.
            assert _user32 is not None
            if not _user32.SetClipboardData(CF_UNICODETEXT, memory):
                _raise_clipboard_error("SetClipboardData")
            memory = None
        finally:
            if memory:
                assert _kernel32 is not None
                _kernel32.GlobalFree(memory)
            assert _user32 is not None
            _user32.CloseClipboard()


class TextInjector:
    """Inject text into whichever control currently owns keyboard focus."""

    def paste(
        self,
        text: str,
        *,
        restore_unicode_clipboard: bool = False,
        restore_delay_seconds: float = 0.20,
    ) -> None:
        """
        Paste Unicode text using Ctrl+V.

        Clipboard restoration is optional because restoring it can conflict with
        a user intentionally copying something immediately after dictation.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return

        previous_text = (
            WindowsClipboard.get_unicode_text()
            if restore_unicode_clipboard
            else None
        )

        WindowsClipboard.set_unicode_text(text)
        self._send_paste_shortcut()

        if restore_unicode_clipboard and previous_text is not None:
            timer = threading.Timer(
                restore_delay_seconds,
                self._restore_if_unchanged,
                args=(previous_text, text),
            )
            timer.daemon = True
            timer.start()

    def type_text(self, text: str) -> None:
        """
        Direct keystroke fallback for simple ASCII text.

        Clipboard pasting is preferred because it safely handles all Unicode
        transcript output, punctuation, and multi-line formatting.
        """
        if any(ord(character) > 127 for character in text):
            raise ValueError("Direct typing only supports ASCII; use paste() instead.")

        previous_pause = pyautogui.PAUSE
        try:
            pyautogui.PAUSE = 0.0
            for character in text:
                if character == "\n":
                    pyautogui.press("enter")
                elif character == "\t":
                    pyautogui.press("tab")
                else:
                    pyautogui.write(character)
        finally:
            pyautogui.PAUSE = previous_pause

    @staticmethod
    def _send_paste_shortcut() -> None:
        previous_pause = pyautogui.PAUSE
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.hotkey("ctrl", "v")
        finally:
            pyautogui.PAUSE = previous_pause

    @staticmethod
    def _restore_if_unchanged(previous_text: str, inserted_text: str) -> None:
        try:
            if WindowsClipboard.get_unicode_text() == inserted_text:
                WindowsClipboard.set_unicode_text(previous_text)
        except ClipboardError:
            pass


def _ensure_windows() -> None:
    if _user32 is None or _kernel32 is None:
        raise OSError("Text injection is only supported on Windows.")


def _open_clipboard(retries: int = 20, retry_delay_seconds: float = 0.005) -> None:
    _ensure_windows()
    assert _user32 is not None
    for _ in range(retries):
        if _user32.OpenClipboard(None):
            return
        time.sleep(retry_delay_seconds)

    _raise_clipboard_error("OpenClipboard")


def _raise_clipboard_error(operation: str) -> None:
    error_code = ctypes.get_last_error()
    raise ClipboardError(f"{operation} failed (Win32 error {error_code}).")