"""Focused-field context capture through Windows UI Automation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import uiautomation as auto
except ImportError:
    auto = None

import logging

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class FocusedContext:
    active_window_name: str
    focused_control_name: str
    focused_text_tail: str


def capture_context(character_limit: int = 300) -> FocusedContext:
    """
    Return focused-field context.

    UI Automation is COM-based, so initialization must happen in the exact
    thread that performs the UIA calls. The pynput hotkey callback is one such
    background thread.
    """
    if character_limit < 1:
        raise ValueError("character_limit must be at least 1")

    if auto is None:
        return FocusedContext("", "", "")

    try:
        # Initializes COM for this thread and releases it safely on exit.
        with auto.UIAutomationInitializerInThread():
            foreground = _safe_call(auto.GetForegroundControl)
            focused_control = _safe_call(auto.GetFocusedControl)

            window_name = _control_name(foreground)
            control_name = _control_name(focused_control)
            field_text = _read_text_value(focused_control)

            return FocusedContext(
                active_window_name=window_name,
                focused_control_name=control_name,
                focused_text_tail=field_text[-character_limit:],
            )

    except Exception:
        # Do not log captured text: it may contain sensitive user content.
        LOGGER.exception("UI Automation context capture failed.")
        return FocusedContext("", "", "")

def _read_text_value(control: Any) -> str:
    """Try ValuePattern first, then TextPattern for rich-text controls."""
    if control is None:
        return ""

    try:
        value_pattern = control.GetValuePattern()
        value = getattr(value_pattern, "Value", None)
        if value is not None:
            return str(value).rstrip("\x00")
    except Exception:
        pass

    try:
        text_pattern = control.GetTextPattern()
        document_range = getattr(text_pattern, "DocumentRange", None)
        if document_range is not None:
            return str(document_range.GetText(-1)).rstrip("\x00")
    except Exception:
        pass

    return ""


def _control_name(control: Any) -> str:
    try:
        return str(getattr(control, "Name", "") or "")
    except Exception:
        return ""


def _safe_call(function: Any) -> Any:
    try:
        return function()
    except Exception:
        return None