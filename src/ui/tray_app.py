"""System-tray UI and application state."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import sys
import threading
from typing import Callable

from PIL import Image, ImageDraw
import pystray


class TrayState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"


class SystemTrayApp:
    """Owns the invisible tray UI and its three visual states."""

    def __init__(
        self,
        title: str = "HexaFlow",
        on_exit: Callable[[], None] | None = None,
    ) -> None:
        self._title = title
        self._on_exit = on_exit
        self._state = TrayState.IDLE
        self._state_lock = threading.Lock()
        self._running = threading.Event()

        self._icons = self._load_icons()
        self._icon = pystray.Icon(
            name="hexaflow",
            icon=self._icons[TrayState.IDLE],
            title=f"{self._title} — Idle",
            menu=self._build_menu(),
        )

    @property
    def state(self) -> TrayState:
        with self._state_lock:
            return self._state

    def set_state(self, state: TrayState) -> None:
        """Update the icon and tooltip without opening a visible window."""
        state = TrayState(state)

        with self._state_lock:
            self._state = state

        self._icon.icon = self._icons[state]
        self._icon.title = f"{self._title} — {state.value.title()}"
        self._icon.update_menu()

    def run(self, on_ready: Callable[[], None] | None = None) -> None:
        """Run the tray loop in the calling thread."""
        if self._running.is_set():
            return

        def setup(icon: pystray.Icon) -> None:
            # pystray requires this when a custom setup callback is supplied.
            icon.visible = True

            if on_ready is not None:
                on_ready()

        self._running.set()
        try:
            self._icon.run(setup=setup)
        finally:
            self._running.clear()

    def start(self, on_ready: Callable[[], None] | None = None) -> threading.Thread:
        """Optional non-blocking tray startup for apps with another main loop."""
        thread = threading.Thread(
            target=self.run,
            args=(on_ready,),
            name="hexaflow-tray",
            daemon=True,
        )
        thread.start()
        return thread

    def stop(self) -> None:
        self._icon.stop()

    def _build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem(
                lambda _item: f"State: {self.state.value.title()}",
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit HexaFlow", self._quit),
        )

    def _quit(self, _icon: pystray.Icon, _item: object) -> None:
        if self._on_exit is not None:
            self._on_exit()
        self.stop()

    def _load_icons(self) -> dict[TrayState, Image.Image]:
        project_root = self._project_root()
        icon_paths = {
            TrayState.IDLE: project_root / "assets" / "icons" / "tray_idle.ico",
            TrayState.LISTENING: project_root / "assets" / "icons" / "tray_listen.ico",
            TrayState.PROCESSING: project_root / "assets" / "icons" / "tray_process.ico",
        }

        fallback_colours = {
            TrayState.IDLE: "#5F6368",
            TrayState.LISTENING: "#E53935",
            TrayState.PROCESSING: "#1E88E5",
        }

        icons: dict[TrayState, Image.Image] = {}
        for state, path in icon_paths.items():
            try:
                with Image.open(path) as image:
                    icons[state] = image.convert("RGBA").copy()
            except OSError:
                # Keeps the shell usable if an icon is missing during development.
                icons[state] = self._fallback_icon(fallback_colours[state])

        return icons

    @staticmethod
    def _project_root() -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS)  # type: ignore[attr-defined]
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _fallback_icon(colour: str) -> Image.Image:
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((8, 8, 56, 56), fill=colour)
        return image