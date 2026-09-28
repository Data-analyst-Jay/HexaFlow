"""PyInstaller runtime hook: expose bundled QAIRT/GenieX DLL locations."""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Keep handles alive for the entire process; Windows removes a DLL directory when
# its handle is garbage-collected.
_DLL_DIRECTORY_HANDLES: list[object] = []


def _add_dll_directory(path: Path) -> None:
    if not path.is_dir():
        return

    try:
        handle = os.add_dll_directory(str(path))
    except (AttributeError, OSError):
        return

    _DLL_DIRECTORY_HANDLES.append(handle)


if sys.platform == "win32":
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))

    # QAIRT and GenieX packages preserve these locations when collected by the
    # spec file. Adding them before imports lets native modules resolve QNN DLLs.
    dll_directories = (
        bundle_root,
        bundle_root / "qai_appbuilder",
        bundle_root / "qai_appbuilder" / "libs",
        bundle_root / "geniex",
        bundle_root / "geniex" / "libs",
        bundle_root / "onnxruntime" / "capi",
    )

    for directory in dll_directories:
        _add_dll_directory(directory)

    # Some native loaders still consult PATH rather than AddDllDirectory.
    existing_path = os.environ.get("PATH", "")
    bundled_paths = [
        str(directory)
        for directory in dll_directories
        if directory.is_dir()
    ]
    os.environ["PATH"] = os.pathsep.join([*bundled_paths, existing_path])