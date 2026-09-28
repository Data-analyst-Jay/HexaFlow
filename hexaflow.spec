"""ARM64 one-file HexaFlow build with local QAIRT, GenieX, and model assets."""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


ROOT = Path(SPEC).resolve().parent
APP_NAME = "WisprFlowClone"

# collect_all() adds package data, hidden imports, and native binaries. In a
# spec file these are the equivalents of --add-data, --hidden-import, and
# --add-binary respectively.
RUNTIME_PACKAGES = (
    "qai_appbuilder",
    "geniex",
    "onnxruntime",
    "sounddevice",
    "pystray",
    "pynput",
    "pyautogui",
    "uiautomation",
    "transformers",
    "tokenizers",
)

datas: list[tuple[str, str]] = [
    # Equivalent to: --add-data "models;models"
    (str(ROOT / "models"), "models"),
    (str(ROOT / "assets"), "assets"),
]
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = []

for package_name in RUNTIME_PACKAGES:
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package_name,
        include_py_files=False,
    )
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)

# Retain distribution metadata required by dynamic package discovery.
for distribution_name in (
    "qai-appbuilder",
    "geniex",
    "geniex-qairt",
    "onnxruntime-qnn",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
):
    datas.extend(copy_metadata(distribution_name))

a = Analysis(
    [str(ROOT / "src" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(ROOT / "packaging" / "pyi_rth_qairt.py")],
    excludes=(
        "torch",
        "tensorflow",
        "pytest",
        "IPython",
        "notebook",
    ),
    noarchive=False,
)

pyz = PYZ(a.pure)

# No COLLECT object means PyInstaller produces one self-extracting .exe.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    icon=str(ROOT / "assets" / "icons" / "tray_idle.ico"),
    console=False,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
)