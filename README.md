# HexaFlow

HexaFlow is a private, push-to-talk Windows dictation application built for Snapdragon PCs. It captures speech from a selected microphone, transcribes it locally with Whisper on the Qualcomm Hexagon NPU, conservatively formats the transcript with Qwen3, and pastes the result into the currently focused application.

The application runs from the system tray and does not require a cloud transcription service or API key.

## Features

- **Local NPU inference** — Whisper Small transcription and Qwen3 0.6B formatting run through Qualcomm AI Runtime (QAIRT).
- **Push-to-talk dictation** — hold `Ctrl` + `Shift` to record; release either key to finish.
- **Speech-aware recording** — Silero VAD removes leading/trailing silence and can finish an utterance after a pause.
- **Context-aware, conservative formatting** — preserves the speaker's wording while correcting obvious spacing, capitalization, punctuation, and grammar issues.
- **Direct text insertion** — pastes Unicode text into the application that has keyboard focus.
- **System-tray status** — the tray icon indicates idle, listening, and processing states, with a Quit option.
- **Private by design** — model assets are loaded from disk; the dictation path does not call a cloud inference API.

## Getting Started

### Prerequisites

HexaFlow currently targets the following environment:

- Windows 11 on ARM64 (`ARM64`/`AArch64`) with a Qualcomm Snapdragon processor and Hexagon NPU.
- Python 3.12 (Python 3.13 and later are not supported by this project configuration).(You can download from here: https://www.python.org/ftp/python/3.12.10/python-3.12.10-arm64.exe?utm_source=chatgpt.com)
- [uv](powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex") for dependency and virtual-environment management.
- Visual Studio Code with its integrated PowerShell terminal.
- A microphone that supports 16 kHz mono capture.
- The compatible Qualcomm QAIRT/AppBuilder runtime and model assets described below.

> HexaFlow validates the operating system and processor architecture before loading the QAIRT models. It will not run its NPU inference pipeline on x64 Windows, macOS, Linux, or a non-Snapdragon PC.

### Installation

Open the project folder in VS Code, then open **Terminal → New Terminal**. The terminal profile should be PowerShell.

Install the project dependencies:

```powershell
uv sync
```
After running:
``` uv suyc ```

If the error comes in installing the geniex-qairt module then run these commands:
``` 
$env:PYTHONHTTPSVERIFY = "0”
uv pip install --upgrade certifi
$env:SSL_CERT_FILE = $(python -c "import certifi; print(certifi.where())")
uv add geniex 
```

Install the local models:

1. vad model:
``` Invoke-WebRequest -Uri "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx" -OutFile ".\models\silero_vad.onnx" ```

2. whisper-small-quantized model:
``` qai-hub-models fetch Whisper-Small-Quantized ` --runtime qnn_context_binary ` --precision w8a16 ` --chipset qualcomm-snapdragon-x-elite ```

Rename the model file as whisper_small_quantized and place it under .models/ directory.

3. Download hf modules for whisper model:
``` hf download openai/whisper-small ` config.json ` preprocessor_config.json ` tokenizer.json ` tokenizer_config.json ` special_tokens_map.json ` vocab.json ` merges.txt ` normalizer.json ` --local-dir .\models\whisper-small-quantized\hf ```
4. Download qwen3-0.6b slm:
``` qai-hub-models fetch Qwen3-0.6B --runtime geniex_qairt --precision w4a16 --chipset "Snapdragon X Elite”  ```

Rename the directory containing the slm as 'qwen3_0_6b_w4a16' and place it under .models/ directory.

5. Now you are ready for running the application

### Provision Local Models

Large model files are intentionally excluded from version control. Before starting HexaFlow, provision the compatible QAIRT bundles into `models/` with this layout:

```text
models/
├── silero_vad.onnx
├── whisper_small_quantized/
│   ├── encoder.bin
│   ├── decoder.bin
│   └── hf/
│       ├── config.json
│       ├── preprocessor_config.json
│       └── tokenizer.json
└── qwen3_0_6b_w4a16/
    ├── metadata.json
    └── *.bin
```

Use model bundles compatible with the versions pinned in `pyproject.toml`. HexaFlow only loads these assets locally; it does not download models at startup. Keep the model files out of Git—`models/.gitignore` is configured for that purpose.

## Usage

### Run Locally

From the VS Code PowerShell terminal at the repository root, run:

```powershell
uv run python -m src.main
```

On launch, HexaFlow lists compatible microphone devices. Enter the displayed number for the microphone to use, or press `Enter` to accept the default device. The app then appears in the Windows system tray while its local models warm up.

To dictate:

1. Place the cursor in the target text field or document.
2. Hold `Ctrl` + `Shift` while speaking.
3. Release either key to stop recording. A spoken utterance can also finish automatically after approximately one second of silence.
4. Wait for the tray icon to return to **Idle**. HexaFlow pastes the processed text into the focused field.


### Tray States

| State | Meaning |
| --- | --- |
| Idle | Ready for the next dictation. |
| Listening | Microphone capture is in progress. |
| Processing | Speech is being transcribed, formatted, and inserted. |

## Configuration

HexaFlow has no required environment variables or API keys. Its core behavior is configured in source for the current prototype:

- The recording hotkey is `Ctrl` + `Shift`.
- Audio is captured as 16 kHz mono.
- Automatic speech completion occurs after roughly one second of non-speech.
- The formatter receives limited context from the active window and focused field to disambiguate dictation. It is instructed to use that context only as reference data.
- Inserted text is sent through the Windows clipboard and `Ctrl` + `V`; the clipboard contents may therefore be replaced by the dictated text.

## Logs and Troubleshooting

Logs are written to:

```text
%LOCALAPPDATA%\HexaFlow\logs\hexaflow.log
```

In PowerShell, open the log directory with:

```powershell
explorer "$env:LOCALAPPDATA\HexaFlow\logs"
```

Common startup issues:

- **No microphone is listed:** connect or enable a microphone in Windows Settings, then run the command again.
- **Missing model asset:** check that every required file is present in the `models/` layout above.
- **Unsupported platform message:** use native Windows ARM64 Python on a supported Snapdragon PC; emulated or x64 Python is not sufficient for QAIRT HTP inference.
- **Text is pasted into the wrong place:** make sure the intended text field has focus before pressing the hotkey, and avoid changing focus until processing finishes.
- **Another application blocks insertion:** Windows clipboard access or simulated `Ctrl` + `V` may be restricted in elevated, protected, or remote applications.


## Project Structure

```text
src/
├── ai/               # QAIRT Whisper ASR and Qwen3 formatter
├── audio/            # Microphone capture, device selection, and VAD
├── os_integration/   # Focused-context capture and Windows text injection
├── ui/               # Global hotkey and system-tray UI
└── main.py           # Application entry point
models/               # Local model assets (not committed)
assets/               # Tray icons
scripts/              # Development and packaging helpers
```
