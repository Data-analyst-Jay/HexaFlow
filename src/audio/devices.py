"""Terminal-based microphone discovery and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.audio.vad import SAMPLE_RATE


class InputDeviceSelectionError(RuntimeError):
    """Raised when an input device cannot be selected."""


@dataclass(frozen=True, slots=True)
class InputDevice:
    """A microphone exposed by PortAudio/sounddevice."""

    index: int
    name: str
    hostapi_name: str
    max_input_channels: int
    is_default: bool = False


def list_input_devices() -> list[InputDevice]:
    """Return every currently available audio-input device."""
    try:
        import sounddevice as sd
    except ImportError as error:
        raise InputDeviceSelectionError(
            "sounddevice is required for microphone capture."
        ) from error

    try:
        default_input_index = int(sd.default.device[0])
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
    except Exception as error:
        raise InputDeviceSelectionError(
            f"Could not discover audio devices: {error}"
        ) from error

    input_devices: list[InputDevice] = []

    for index, device in enumerate(devices):
        max_input_channels = int(device["max_input_channels"])
        if max_input_channels <= 0:
            continue

        # Do not show microphones that cannot support HexaFlow's required format.
        try:
            sd.check_input_settings(
                device=index,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
        except Exception:
            continue

        hostapi_index = int(device["hostapi"])
        hostapi_name = str(hostapis[hostapi_index]["name"])

        input_devices.append(
            InputDevice(
                index=index,
                name=str(device["name"]),
                hostapi_name=hostapi_name,
                max_input_channels=max_input_channels,
                is_default=index == default_input_index,
            )
        )

    return input_devices


def validate_input_device(device_index: int) -> None:
    """Ensure the selected device supports HexaFlow's capture format."""
    try:
        import sounddevice as sd

        sd.check_input_settings(
            device=device_index,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
    except Exception as error:
        raise InputDeviceSelectionError(
            "This microphone cannot be opened as a 16 kHz mono input: "
            f"{error}"
        ) from error


def prompt_for_input_device(
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> int:
    """
    Display available microphones and return the selected sounddevice index.

    The number shown to the user is separate from the PortAudio device index,
    so the list remains simple even if Windows assigns non-sequential IDs.
    """
    devices = list_input_devices()

    if not devices:
        raise InputDeviceSelectionError(
            "No audio-input devices were found. Connect or enable a microphone "
            "and start HexaFlow again."
        )

    default_device = next(
        (device for device in devices if device.is_default),
        devices[0],
    )

    output_func("")
    output_func("Available audio-input devices:")

    for number, device in enumerate(devices, start=1):
        default_marker = " [default]" if device.is_default else ""
        output_func(
            f"  {number}. {device.name}{default_marker} "
            f"— {device.max_input_channels} input channel(s), "
            f"{device.hostapi_name}"
        )

    output_func("")

    while True:
        try:
            response = input_func(
                f"Select microphone [default: {default_device.name}]: "
            ).strip()
        except (EOFError, KeyboardInterrupt) as error:
            raise InputDeviceSelectionError(
                "Microphone selection was cancelled."
            ) from error

        if not response:
            selected_device = default_device
        else:
            try:
                selection = int(response)
                selected_device = devices[selection - 1]
            except (ValueError, IndexError):
                output_func(
                    f"Please enter a number from 1 to {len(devices)}."
                )
                continue

        try:
            validate_input_device(selected_device.index)
        except InputDeviceSelectionError as error:
            output_func(
                f"'{selected_device.name}' cannot be used: {error}"
            )
            output_func("Please select another microphone.")
            continue

        output_func(f"Using microphone: {selected_device.name}")
        return selected_device.index