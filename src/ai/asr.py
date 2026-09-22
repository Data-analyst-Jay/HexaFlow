"""QAIRT/HTP Whisper-Small-Quantized transcription."""

from __future__ import annotations

from pathlib import Path
import platform
import threading

import numpy as np


SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS
MAX_DECODE_TOKENS = 200
MASK_NEGATIVE_VALUE = -100.0


class AsrModelError(RuntimeError):
    """Raised when the local QAIRT Whisper deployment is invalid."""


class WhisperSmallQairt:
    """
    Runs Whisper-Small-Quantized on the Qualcomm Hexagon HTP/NPU.

    QAI AppBuilder is a Python wrapper around QAIRT. No llama.cpp, cloud API,
    or CPU ASR inference is used here. CPU work is limited to Whisper's small
    log-mel feature frontend and token decoding.
    """

    _runtime_lock = threading.Lock()
    _runtime_configured = False

    def __init__(self, model_dir: Path | None = None) -> None:
        self._require_windows_arm64()

        self._model_dir = model_dir or (
            Path(__file__).resolve().parents[2]
            / "models"
            / "whisper_small_quantized"
        )
        self._encoder_path = self._model_dir / "encoder.bin"
        self._decoder_path = self._model_dir / "decoder.bin"
        self._hf_assets_dir = self._model_dir / "hf"
        self._inference_lock = threading.Lock()
        self._closed = False

        self._require_files()

        try:
            from qai_appbuilder import (
                DataType,
                LogLevel,
                PerfProfile,
                ProfilingLevel,
                QNNConfig,
                QNNContext,
                Runtime,
            )
            from transformers import (
                WhisperConfig,
                WhisperFeatureExtractor,
                WhisperTokenizer,
            )
        except ImportError as error:
            raise AsrModelError(
                "Install qai-appbuilder and transformers in the ARM64 runtime."
            ) from error

        self._perf_profile = PerfProfile
        self._context_type = QNNContext
        self._data_type = DataType

        self._config = WhisperConfig.from_pretrained(
            self._hf_assets_dir,
            local_files_only=True,
        )
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self._hf_assets_dir,
            local_files_only=True,
        )
        self._tokenizer = WhisperTokenizer.from_pretrained(
            self._hf_assets_dir,
            local_files_only=True,
        )

        with self._runtime_lock:
            if not type(self)._runtime_configured:
                QNNConfig.Config(
                    runtime=Runtime.HTP,
                    log_level=LogLevel.WARN,
                    profiling_level=ProfilingLevel.BASIC,
                )
                type(self)._runtime_configured = True

        # FLOAT mode lets AppBuilder perform the required QAIRT boundary
        # conversions for the AI Hub w8a16 context binaries.
        self._encoder = QNNContext(
            "hexaflow_whisper_encoder",
            str(self._encoder_path),
            input_data_type=DataType.FLOAT,
            output_data_type=DataType.FLOAT,
        )
        self._decoder = QNNContext(
            "hexaflow_whisper_decoder",
            str(self._decoder_path),
            input_data_type=DataType.FLOAT,
            output_data_type=DataType.FLOAT,
        )

        self._encoder_input_names = list(self._encoder.getInputName())
        self._encoder_output_names = list(self._encoder.getOutputName())
        self._decoder_input_names = list(self._decoder.getInputName())
        self._decoder_output_names = list(self._decoder.getOutputName())
        self._decoder_input_shapes = dict(
            zip(
                self._decoder_input_names,
                self._decoder.getInputShapes(),
                strict=True,
            )
        )

        self._layers = int(self._config.decoder_layers)
        self._start_token = int(self._config.decoder_start_token_id)
        self._end_token = int(self._config.eos_token_id)

        self._self_cache_input_names = [
            f"{prefix}_cache_self_{layer}_in"
            for layer in range(self._layers)
            for prefix in ("k", "v")
        ]
        self._cross_cache_names = [
            f"{prefix}_cache_cross_{layer}"
            for layer in range(self._layers)
            for prefix in ("k", "v")
        ]
        self._self_cache_output_names = [
            f"{prefix}_cache_self_{layer}_out"
            for layer in range(self._layers)
            for prefix in ("k", "v")
        ]

        self._validate_model_contract()

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """Return Whisper's raw transcript without SLM formatting."""
        if self._closed:
            raise RuntimeError("The ASR engine has been closed.")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Whisper expects {SAMPLE_RATE} Hz audio, received {sample_rate}."
            )

        waveform = np.asarray(samples, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError("ASR expects one mono audio channel.")
        if waveform.size == 0:
            return ""

        waveform = np.nan_to_num(
            waveform,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        waveform = np.clip(waveform, -1.0, 1.0)

        with self._inference_lock:
            chunks = [
                waveform[start : start + MAX_AUDIO_SAMPLES]
                for start in range(0, len(waveform), MAX_AUDIO_SAMPLES)
            ]
            transcripts = [
                self._transcribe_chunk(chunk)
                for chunk in chunks
            ]

        return " ".join(text for text in transcripts if text).strip()

    def close(self) -> None:
        """Release Python references; QAIRT exits with the application."""
        self._closed = True
        del self._encoder
        del self._decoder

    def _transcribe_chunk(self, samples: np.ndarray) -> str:
        features = self._feature_extractor(
            samples,
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            max_length=MAX_AUDIO_SAMPLES,
            truncation=True,
            return_tensors="np",
        )["input_features"].astype(np.float32, copy=False)

        if features.shape != (1, 80, 3000):
            raise AsrModelError(
                f"Unexpected Whisper feature shape: {features.shape}."
            )

        self._perf_profile.SetPerfProfileGlobal(
            self._perf_profile.BURST
        )
        try:
            encoder_outputs = self._encoder.Inference(
                [np.ascontiguousarray(features)]
            )
            cross_cache = dict(
                zip(
                    self._encoder_output_names,
                    encoder_outputs,
                    strict=True,
                )
            )

            self_cache = {
                name: np.zeros(
                    tuple(self._decoder_input_shapes[name]),
                    dtype=np.float32,
                )
                for name in self._self_cache_input_names
            }
            attention_mask = np.full(
                (1, 1, 1, MAX_DECODE_TOKENS),
                MASK_NEGATIVE_VALUE,
                dtype=np.float32,
            )
            tokens = [self._start_token]

            for position in range(MAX_DECODE_TOKENS - 1):
                # This is Whisper's causal-attention mask layout.
                attention_mask[
                    :, :, :, MAX_DECODE_TOKENS - position - 1
                ] = 0.0

                inputs: dict[str, np.ndarray] = {
                    "input_ids": np.array(
                        [[tokens[-1]]],
                        dtype=np.int32,
                    ),
                    "attention_mask": attention_mask,
                    "position_ids": np.array(
                        [position],
                        dtype=np.int32,
                    ),
                    **self_cache,
                    **cross_cache,
                }

                decoder_outputs = self._decoder.Inference(
                    [
                        np.ascontiguousarray(inputs[name])
                        for name in self._decoder_input_names
                    ]
                )
                outputs_by_name = dict(
                    zip(
                        self._decoder_output_names,
                        decoder_outputs,
                        strict=True,
                    )
                )

                self_cache = {
                    output_name.replace("_out", "_in"): value
                    for output_name, value in outputs_by_name.items()
                    if output_name != "logits"
                }

                logits = np.asarray(outputs_by_name["logits"]).reshape(-1)
                next_token = int(np.argmax(logits))
                tokens.append(next_token)

                if next_token == self._end_token:
                    break
        finally:
            self._perf_profile.RelPerfProfileGlobal()

        # Skip Whisper control/language/timestamp tokens, but do not use an SLM
        # or alter words, fillers, punctuation, or sentence structure.
        return self._tokenizer.decode(
            tokens,
            skip_special_tokens=True,
        ).strip()

    def _validate_model_contract(self) -> None:
        expected_decoder_inputs = {
            "input_ids",
            "attention_mask",
            "position_ids",
            *self._self_cache_input_names,
            *self._cross_cache_names,
        }
        expected_decoder_outputs = {
            "logits",
            *self._self_cache_output_names,
        }

        if self._encoder_input_names != ["input_features"]:
            raise AsrModelError(
                f"Unexpected encoder inputs: {self._encoder_input_names}"
            )
        if set(self._encoder_output_names) != set(self._cross_cache_names):
            raise AsrModelError(
                "The encoder context does not match Whisper-Small-Quantized. "
                f"Found outputs: {self._encoder_output_names}"
            )
        if set(self._decoder_input_names) != expected_decoder_inputs:
            raise AsrModelError(
                "The decoder context does not match Whisper-Small-Quantized. "
                f"Found inputs: {self._decoder_input_names}"
            )
        if set(self._decoder_output_names) != expected_decoder_outputs:
            raise AsrModelError(
                "The decoder context does not match Whisper-Small-Quantized. "
                f"Found outputs: {self._decoder_output_names}"
            )

    def _require_files(self) -> None:
        for path in (
            self._encoder_path,
            self._decoder_path,
            self._hf_assets_dir / "config.json",
            self._hf_assets_dir / "preprocessor_config.json",
            self._hf_assets_dir / "tokenizer.json",
        ):
            if not path.is_file():
                raise AsrModelError(
                    f"Missing ASR deployment asset: {path}"
                )

    @staticmethod
    def _require_windows_arm64() -> None:
        machine = platform.machine().lower()
        if platform.system() != "Windows" or machine not in {
            "arm64",
            "aarch64",
        }:
            raise AsrModelError(
                "QAIRT HTP ASR requires Windows ARM64 on a Snapdragon PC."
            )