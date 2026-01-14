#!/usr/bin/env python3
"""
Speech-to-text CLI.

Reads from a PipeWire device, audio file, or stdin and prints the raw,
unprocessed transcription to stdout.
"""

import argparse
import os
import subprocess
import sys
import time
import warnings
from functools import cached_property
from typing import Optional

import numpy as np

os.environ["ALSA_LOG_LEVEL"] = "0"


import librosa
import pyaudio
import torch
from pysilero_vad import SileroVoiceActivityDetector
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class VoiceTranscriber:
    def __init__(
        self,
        source: str,
        model_id: str,
        language: Optional[str] = None,
        timeout: float = 2.0,
    ) -> None:
        self.source: str = source
        self.model_id: str = model_id
        self.language: Optional[str] = language
        self.timeout: float = timeout
        self.sample_rate: int = 16000

        self.playing_players: list[str] = []

        self.torch_device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    @cached_property
    def model(self) -> AutoModelForSpeechSeq2Seq:
        m = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        m.to(self.torch_device)
        if hasattr(m, "config"):
            m.config.use_cache = True
        return m

    @cached_property
    def processor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.model_id)

    @cached_property
    def pipe(self) -> pipeline:
        return pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=self.torch_dtype,
            device=self.torch_device,
            generate_kwargs={"language": self.language} if self.language else {},
        )

    @cached_property
    def vad(self) -> SileroVoiceActivityDetector:
        return SileroVoiceActivityDetector()

    @cached_property
    def chunk_bytes(self) -> int:
        return self.vad.chunk_bytes()

    @cached_property
    def chunk_samples(self) -> int:
        return self.vad.chunk_samples()

    @staticmethod
    def get_default_input_device() -> Optional[int]:
        p = pyaudio.PyAudio()

        try:
            default_device = p.get_default_input_device_info()
            if int(default_device.get("maxInputChannels", 0)) > 0:
                device_index = default_device["index"]
                return device_index
        except Exception:
            pass

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                return i

        p.terminate()
        return None

    def pause_media_players(self) -> None:
        try:
            result = subprocess.run(["playerctl", "-l"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                players = result.stdout.strip().split("\n")
                for player in players:
                    if player.strip():
                        status_result = subprocess.run(
                            ["playerctl", "-p", player, "status"],
                            capture_output=True,
                            text=True,
                            timeout=2,
                        )
                        if status_result.returncode == 0 and "Playing" in status_result.stdout:
                            self.playing_players.append(player)

                if self.playing_players:
                    subprocess.run(["playerctl", "-a", "pause"], timeout=5)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def resume_media_players(self) -> None:
        if self.playing_players:
            for player in self.playing_players:
                try:
                    subprocess.run(["playerctl", "-p", player, "play"], timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            self.playing_players = []

    def play_sound(self, sound_file: str) -> None:
        sounds_dir = os.environ.get("SOUNDS_DIR", "sounds")
        try:
            sound_path = os.path.join(sounds_dir, sound_file)
            subprocess.run(["pw-play", sound_path], timeout=3)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    def process_text(self, text: str) -> None:
        if text:
            print(text.strip())
            sys.stdout.flush()

    def transcribe_file(self, file_path: str) -> None:
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=self.sample_rate, mono=True)

            result = self.pipe(audio_data, return_timestamps=False)
            text = result["text"]
            self.process_text(text)

        except Exception as e:
            print(f"Error transcribing file: {e}")
            sys.exit(1)

    def transcribe_stdin(self) -> None:
        try:
            audio_data = sys.stdin.buffer.read()
            if not audio_data:
                return

            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            result = self.pipe(audio_array, return_timestamps=False)
            text = result["text"]
            self.process_text(text)

        except Exception as e:
            print(f"Error transcribing stdin: {e}")
            sys.exit(1)

    def run(self) -> None:
        if self.source == "-":
            self.transcribe_stdin()
            return
        elif os.path.isfile(self.source):
            self.transcribe_file(self.source)
            return

        device = self.source

        self.pause_media_players()
        self.play_sound("start.wav")

        cmd = [
            "pw-record",
            "--target",
            device,
            "--format=s16",
            "--rate=16000",
            "--channels=1",
            "-",
        ]

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("Error: pw-record not found. Install PipeWire.")
            sys.exit(1)

        buffer = bytearray()
        last_voice_time = time.time()

        while True:
            chunk = proc.stdout.read(self.chunk_bytes)
            if not chunk:
                break

            speech_prob = self.vad(chunk)

            if speech_prob >= 0.5:
                last_voice_time = time.time()
                buffer.extend(chunk)
            elif time.time() - last_voice_time > self.timeout:
                break
            else:
                buffer.extend(chunk)

        proc.terminate()
        proc.wait()

        self.play_sound("stop.wav")

        self.resume_media_players()

        if buffer:
            audio_array = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0

            result = self.pipe(audio_array, return_timestamps=False)
            text = result["text"]

            if text:
                print(text)
                sys.stdout.flush()


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Speech to Text")
    parser.add_argument(
        "source",
        nargs="?",
        help="Audio file to transcribe, PipeWire device target, or '-' for stdin (default: system default)",
    )
    parser.add_argument(
        "--model",
        default="distil-whisper/distil-medium.en",
        help="Hugging Face model ID (default: distil-whisper/distil-medium.en)",
    )
    parser.add_argument(
        "--language",
        help="Language code (e.g., 'en', 'es'). Auto-detect if not specified.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Silence timeout in seconds (default: 2.0)",
    )

    args = parser.parse_args()

    if args.source == "-":
        source = "-"
    elif args.source and os.path.isfile(args.source):
        source = args.source
    elif args.source:
        source = args.source
    elif not sys.stdin.isatty():
        source = "-"
    else:
        default_device_index = VoiceTranscriber.get_default_input_device()
        if default_device_index is None:
            sys.exit(1)
        source = str(default_device_index)

    transcriber = VoiceTranscriber(
        source=source,
        model_id=args.model,
        language=args.language,
        timeout=args.timeout,
    )

    transcriber.run()


if __name__ == "__main__":
    main()
