#!/usr/bin/env python3
"""Realtime local English->Japanese translator.

Pipeline:
  microphone/loopback -> WebRTC VAD -> faster-whisper -> Ollama or LM Studio
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import queue
import signal
import sys
import threading
import time
import wave
from io import BytesIO
from typing import Iterable, Literal

import numpy as np
import requests
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

SAMPLE_RATE = 16_000
CHANNELS = 1
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # int16 mono


@dataclasses.dataclass
class AudioSegment:
    pcm16: bytes
    started_at: float
    ended_at: float


class SpeechSegmenter:
    """WebRTC VAD-based segmenter for 16 kHz 16-bit mono PCM."""

    def __init__(
        self,
        aggressiveness: int = 2,
        padding_ms: int = 300,
        silence_ms: int = 600,
        min_speech_ms: int = 300,
        max_segment_s: float = 8.0,
    ) -> None:
        if FRAME_MS not in (10, 20, 30):
            raise ValueError("FRAME_MS must be 10, 20, or 30 for WebRTC VAD")
        self.vad = webrtcvad.Vad(aggressiveness)
        self.padding_frames = max(1, padding_ms // FRAME_MS)
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.max_segment_frames = max(1, int(max_segment_s * 1000 // FRAME_MS))
        self.ring = collections.deque(maxlen=self.padding_frames)
        self.triggered = False
        self.voiced: list[bytes] = []
        self.silence_count = 0
        self.started_at = 0.0

    def process_frame(self, frame: bytes) -> AudioSegment | None:
        now = time.time()
        is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

        if not self.triggered:
            self.ring.append(frame)
            speech_count = sum(self.vad.is_speech(f, SAMPLE_RATE) for f in self.ring)
            if speech_count > 0.75 * self.ring.maxlen:
                self.triggered = True
                self.started_at = now - (len(self.ring) * FRAME_MS / 1000.0)
                self.voiced.extend(self.ring)
                self.ring.clear()
            return None

        self.voiced.append(frame)
        self.silence_count = self.silence_count + 1 if not is_speech else 0

        too_long = len(self.voiced) >= self.max_segment_frames
        enough_silence = self.silence_count >= self.silence_frames
        if too_long or enough_silence:
            segment = self._finish(now)
            return segment
        return None

    def flush(self) -> AudioSegment | None:
        if not self.triggered or not self.voiced:
            return None
        return self._finish(time.time())

    def _finish(self, ended_at: float) -> AudioSegment | None:
        frames = self.voiced
        self.voiced = []
        self.triggered = False
        self.silence_count = 0
        self.ring.clear()
        if len(frames) < self.min_speech_frames:
            return None
        return AudioSegment(pcm16=b"".join(frames), started_at=self.started_at, ended_at=ended_at)


def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def list_devices() -> None:
    print(sd.query_devices())


def transcribe(model: WhisperModel, segment: AudioSegment, language: str) -> str:
    audio = pcm16_to_float32(segment.pcm16)
    segments, _info = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
        temperature=0.0,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text


def build_messages(text: str, target_language: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a professional real-time conference interpreter. "
                f"Translate the user's English transcript into natural {target_language}. "
                "Return only the translation. Do not add explanations, notes, labels, markdown, or quotes. "
                "If the input is incomplete, translate only what is clear and keep it concise."
            ),
        },
        {"role": "user", "content": text},
    ]


def translate_ollama(text: str, model: str, target_language: str, host: str) -> str:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": build_messages(text, target_language),
        "stream": False,
        "options": {"temperature": 0.0},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def translate_lmstudio(text: str, model: str, target_language: str, host: str) -> str:
    url = host.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": build_messages(text, target_language),
        "temperature": 0.0,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def should_skip_transcript(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    # Common Whisper silence artifacts in English meetings.
    artifacts = {
        "thank you.",
        "thanks for watching.",
        "you",
        "bye.",
        "uh",
        "um",
    }
    return normalized in artifacts


def worker_loop(args: argparse.Namespace, in_q: queue.Queue[AudioSegment], stop: threading.Event) -> None:
    print(f"[init] Loading Whisper model: {args.whisper_model}", flush=True)
    model = WhisperModel(args.whisper_model, device=args.whisper_device, compute_type=args.compute_type)
    print("[ready] Listening. Press Ctrl+C to stop.\n", flush=True)

    while not stop.is_set() or not in_q.empty():
        try:
            segment = in_q.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            stt_started = time.time()
            text = transcribe(model, segment, args.source_language)
            if should_skip_transcript(text):
                continue

            if args.backend == "ollama":
                translation = translate_ollama(text, args.llm_model, args.target_language, args.ollama_host)
            else:
                translation = translate_lmstudio(text, args.llm_model, args.target_language, args.lmstudio_host)

            latency = time.time() - segment.ended_at
            stt_latency = time.time() - stt_started
            print(f"EN: {text}", flush=True)
            print(f"JA: {translation}", flush=True)
            print(f"    latency={latency:.2f}s processing={stt_latency:.2f}s\n", flush=True)
        except Exception as exc:  # keep app alive during transient LLM/server errors
            print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def audio_loop(args: argparse.Namespace, out_q: queue.Queue[AudioSegment], stop: threading.Event) -> None:
    segmenter = SpeechSegmenter(
        aggressiveness=args.vad_aggressiveness,
        silence_ms=args.silence_ms,
        min_speech_ms=args.min_speech_ms,
        max_segment_s=args.max_segment_s,
    )

    def callback(indata: bytes, frames: int, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"[audio] {status}", file=sys.stderr, flush=True)
        # RawInputStream may deliver multiple frames per callback.
        for i in range(0, len(indata), FRAME_BYTES):
            frame = indata[i : i + FRAME_BYTES]
            if len(frame) != FRAME_BYTES:
                continue
            segment = segmenter.process_frame(frame)
            if segment is not None:
                try:
                    out_q.put_nowait(segment)
                except queue.Full:
                    print("[warn] segment queue full; dropping audio segment", file=sys.stderr, flush=True)

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * FRAME_MS / 1000),
        dtype="int16",
        channels=CHANNELS,
        device=args.device_index,
        callback=callback,
    ):
        while not stop.is_set():
            time.sleep(0.1)

    final_segment = segmenter.flush()
    if final_segment is not None:
        out_q.put(final_segment)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local realtime English->Japanese translator via Whisper + Ollama/LM Studio")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    p.add_argument("--device-index", type=int, default=None, help="sounddevice input device index")
    p.add_argument("--backend", choices=["ollama", "lmstudio"], default="ollama")
    p.add_argument("--llm-model", default="qwen2.5:7b", help="Ollama/LM Studio model name")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--lmstudio-host", default="http://localhost:1234/v1")
    p.add_argument("--whisper-model", default="base", help="tiny, base, small, medium, large-v3, distil-large-v3, etc.")
    p.add_argument("--whisper-device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--compute-type", default="int8", help="int8 for CPU; float16 for CUDA; auto also works")
    p.add_argument("--source-language", default="en", help="Whisper source language code")
    p.add_argument("--target-language", default="Japanese")
    p.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    p.add_argument("--silence-ms", type=int, default=700, help="Silence duration before finalizing a segment")
    p.add_argument("--min-speech-ms", type=int, default=400)
    p.add_argument("--max-segment-s", type=float, default=8.0)
    return p.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    if args.list_devices:
        list_devices()
        return 0

    stop = threading.Event()
    segments: queue.Queue[AudioSegment] = queue.Queue(maxsize=8)

    def handle_stop(signum=None, frame=None) -> None:  # noqa: ANN001
        stop.set()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    worker = threading.Thread(target=worker_loop, args=(args, segments, stop), daemon=True)
    worker.start()

    try:
        audio_loop(args, segments, stop)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        worker.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
