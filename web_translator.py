#!/usr/bin/env python3
"""Browser-based GUI for local realtime speech translation via LM Studio."""
from __future__ import annotations

import argparse
import dataclasses
import json
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
import sounddevice as sd
from flask import Flask, Response, jsonify, request
from faster_whisper import WhisperModel

from realtime_local_translator import (
    CHANNELS,
    FRAME_BYTES,
    FRAME_MS,
    SAMPLE_RATE,
    AudioSegment,
    SpeechSegmenter,
    pcm16_to_float32,
    should_skip_transcript,
)


LANGUAGES = {
    "en-ja": {
        "source_code": "en",
        "source_name": "English",
        "source_label": "英語の文字起こし",
        "target_name": "Japanese",
        "target_label": "日本語訳",
    },
    "ja-en": {
        "source_code": "ja",
        "source_name": "Japanese",
        "source_label": "日本語の文字起こし",
        "target_name": "English",
        "target_label": "英訳",
    },
}

DEFAULT_LMSTUDIO_HOST = "http://127.0.0.1:1234"


def normalize_lmstudio_host(host: str) -> str:
    host = (host or DEFAULT_LMSTUDIO_HOST).strip().rstrip("/")
    return host if host.endswith("/v1") else host + "/v1"


@dataclasses.dataclass
class RuntimeConfig:
    input_device: int | None
    lmstudio_host: str
    llm_model: str
    direction: str
    whisper_model: str
    whisper_device: str
    compute_type: str
    vad_aggressiveness: int
    silence_ms: int
    min_speech_ms: int
    max_segment_s: float


class TranslatorRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.segment_queue: queue.Queue[AudioSegment] = queue.Queue(maxsize=12)
        self.audio_thread: threading.Thread | None = None
        self.worker_thread: threading.Thread | None = None
        self.config: RuntimeConfig | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=300)
        self.event_cond = threading.Condition()
        self.event_id = 0
        self.items: list[dict[str, Any]] = []
        self.status = "stopped"
        self.error: str | None = None
        self.audio_check_stop = threading.Event()
        self.audio_check_thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return bool(self.audio_thread and self.audio_thread.is_alive())

    def is_audio_check_running(self) -> bool:
        return bool(self.audio_check_thread and self.audio_check_thread.is_alive())

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.is_running(),
                "status": self.status,
                "error": self.error,
                "config": dataclasses.asdict(self.config) if self.config else None,
                "items": list(self.items),
                "audio_check_running": self.is_audio_check_running(),
            }

    def start(self, config: RuntimeConfig) -> None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("すでに実行中です")
            self.stop_event = threading.Event()
            self.segment_queue = queue.Queue(maxsize=12)
            self.config = config
            self.items = []
            self.error = None
            self.status = "starting"

        self._publish("status", {"status": "starting", "message": "Whisperモデルを読み込んでいます"})
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.worker_thread.start()
        self.audio_thread.start()

    def start_audio_check(self, input_device: int | None) -> None:
        with self.lock:
            if self.is_audio_check_running():
                raise RuntimeError("音声入力チェックはすでに実行中です")
            self.audio_check_stop = threading.Event()
            self.audio_check_thread = threading.Thread(
                target=self._audio_check_loop,
                args=(input_device,),
                daemon=True,
            )
            self.audio_check_thread.start()
        self._publish("audio_check_status", {"running": True, "message": "音声入力チェック中です"})

    def stop_audio_check(self) -> None:
        self.audio_check_stop.set()
        self._publish("audio_check_status", {"running": False, "message": "音声入力チェックを停止しました"})

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            self.status = "stopping"
        self._publish("status", {"status": "stopping", "message": "停止しています"})

    def save_transcript(self) -> Path:
        with self.lock:
            items = list(self.items)
            config = dataclasses.asdict(self.config) if self.config else {}
        if not items:
            raise RuntimeError("保存できる文字起こしがありません")

        out_dir = Path("transcripts")
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        lines = [
            "# Local LLM Realtime Translator Transcript",
            f"saved_at: {datetime.now().isoformat(timespec='seconds')}",
            f"config: {json.dumps(config, ensure_ascii=False)}",
            "",
        ]
        for item in items:
            lines.extend(
                [
                    f"[{item['time']}] {item['source_label']}",
                    item["source_text"],
                    f"{item['target_label']}",
                    item["translation"],
                    "",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def clear_transcript(self) -> None:
        with self.lock:
            self.items = []
        self._publish("transcript_cleared", {"ok": True})

    def iter_events(self):
        next_id = 0
        while True:
            with self.event_cond:
                self.event_cond.wait_for(lambda: self.event_id > next_id, timeout=15)
                pending = [event for event in self.events if event["id"] > next_id]
                if pending:
                    next_id = pending[-1]["id"]
            if not pending:
                yield ": keepalive\n\n"
                continue
            for event in pending:
                yield f"id: {event['id']}\nevent: {event['type']}\ndata: {json.dumps(event['payload'], ensure_ascii=False)}\n\n"

    def _publish(self, event_type: str, payload: dict[str, Any]) -> None:
        with self.event_cond:
            self.event_id += 1
            self.events.append({"id": self.event_id, "type": event_type, "payload": payload})
            self.event_cond.notify_all()

    def _set_error(self, message: str) -> None:
        with self.lock:
            self.error = message
            self.status = "error"
        self._publish("error", {"message": message})

    def _publish_audio_level(self, pcm16: bytes, mode: str) -> None:
        samples = np.frombuffer(pcm16, dtype=np.int16)
        if samples.size == 0:
            return
        values = samples.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(values * values)))
        peak = float(np.max(np.abs(values)))
        db = -80.0 if rms <= 0 else max(-80.0, 20.0 * float(np.log10(rms)))
        self._publish(
            "audio_level",
            {
                "mode": mode,
                "rms": round(rms, 4),
                "peak": round(peak, 4),
                "db": round(db, 1),
                "active": peak >= 0.015 or rms >= 0.006,
            },
        )

    def _audio_check_loop(self, input_device: int | None) -> None:
        last_level_at = 0.0

        def callback(indata: bytes, frames: int, time_info, status) -> None:  # noqa: ANN001
            nonlocal last_level_at
            if status:
                self._publish("audio_check_status", {"running": True, "message": str(status)})
            now = time.time()
            if now - last_level_at >= 0.1:
                last_level_at = now
                self._publish_audio_level(bytes(indata), "check")

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=int(SAMPLE_RATE * FRAME_MS / 1000),
                dtype="int16",
                channels=CHANNELS,
                device=input_device,
                callback=callback,
            ):
                while not self.audio_check_stop.is_set():
                    time.sleep(0.1)
        except Exception as exc:
            self._publish("audio_check_status", {"running": False, "message": f"音声入力チェックエラー: {exc}"})
        finally:
            self.audio_check_stop.set()
            self._publish_audio_level(bytes(FRAME_BYTES), "check")
            self._publish("audio_check_status", {"running": False, "message": "音声入力チェックを停止しました"})

    def _audio_loop(self) -> None:
        config = self.config
        if config is None:
            return
        segmenter = SpeechSegmenter(
            aggressiveness=config.vad_aggressiveness,
            silence_ms=config.silence_ms,
            min_speech_ms=config.min_speech_ms,
            max_segment_s=config.max_segment_s,
        )
        last_level_at = 0.0

        def callback(indata: bytes, frames: int, time_info, status) -> None:  # noqa: ANN001
            nonlocal last_level_at
            if status:
                self._publish("status", {"status": "audio", "message": str(status)})
            now = time.time()
            if now - last_level_at >= 0.1:
                last_level_at = now
                self._publish_audio_level(bytes(indata), "translate")
            for i in range(0, len(indata), FRAME_BYTES):
                frame = indata[i : i + FRAME_BYTES]
                if len(frame) != FRAME_BYTES:
                    continue
                segment = segmenter.process_frame(frame)
                if segment is None:
                    continue
                try:
                    self.segment_queue.put_nowait(segment)
                except queue.Full:
                    self._publish("status", {"status": "warning", "message": "音声キューが満杯のため一部を破棄しました"})

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=int(SAMPLE_RATE * FRAME_MS / 1000),
                dtype="int16",
                channels=CHANNELS,
                device=config.input_device,
                callback=callback,
            ):
                with self.lock:
                    self.status = "listening"
                self._publish("status", {"status": "listening", "message": "音声入力を待っています"})
                while not self.stop_event.is_set():
                    time.sleep(0.1)
        except Exception as exc:
            self._set_error(f"音声入力エラー: {type(exc).__name__}: {exc}")
            self.stop_event.set()
        finally:
            final_segment = segmenter.flush()
            if final_segment is not None:
                self.segment_queue.put(final_segment)
            with self.lock:
                if self.status != "error":
                    self.status = "stopped"
            self._publish("status", {"status": "stopped", "message": "停止しました"})

    def _worker_loop(self) -> None:
        config = self.config
        if config is None:
            return
        lang = LANGUAGES[config.direction]
        try:
            model = WhisperModel(config.whisper_model, device=config.whisper_device, compute_type=config.compute_type)
        except Exception as exc:
            self._set_error(f"Whisperモデルの読み込みに失敗しました: {type(exc).__name__}: {exc}")
            self.stop_event.set()
            return

        while not self.stop_event.is_set() or not self.segment_queue.empty():
            try:
                segment = self.segment_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                source_text = transcribe_segment(model, segment, lang["source_code"])
                if should_skip_transcript(source_text):
                    continue
                translation = translate_lmstudio(
                    source_text,
                    config.llm_model,
                    lang["source_name"],
                    lang["target_name"],
                    config.lmstudio_host,
                )
                item = {
                    "id": len(self.items) + 1,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "source_text": source_text,
                    "translation": translation,
                    "source_label": lang["source_label"],
                    "target_label": lang["target_label"],
                    "latency": round(time.time() - segment.ended_at, 2),
                }
                with self.lock:
                    self.items.append(item)
                self._publish("translation", item)
            except Exception as exc:
                self._publish("error", {"message": f"処理エラー: {type(exc).__name__}: {exc}"})


def transcribe_segment(model: WhisperModel, segment: AudioSegment, language: str) -> str:
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
    return " ".join(s.text.strip() for s in segments).strip()


def build_translation_messages(text: str, source_language: str, target_language: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a professional real-time conference interpreter. "
                f"Translate the user's {source_language} transcript into natural {target_language}. "
                "Return only the translation. Do not add explanations, notes, labels, markdown, or quotes. "
                "If the input is incomplete, translate only what is clear and keep it concise."
            ),
        },
        {"role": "user", "content": text},
    ]


def translate_lmstudio(text: str, model: str, source_language: str, target_language: str, host: str) -> str:
    url = normalize_lmstudio_host(host) + "/chat/completions"
    payload = {
        "model": model,
        "messages": build_translation_messages(text, source_language, target_language),
        "temperature": 0.0,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def create_app() -> Flask:
    app = Flask(__name__)
    runtime = TranslatorRuntime()

    @app.get("/")
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/devices")
    def devices():
        result = []
        for index, device in enumerate(sd.query_devices()):
            if int(device.get("max_input_channels", 0)) <= 0:
                continue
            result.append(
                {
                    "index": index,
                    "name": device["name"],
                    "channels": int(device["max_input_channels"]),
                    "hostapi": int(device["hostapi"]),
                }
            )
        return jsonify(result)

    @app.get("/api/models")
    def models():
        host = normalize_lmstudio_host(request.args.get("host", DEFAULT_LMSTUDIO_HOST))
        try:
            r = requests.get(host + "/models", timeout=10)
            r.raise_for_status()
            data = r.json().get("data", [])
            return jsonify([item.get("id") for item in data if item.get("id")])
        except Exception as exc:
            return jsonify({"error": f"LM Studioモデル一覧を取得できません: {exc}"}), 502

    @app.get("/api/status")
    def status():
        return jsonify(runtime.snapshot())

    @app.post("/api/start")
    def start():
        data = request.get_json(force=True)
        direction = data.get("direction", "en-ja")
        if direction not in LANGUAGES:
            return jsonify({"error": "翻訳方向が不正です"}), 400
        raw_device = data.get("input_device")
        config = RuntimeConfig(
            input_device=None if raw_device in (None, "") else int(raw_device),
            lmstudio_host=normalize_lmstudio_host(data.get("lmstudio_host") or DEFAULT_LMSTUDIO_HOST),
            llm_model=data.get("llm_model") or "",
            direction=direction,
            whisper_model=data.get("whisper_model") or "base",
            whisper_device=data.get("whisper_device") or "auto",
            compute_type=data.get("compute_type") or "int8",
            vad_aggressiveness=int(data.get("vad_aggressiveness", 2)),
            silence_ms=int(data.get("silence_ms", 700)),
            min_speech_ms=int(data.get("min_speech_ms", 400)),
            max_segment_s=float(data.get("max_segment_s", 8.0)),
        )
        if not config.llm_model:
            return jsonify({"error": "LM Studioのモデルを選択してください"}), 400
        try:
            runtime.start(config)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True})

    @app.post("/api/stop")
    def stop():
        runtime.stop()
        return jsonify({"ok": True})

    @app.post("/api/audio-check/start")
    def audio_check_start():
        data = request.get_json(force=True)
        raw_device = data.get("input_device")
        input_device = None if raw_device in (None, "") else int(raw_device)
        try:
            runtime.start_audio_check(input_device)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True})

    @app.post("/api/audio-check/stop")
    def audio_check_stop():
        runtime.stop_audio_check()
        return jsonify({"ok": True})

    @app.post("/api/save")
    def save():
        try:
            path = runtime.save_transcript()
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"path": str(path.resolve())})

    @app.post("/api/clear")
    def clear():
        runtime.clear_transcript()
        return jsonify({"ok": True})

    @app.get("/api/events")
    def events():
        return Response(runtime.iter_events(), mimetype="text/event-stream")

    return app


INDEX_HTML = r"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local LLM Realtime Translator</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #1b1f24;
      --muted: #667085;
      --line: #d8dde6;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --danger: #b42318;
      --shadow: 0 8px 24px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 28px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { font-size: 20px; margin: 0; font-weight: 650; }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      height: calc(100vh - 67px);
      min-height: 0;
      overflow: hidden;
    }
    aside, section.workspace {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    aside {
      padding: 18px;
      align-self: stretch;
      min-height: 0;
      overflow: auto;
    }
    .field { margin-bottom: 14px; }
    label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
    input, select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }
    button {
      border: 1px solid transparent;
      border-radius: 6px;
      min-height: 38px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
      background: #eef2f6;
      color: var(--text);
    }
    button.primary { background: var(--accent); color: #fff; }
    button.primary:hover { background: var(--accent-strong); }
    button.danger { color: var(--danger); border-color: #f1b8b2; background: #fff6f5; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #98a2b3;
    }
    .dot.running { background: #12b76a; }
    .dot.error { background: #f04438; }
    .workspace {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-height: 0;
      overflow: hidden;
    }
    .toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .columns {
      display: grid;
      grid-template-columns: 1fr 1fr;
      min-height: 0;
    }
    .transcript-scroll {
      min-height: 0;
      height: 100%;
      overflow: auto;
    }
    .transcript-header,
    .entry-pair {
      display: grid;
      grid-template-columns: 1fr 1fr;
    }
    .transcript-header {
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    .transcript-heading {
      margin: 0;
      padding: 16px;
      font-size: 15px;
      color: var(--muted);
      font-weight: 600;
    }
    .transcript-heading + .transcript-heading,
    .entry-cell + .entry-cell {
      border-left: 1px solid var(--line);
    }
    .entry-pair {
      border-bottom: 1px solid #edf0f5;
    }
    .entry-cell {
      min-width: 0;
      padding: 12px 16px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.55;
    }
    .pane {
      min-width: 0;
      padding: 16px;
      overflow: auto;
    }
    .pane + .pane { border-left: 1px solid var(--line); }
    .pane h2 {
      margin: 0 0 12px;
      font-size: 15px;
      color: var(--muted);
      font-weight: 600;
    }
    .entry {
      border-bottom: 1px solid #edf0f5;
      padding: 12px 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.55;
    }
    .entry time {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .entry-cell time {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .notice-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
    }
    .notice-cell {
      color: var(--muted);
      font-size: 14px;
      padding: 20px 16px;
    }
    .notice-cell + .notice-cell { border-left: 1px solid var(--line); }
    .notice {
      color: var(--muted);
      font-size: 14px;
      padding: 20px 0;
    }
    .message {
      min-height: 20px;
      color: var(--muted);
      font-size: 13px;
      margin-top: 12px;
      overflow-wrap: anywhere;
    }
    .message.error { color: var(--danger); }
    .audio-check {
      margin-top: 14px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
    }
    .meter-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      margin-top: 8px;
    }
    .meter {
      height: 14px;
      overflow: hidden;
      border-radius: 999px;
      background: #e6eaf0;
    }
    .meter-fill {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, #0f766e, #f79009);
      transition: width 90ms linear;
    }
    .level-text {
      min-width: 72px;
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }
    .audio-state {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .audio-state.active { color: var(--accent-strong); font-weight: 600; }
    @media (max-width: 860px) {
      main {
        grid-template-columns: 1fr;
        height: auto;
        min-height: calc(100vh - 67px);
        overflow: visible;
      }
      .workspace { min-height: 60vh; }
      .columns { grid-template-columns: 1fr; }
      .pane + .pane { border-left: 0; border-top: 1px solid var(--line); }
      .transcript-header,
      .entry-pair,
      .notice-row { grid-template-columns: 1fr; }
      .transcript-heading + .transcript-heading,
      .entry-cell + .entry-cell,
      .notice-cell + .notice-cell {
        border-left: 0;
        border-top: 1px solid var(--line);
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Local LLM Realtime Translator</h1>
    <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">停止中</span></div>
  </header>
  <main>
    <aside>
      <div class="field">
        <label for="direction">翻訳方向</label>
        <select id="direction">
          <option value="en-ja">英語から日本語</option>
          <option value="ja-en">日本語から英語</option>
        </select>
      </div>
      <div class="field">
        <label for="inputDevice">音声入力</label>
        <select id="inputDevice"></select>
      </div>
      <div class="field">
        <label for="lmstudioHost">LM Studio API URL</label>
        <input id="lmstudioHost" value="http://127.0.0.1:1234">
      </div>
      <div class="field">
        <label for="llmModel">LLMモデル</label>
        <select id="llmModel"></select>
      </div>
      <div class="actions">
        <button id="reloadModels">モデル更新</button>
        <button id="reloadDevices">入力更新</button>
      </div>
      <div class="audio-check">
        <label>音声入力チェック</label>
        <div class="actions">
          <button id="audioCheckStartBtn">チェック開始</button>
          <button id="audioCheckStopBtn" disabled>チェック停止</button>
        </div>
        <div class="meter-row">
          <div class="meter"><div id="audioMeterFill" class="meter-fill"></div></div>
          <div id="audioLevelText" class="level-text">-80.0 dB</div>
        </div>
        <div id="audioStateText" class="audio-state">未チェック</div>
      </div>
      <div class="grid2">
        <div class="field">
          <label for="whisperModel">Whisper</label>
          <select id="whisperModel">
            <option>tiny</option>
            <option selected>base</option>
            <option>small</option>
            <option>medium</option>
            <option>large-v3</option>
            <option>distil-large-v3</option>
          </select>
        </div>
        <div class="field">
          <label for="computeType">計算精度</label>
          <select id="computeType">
            <option selected>int8</option>
            <option>float16</option>
            <option>auto</option>
          </select>
        </div>
      </div>
      <div class="grid2">
        <div class="field">
          <label for="silenceMs">無音判定 ms</label>
          <input id="silenceMs" type="number" value="700" min="200" step="100">
        </div>
        <div class="field">
          <label for="maxSegmentS">最大秒数</label>
          <input id="maxSegmentS" type="number" value="8" min="2" step="1">
        </div>
      </div>
      <div class="actions">
        <button id="startBtn" class="primary">開始</button>
        <button id="stopBtn" class="danger" disabled>停止</button>
        <button id="saveBtn">文字起こし保存</button>
        <button id="clearBtn">表示クリア</button>
      </div>
      <div id="message" class="message"></div>
    </aside>
    <section class="workspace">
      <div class="toolbar">
        <strong id="sessionTitle">翻訳・文字起こし</strong>
        <span id="countText">0件</span>
      </div>
      <div id="transcriptScroll" class="transcript-scroll">
        <div class="transcript-header">
          <h2 id="sourceHeading" class="transcript-heading">英語の文字起こし</h2>
          <h2 id="targetHeading" class="transcript-heading">日本語訳</h2>
        </div>
        <div id="transcriptRows">
          <div class="notice-row">
            <div class="notice-cell">開始すると文字起こしがここに表示されます。</div>
            <div class="notice-cell">翻訳結果がここに表示されます。</div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const state = { count: 0, running: false, audioCheckRunning: false };
    const $ = (id) => document.getElementById(id);

    const labels = {
      "en-ja": ["英語の文字起こし", "日本語訳"],
      "ja-en": ["日本語の文字起こし", "英訳"]
    };

    function setMessage(text, isError = false) {
      $("message").textContent = text || "";
      $("message").className = "message" + (isError ? " error" : "");
    }

    function setRunning(running, statusText = "") {
      state.running = running;
      $("startBtn").disabled = running;
      $("stopBtn").disabled = !running;
      $("statusDot").className = "dot" + (running ? " running" : "");
      $("statusText").textContent = statusText || (running ? "実行中" : "停止中");
    }

    function setAudioCheckRunning(running) {
      state.audioCheckRunning = running;
      $("audioCheckStartBtn").disabled = running;
      $("audioCheckStopBtn").disabled = !running;
    }

    function updateAudioLevel(level) {
      const rmsPercent = Math.min(100, Math.max(0, level.rms * 650));
      const peakPercent = Math.min(100, Math.max(rmsPercent, level.peak * 100));
      $("audioMeterFill").style.width = `${peakPercent}%`;
      $("audioLevelText").textContent = `${level.db.toFixed(1)} dB`;
      $("audioStateText").textContent = level.active ? "音声が入っています" : "ほぼ無音です";
      $("audioStateText").className = "audio-state" + (level.active ? " active" : "");
    }

    function updateHeadings() {
      const [source, target] = labels[$("direction").value];
      $("sourceHeading").textContent = source;
      $("targetHeading").textContent = target;
    }

    function clearLists() {
      state.count = 0;
      $("countText").textContent = "0件";
      $("transcriptRows").innerHTML = `
        <div class="notice-row">
          <div class="notice-cell">開始すると文字起こしがここに表示されます。</div>
          <div class="notice-cell">翻訳結果がここに表示されます。</div>
        </div>`;
      $("transcriptScroll").scrollTop = 0;
    }

    function addEntry(item) {
      if (state.count === 0) {
        $("transcriptRows").innerHTML = "";
      }
      state.count += 1;
      $("countText").textContent = `${state.count}件`;
      const row = document.createElement("div");
      row.className = "entry-pair";
      row.innerHTML = `
        <div class="entry-cell"><time>${item.time}</time>${escapeHtml(item.source_text)}</div>
        <div class="entry-cell"><time>${item.time} / ${item.latency}s</time>${escapeHtml(item.translation)}</div>`;
      $("transcriptRows").appendChild(row);
      $("transcriptScroll").scrollTop = $("transcriptScroll").scrollHeight;
    }

    function escapeHtml(text) {
      return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    }

    async function loadDevices() {
      const r = await fetch("/api/devices");
      const devices = await r.json();
      $("inputDevice").innerHTML = '<option value="">OS既定入力</option>' + devices.map(
        (d) => `<option value="${d.index}">#${d.index} ${escapeHtml(d.name)} (${d.channels}ch)</option>`
      ).join("");
    }

    async function loadModels() {
      const host = encodeURIComponent($("lmstudioHost").value);
      const r = await fetch(`/api/models?host=${host}`);
      const data = await r.json();
      if (!r.ok) {
        setMessage(data.error || "モデル一覧を取得できません", true);
        return;
      }
      $("llmModel").innerHTML = data.map((id) => `<option>${escapeHtml(id)}</option>`).join("");
      setMessage(data.length ? "LM Studioモデル一覧を更新しました" : "LM Studioでモデルを読み込んでください");
    }

    async function start() {
      clearLists();
      updateHeadings();
      const payload = {
        direction: $("direction").value,
        input_device: $("inputDevice").value,
        lmstudio_host: $("lmstudioHost").value,
        llm_model: $("llmModel").value,
        whisper_model: $("whisperModel").value,
        whisper_device: "auto",
        compute_type: $("computeType").value,
        vad_aggressiveness: 2,
        silence_ms: $("silenceMs").value,
        min_speech_ms: 400,
        max_segment_s: $("maxSegmentS").value
      };
      const r = await fetch("/api/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await r.json();
      if (!r.ok) {
        setMessage(data.error || "開始できません", true);
        return;
      }
      setRunning(true, "起動中");
      setMessage("音声入力を開始しました");
    }

    async function stop() {
      await fetch("/api/stop", { method: "POST" });
      setRunning(false, "停止中");
    }

    async function startAudioCheck() {
      const payload = { input_device: $("inputDevice").value };
      const r = await fetch("/api/audio-check/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await r.json();
      if (!r.ok) {
        setMessage(data.error || "音声入力チェックを開始できません", true);
        return;
      }
      setAudioCheckRunning(true);
      setMessage("音声入力チェックを開始しました");
    }

    async function stopAudioCheck() {
      await fetch("/api/audio-check/stop", { method: "POST" });
      setAudioCheckRunning(false);
    }

    async function saveTranscript() {
      const r = await fetch("/api/save", { method: "POST" });
      const data = await r.json();
      if (!r.ok) {
        setMessage(data.error || "保存できません", true);
        return;
      }
      setMessage(`保存しました: ${data.path}`);
    }

    async function clearTranscript() {
      if (state.count === 0) {
        setMessage("クリアする文字起こしはありません");
        return;
      }
      if (!window.confirm("表示中の文字起こしと翻訳をクリアします。保存していない内容は画面と保存対象から消えます。よろしいですか？")) {
        return;
      }
      const r = await fetch("/api/clear", { method: "POST" });
      const data = await r.json();
      if (!r.ok) {
        setMessage(data.error || "クリアできません", true);
        return;
      }
      clearLists();
      setMessage("文字起こしと翻訳をクリアしました");
    }

    const events = new EventSource("/api/events");
    events.addEventListener("translation", (event) => addEntry(JSON.parse(event.data)));
    events.addEventListener("transcript_cleared", () => clearLists());
    events.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      if (data.status === "listening") setRunning(true, "待機中");
      if (data.status === "stopped") setRunning(false, "停止中");
      setMessage(data.message || "");
    });
    events.addEventListener("audio_level", (event) => updateAudioLevel(JSON.parse(event.data)));
    events.addEventListener("audio_check_status", (event) => {
      const data = JSON.parse(event.data);
      setAudioCheckRunning(Boolean(data.running));
      if (data.message) setMessage(data.message);
    });
    events.addEventListener("error", (event) => {
      const data = JSON.parse(event.data);
      $("statusDot").className = "dot error";
      $("statusText").textContent = "エラー";
      setMessage(data.message, true);
    });

    $("direction").addEventListener("change", updateHeadings);
    $("reloadDevices").addEventListener("click", loadDevices);
    $("reloadModels").addEventListener("click", loadModels);
    $("startBtn").addEventListener("click", start);
    $("stopBtn").addEventListener("click", stop);
    $("audioCheckStartBtn").addEventListener("click", startAudioCheck);
    $("audioCheckStopBtn").addEventListener("click", stopAudioCheck);
    $("saveBtn").addEventListener("click", saveTranscript);
    $("clearBtn").addEventListener("click", clearTranscript);

    updateHeadings();
    loadDevices().catch((e) => setMessage(String(e), true));
    loadModels().catch((e) => setMessage(String(e), true));
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web GUI for LM Studio realtime speech translation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
