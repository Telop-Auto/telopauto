"""音声の読み込み(ffmpeg で 16kHz モノラルへ変換)。"""

from __future__ import annotations

import json
import subprocess

import numpy as np

SAMPLE_RATE = 16000

AUDIO_EXTENSIONS = (
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi",
)


def load_audio(path: str) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg が見つかりません。") from e
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", "ignore")[-500:]
        raise RuntimeError(f"音声ファイルを読み込めませんでした。\n{msg}") from e
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def probe(path: str) -> dict:
    """ファイルの長さと音声トラックの有無を調べる。"""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
           "-of", "json", path]
    try:
        info = json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout)
    except Exception:
        return {"ok": False, "duration": 0.0, "has_audio": False}
    streams = info.get("streams", [])
    duration = float(info.get("format", {}).get("duration", 0) or 0)
    return {
        "ok": True,
        "duration": duration,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }
