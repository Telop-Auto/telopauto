"""Whisper による文字起こし。

標準は faster-whisper(CTranslate2)。モデルは OpenAI の Whisper と同じ重みで、
openai-whisper より高速。Colab に最初から入っている PyTorch / NumPy を
入れ替えないため、セッションの再起動が不要。

万一 faster-whisper が GPU で動かない場合は、Colab 標準の transformers で
同じ Whisper モデルを動かす方式に自動で切り替える。

戻り値は [{"start": 秒, "end": 秒, "text": 本文}] のリスト。
"""

from __future__ import annotations

import gc
import glob
import os
from typing import Callable, List

import numpy as np

from .audio import SAMPLE_RATE

Log = Callable[[str], None]


def _preload_cuda_libs() -> None:
    """pip で入っている cuDNN / cuBLAS を先に読み込む。

    CTranslate2 が「libcudnn_ops.so.9 が見つからない」で落ちるのを防ぐ。
    """
    import ctypes

    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            mod = __import__(pkg, fromlist=["_"])
        except Exception:
            continue
        for base in getattr(mod, "__path__", []):
            for so in sorted(glob.glob(os.path.join(base, "lib", "lib*.so*"))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _transcribe_faster_whisper(audio, model_size, device, log: Log) -> List[dict]:
    from faster_whisper import WhisperModel

    if device == "cuda":
        _preload_cuda_libs()
    compute_type = "float16" if device == "cuda" else "int8"
    log(f"Whisperモデル({model_size})を読み込んでいます...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    try:
        segments, info = model.transcribe(
            audio,
            language="ja",
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        out = []
        total = max(info.duration, 1e-6)
        last_report = 0.0
        for seg in segments:
            text = seg.text.strip()
            if text:
                out.append({"start": float(seg.start), "end": float(seg.end), "text": text})
            if seg.end - last_report >= 60 or seg.end >= total:
                last_report = seg.end
                log(f"  文字起こし中... {min(seg.end / total, 1.0) * 100:.0f}%")
        return out
    finally:
        del model
        _free_gpu()


def _transcribe_transformers(audio, model_size, device, log: Log) -> List[dict]:
    import torch
    from transformers import pipeline

    log(f"Whisperモデル({model_size}・transformers版)を読み込んでいます...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = dict(model=f"openai/whisper-{model_size}", device=0 if device == "cuda" else -1)
    try:
        asr = pipeline("automatic-speech-recognition", dtype=dtype, **kwargs)
    except TypeError:  # 古い transformers
        asr = pipeline("automatic-speech-recognition", torch_dtype=dtype, **kwargs)
    try:
        result = asr(
            {"raw": audio, "sampling_rate": SAMPLE_RATE},
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=True,
            generate_kwargs={"language": "ja", "task": "transcribe"},
        )
        out = []
        duration = len(audio) / SAMPLE_RATE
        for ch in result.get("chunks", []):
            start, end = ch["timestamp"]
            start = float(start or 0.0)
            end = float(end if end is not None else duration)
            text = ch["text"].strip()
            if text:
                out.append({"start": start, "end": max(end, start), "text": text})
        return out
    finally:
        del asr
        _free_gpu()


def transcribe(audio: np.ndarray, model_size: str, device: str, log: Log = print) -> List[dict]:
    try:
        segments = _transcribe_faster_whisper(audio, model_size, device, log)
        log(f"文字起こし完了(faster-whisper):{len(segments)}区間")
        return segments
    except Exception as e:  # GPUライブラリの不整合など
        log(f"⚠️ faster-whisper で処理できませんでした({type(e).__name__}: {e})")
        log("   transformers 版に切り替えて続行します。")
        _free_gpu()
    segments = _transcribe_transformers(audio, model_size, device, log)
    log(f"文字起こし完了(transformers):{len(segments)}区間")
    return segments
