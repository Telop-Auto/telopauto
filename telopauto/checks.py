"""実行前の確認(GPU・Gemini APIキー・音声ファイル)。

問題があれば、何をすればよいかを日本語で返す。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Check:
    ok: bool
    message: str
    fatal: bool = False


GPU_HELP = (
    "上部メニュー「ランタイム」→「ランタイムのタイプを変更」で「T4 GPU」を選び、"
    "保存してから ⓪ のセルをもう一度実行してください。"
)

KEY_HELP = (
    "左側の🔑(シークレット)を開き、名前を GEMINI_API_KEY、値に APIキーを入れて登録し、"
    "「ノートブックからのアクセス」をオンにしてください。"
    "Gemini校正を使わない場合は、設定で「Gemini校正」をオフにすればこのままで大丈夫です。"
)


def check_gpu() -> Check:
    try:
        import torch
    except Exception:
        return Check(False, "PyTorch が見つかりません。", fatal=True)
    if not torch.cuda.is_available():
        return Check(False, "GPU が使えない状態です。CPU だと処理に非常に時間がかかります。\n" + GPU_HELP, fatal=True)
    name = torch.cuda.get_device_name(0)
    return Check(True, f"GPU:{name}")


def get_gemini_key(retries: int = 3) -> tuple:
    """Colab のシークレットから GEMINI_API_KEY を取り出す。戻り値は (キー or None, Check)。"""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key, Check(True, "Gemini APIキー:環境変数から読み込みました")
    try:
        from google.colab import userdata
    except Exception:
        return None, Check(False, "Gemini APIキーが見つかりません(Colab以外の環境です)。")

    for attempt in range(1, retries + 1):
        try:
            key = userdata.get("GEMINI_API_KEY")
            if key:
                return key.strip(), Check(True, "Gemini APIキー:登録を確認しました")
            return None, Check(False, "Gemini APIキーが空です。\n" + KEY_HELP)
        except Exception as e:
            name = type(e).__name__
            if name == "SecretNotFoundError":
                return None, Check(False, "Gemini APIキー(GEMINI_API_KEY)が登録されていません。\n" + KEY_HELP)
            if name == "NotebookAccessError":
                return None, Check(
                    False, "Gemini APIキーへの「ノートブックからのアクセス」がオフになっています。\n" + KEY_HELP
                )
            if attempt < retries:
                time.sleep(3)
                continue
            return None, Check(False, f"Gemini APIキーを読み込めませんでした({name})。少し待ってから再実行してください。")
    return None, Check(False, "Gemini APIキーを読み込めませんでした。")


def verify_gemini(key: str, model: str) -> Check:
    """APIキーとモデル名が使えるかを、実際に問い合わせて確認する(利用枠はほぼ消費しない)。"""
    try:
        from google import genai

        client = genai.Client(api_key=key)
        client.models.get(model=model)
        return Check(True, f"Gemini:{model} に接続できました")
    except Exception as e:
        msg = str(e)
        if "API_KEY_INVALID" in msg or "API key not valid" in msg:
            return Check(False, "Gemini APIキーが正しくありません。Google AI Studio でキーを確認し、登録し直してください。")
        if "404" in msg or "NOT_FOUND" in msg:
            return Check(False, f"Geminiのモデル「{model}」が見つかりません。設定のモデル名を確認してください。")
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return Check(False, "Gemini APIの利用上限に達しています。時間をおくか、Gemini校正をオフにしてください。")
        return Check(False, f"Geminiに接続できませんでした:{msg[:200]}")


def check_audio(path: Optional[str]) -> Check:
    from .audio import AUDIO_EXTENSIONS, probe

    if not path:
        return Check(False, "音声ファイルが選ばれていません。② のセルでアップロードしてください。", fatal=True)
    if not os.path.exists(path):
        return Check(False, f"ファイルが見つかりません:{path}", fatal=True)
    if not path.lower().endswith(AUDIO_EXTENSIONS):
        return Check(False, "音声・動画ファイルではないようです(対応形式:WAV / MP3 / M4A / MP4 など)。", fatal=True)
    info = probe(path)
    if not info["ok"] or not info["has_audio"]:
        return Check(False, "このファイルから音声を読み取れません。ファイルが壊れていないか確認してください。", fatal=True)
    minutes = info["duration"] / 60
    return Check(True, f"音声:{os.path.basename(path)}(約{minutes:.1f}分)")
