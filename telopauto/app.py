"""Colab の1セルから呼ぶ実行処理。

  from telopauto import app
  app.run(settings, use_previous_file=False, start_over=False)

流れ:
  1. 事前チェック(GPU・Gemini APIキー)…問題があればアップロード前に止める
  2. 音声ファイルのアップロード(「前回のファイルを使う」ならスキップ)
  3. SRT 作成(必要な工程だけ実行)
  4. ダウンロード
"""

from __future__ import annotations

import base64
import html
import os
import traceback
from typing import Optional

from . import checks
from .audio import AUDIO_EXTENSIONS
from .settings import Settings

WORK_ROOT = "/content" if os.path.isdir("/content") else os.getcwd()
LAST_FILE_PATH = os.path.join(WORK_ROOT, ".telopauto_last_file")

# 同じセッション内で確認済みの APIキー(毎回の接続確認を省く)
_verified = {"key": None, "model": None}


def _say(ok: bool, message: str) -> None:
    print(("✅ " if ok else "⚠️ ") + message)


def _stop(message: str) -> None:
    print()
    print("━" * 40)
    print("⛔ " + message)
    print("━" * 40)


def _remember_file(path: str) -> None:
    try:
        with open(LAST_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(path)
    except Exception:
        pass


def _last_file() -> Optional[str]:
    try:
        with open(LAST_FILE_PATH, encoding="utf-8") as f:
            path = f.read().strip()
        return path if os.path.exists(path) else None
    except Exception:
        return None


def _gemini_key(settings: Settings) -> tuple:
    """戻り値:(使えるキー or None, 止めるべきか)。"""
    if _verified["key"] and _verified["model"] == settings.gemini_model:
        return _verified["key"], False
    key, kc = checks.get_gemini_key()
    if not key:
        _say(False, kc.message)
        return None, True
    vc = checks.verify_gemini(key, settings.gemini_model)
    _say(vc.ok, vc.message)
    if not vc.ok:
        return None, True
    _verified.update(key=key, model=settings.gemini_model)
    return key, False


def _upload() -> Optional[str]:
    from google.colab import files

    print()
    print("📂 音声ファイルを選んでください(MP3 / WAV / M4A / MP4 など)")
    uploaded = files.upload()
    names = [n for n in uploaded if n.lower().endswith(AUDIO_EXTENSIONS)]
    if not names:
        return None
    if len(names) > 1:
        print(f"※ 複数のファイルが選ばれたため、最初の「{names[0]}」を処理します。")
    return os.path.abspath(names[0])


def _show_result(result) -> None:
    from IPython.display import HTML, display

    b64 = base64.b64encode(result.srt_text.encode("utf-8")).decode("ascii")
    name = html.escape(os.path.basename(result.srt_path))
    display(
        HTML(
            f'<a download="{name}" href="data:application/x-subrip;charset=utf-8;base64,{b64}" '
            'style="display:inline-block;margin:8px 0;padding:8px 18px;background:#1a73e8;color:#fff;'
            'border-radius:6px;text-decoration:none;font-weight:bold;">⬇ SRTをダウンロード</a>'
            f'<span style="margin-left:12px;color:#888;">{name}(左側のファイル一覧にも保存されています)</span>'
            '<details style="margin-top:6px"><summary>プレビュー</summary>'
            '<pre style="max-height:360px;overflow:auto;background:#f7f7f7;color:#222;padding:8px">'
            f"{html.escape(result.srt_text)}</pre></details>"
        )
    )
    try:
        from google.colab import files

        files.download(result.srt_path)
    except Exception:
        pass


def run(settings: Settings, use_previous_file: bool = False, start_over: bool = False) -> None:
    from .pipeline import Pipeline

    s = settings.normalized()
    print("=== 事前チェック ===")

    # 1. GPU
    gpu = checks.check_gpu()
    _say(gpu.ok, gpu.message)
    if not gpu.ok:
        _stop("GPUの設定を直してから、もう一度 ▶ を押してください(アップロードはまだ不要です)。")
        return

    # 2. Gemini
    api_key = None
    if s.use_gemini:
        api_key, stop = _gemini_key(s)
        if stop:
            _stop("Gemini APIキーを確認するか、「use_gemini」のチェックを外して、もう一度 ▶ を押してください。")
            return
    else:
        print("ℹ️ Gemini校正:オフ")

    # 3. 音声ファイル
    path = None
    if use_previous_file:
        path = _last_file()
        if path:
            print(f"ℹ️ 前回のファイルを使います:{os.path.basename(path)}")
        else:
            print("ℹ️ 前回のファイルが見つからないため、アップロードしてください。")
    if not path:
        path = _upload()
        if not path:
            _stop("音声ファイルが選ばれませんでした。もう一度 ▶ を押してください。")
            return
    ac = checks.check_audio(path)
    _say(ac.ok, ac.message)
    if not ac.ok:
        _stop("別のファイルを選んで、もう一度 ▶ を押してください。")
        return
    _remember_file(path)

    # 4. SRT 作成
    print()
    print("=== SRT作成 ===")
    try:
        result = Pipeline(path).run(s, api_key=api_key, force=start_over)
    except Exception as e:
        print()
        print(f"❌ エラーで止まりました:{type(e).__name__}: {e}")
        print("「use_previous_file」にチェックを入れて ▶ を押すと、完了した工程は飛ばして続きから再開します。")
        print("繰り返し失敗する場合は、以下の内容を X の DM でお知らせください。")
        traceback.print_exc()
        return

    print()
    print("設定を変えて作り直す場合は、「use_previous_file」にチェックを入れて ▶ を押してください。")
    print("必要な工程だけやり直します(例:行数の変更なら数秒で完了)。")
    _show_result(result)
