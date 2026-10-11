"""Colab の1セルから呼ぶ実行処理。

ノートブックのセルが「GPU確認 → キーの登録確認 → アップロード(裏でインストール)」
までを行い、その後このモジュールの run() を呼ぶ。

  from telopauto import app
  app.run(settings, audio_path, start_over=False)
"""

from __future__ import annotations

import base64
import html
import os
import traceback

from . import checks
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


def _gemini_key(settings: Settings):
    """使える APIキーを返す。使えなければ None。"""
    if _verified["key"] and _verified["model"] == settings.gemini_model:
        return _verified["key"]
    key, kc = checks.get_gemini_key()
    if not key:
        _say(False, kc.message)
        return None
    vc = checks.verify_gemini(key, settings.gemini_model)
    _say(vc.ok, vc.message)
    if not vc.ok:
        return None
    _verified.update(key=key, model=settings.gemini_model)
    return key


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


def run(settings: Settings, audio_path: str, start_over: bool = False) -> None:
    from .pipeline import Pipeline

    s = settings.normalized()

    ac = checks.check_audio(audio_path)
    _say(ac.ok, ac.message)
    if not ac.ok:
        _stop("別のファイルを選んで、もう一度 ▶ を押してください。")
        return
    _remember_file(audio_path)

    api_key = None
    if s.use_gemini:
        api_key = _gemini_key(s)
        if not api_key:
            _stop(
                "Gemini APIキーを確認するか「use_gemini」のチェックを外し、"
                "「前回のファイルで作り直す」を選んでもう一度 ▶ を押してください(アップロードは不要です)。"
            )
            return
    else:
        print("ℹ️ Gemini校正:オフ")

    print()
    print("=== SRT作成 ===")
    try:
        result = Pipeline(audio_path).run(s, api_key=api_key, force=start_over)
    except Exception as e:
        print()
        print(f"❌ エラーで止まりました:{type(e).__name__}: {e}")
        print("「前回のファイルで作り直す」を選んで ▶ を押すと、完了した工程は飛ばして続きから再開します。")
        print("繰り返し失敗する場合は、以下の内容を X の DM でお知らせください。")
        traceback.print_exc()
        return

    print()
    print("設定を変えて作り直す場合は、「前回のファイルで作り直す」を選んで ▶ を押してください。")
    print("変更が必要な工程だけやり直します(例:行数の変更なら数秒で完了)。")
    _show_result(result)
