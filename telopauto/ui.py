"""Colab 用の操作画面。

  from telopauto import ui
  ui.setup()   # ⓪ 準備:GPU・Gemini APIキーの確認
  ui.panel()   # ③ 設定して「SRT作成」
"""

from __future__ import annotations

import base64
import html
import os
import traceback

from . import checks
from .audio import AUDIO_EXTENSIONS
from .settings import Settings

UPLOAD_DIR = "/content" if os.path.isdir("/content") else os.getcwd()
SETTINGS_PATH = os.path.join(UPLOAD_DIR, "telopauto_settings.json")

_state = {"api_key": None}


def _mark(ok: bool) -> str:
    return "✅" if ok else "⚠️"


def setup(gemini_model: str = Settings.gemini_model) -> None:
    """準備セルから呼ぶ。環境を確認して結果を表示する。"""
    print("=== 環境チェック ===")
    gpu = checks.check_gpu()
    print(_mark(gpu.ok), gpu.message)

    key, kc = checks.get_gemini_key()
    print(_mark(kc.ok), kc.message)
    if key:
        vc = checks.verify_gemini(key, gemini_model)
        print(_mark(vc.ok), vc.message)
        if vc.ok:
            _state["api_key"] = key
        elif "モデル" in vc.message:
            # モデル名だけの問題なら、キー自体は使える可能性がある
            _state["api_key"] = key

    print()
    if gpu.ok:
        print("準備ができました。次のセルで音声ファイルをアップロードしてください。")
    else:
        print("⚠️ GPUの設定を直してから、このセルをもう一度実行してください。")


def find_audio_files(directory: str = UPLOAD_DIR) -> list:
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return []
    files = [
        os.path.join(directory, n)
        for n in names
        if n.lower().endswith(AUDIO_EXTENSIONS) and not n.startswith(".")
    ]
    return sorted(files, key=os.path.getmtime, reverse=True)


def _load_settings() -> Settings:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return Settings.from_json(f.read())
    except Exception:
        return Settings()


def _save_settings(s: Settings) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            f.write(s.to_json())
    except Exception:
        pass


def _download_link(path: str, text: str) -> str:
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    name = html.escape(os.path.basename(path))
    return (
        f'<a download="{name}" href="data:application/x-subrip;charset=utf-8;base64,{b64}" '
        'style="display:inline-block;padding:8px 18px;background:#1a73e8;color:#fff;'
        'border-radius:6px;text-decoration:none;font-weight:bold;">⬇ SRTをダウンロード</a>'
        f'<span style="margin-left:12px;color:#666;">{name}</span>'
    )


def panel() -> None:
    import ipywidgets as w
    from IPython.display import HTML, display

    s = _load_settings()
    style = {"description_width": "150px"}
    full = w.Layout(width="95%")

    # ---- ファイル ----
    file_dd = w.Dropdown(description="音声ファイル", style=style, layout=w.Layout(width="75%"))
    refresh = w.Button(description="一覧を更新", icon="refresh", layout=w.Layout(width="120px"))

    def refresh_files(_=None):
        files = find_audio_files()
        current = file_dd.value
        file_dd.options = [(os.path.basename(p), p) for p in files] or [("(ファイルがありません)", None)]
        file_dd.value = current if current in files else (files[0] if files else None)

    refresh.on_click(refresh_files)
    refresh_files()

    # ---- 字幕の形 ----
    max_chars = w.BoundedIntText(value=s.max_chars_per_line, min=4, max=60, description="1行の文字数", style=style)
    lines = w.ToggleButtons(options=[("1行", 1), ("2行", 2)], value=s.lines, description="字幕の行数", style=style)
    block_max = w.BoundedIntText(
        value=s.block_max_chars, min=4, max=120, description="1枚の最大文字数", style=style
    )
    block_note = w.HTML()

    def update_note(_=None):
        limit = max_chars.value * lines.value
        msg = f"1枚(1ブロック)に入れる最大文字数です。上限は {limit} 文字(1行の文字数×行数)。"
        if block_max.value > limit:
            msg += f" <b>※今の設定では {limit} 文字として扱います。</b>"
        block_note.value = f'<span style="color:#666;font-size:12px;margin-left:155px">{msg}</span>'

    for wd in (max_chars, lines, block_max):
        wd.observe(update_note, names="value")
    update_note()

    filler = w.ToggleButtons(options=["弱", "中", "強"], value=s.filler_level, description="フィラー削除", style=style)

    # ---- Gemini ----
    use_gemini = w.Checkbox(value=s.use_gemini, description="Gemini校正を使う(誤字・誤変換の修正)", indent=False)

    # ---- 詳細設定 ----
    whisper = w.Dropdown(
        options=[("large-v3(高精度・標準)", "large-v3"), ("medium(高速・長尺向け)", "medium")],
        value=s.whisper_model,
        description="Whisperモデル",
        style=style,
    )
    custom_fillers = w.Text(value=s.custom_fillers, description="追加のフィラー", style=style, layout=full,
                            placeholder="カンマ区切り 例:あのー,えっと")
    dictionary = w.Textarea(value=s.user_dictionary, description="ユーザー辞書", style=style,
                            layout=w.Layout(width="95%", height="90px"),
                            placeholder="1行に1つ「置換前=置換後」 例:アイフォン=iPhone")
    assist = w.Text(value=s.gemini_assist_words, description="Gemini補助辞書", style=style, layout=full,
                    placeholder="固有名詞をカンマ区切り 例:iPhone,Premiere")
    gemini_model = w.Text(value=s.gemini_model, description="Geminiモデル名", style=style)
    help_html = w.HTML(
        '<div style="color:#666;font-size:12px;line-height:1.6">'
        "・ユーザー辞書・Gemini補助辞書に登録した語は、途中で改行・分割されません。<br>"
        "・設定を変えて「SRT作成」を押し直すと、変更が必要な工程だけやり直します"
        "(例:行数の変更なら文字起こしやGemini校正はやり直しません)。</div>"
    )
    advanced = w.Accordion(
        children=[w.VBox([whisper, custom_fillers, dictionary, assist, gemini_model, help_html])]
    )
    advanced.set_title(0, "詳細設定(辞書・モデル)")
    advanced.selected_index = None

    force = w.Checkbox(value=False, description="最初からやり直す(文字起こしからすべて)", indent=False)
    run_btn = w.Button(description="SRT作成", button_style="primary", icon="play",
                       layout=w.Layout(width="200px", height="40px"))
    log_out = w.Output(layout=w.Layout(max_height="320px", overflow="auto", border="1px solid #ddd", padding="6px"))
    result_out = w.Output()

    def collect() -> Settings:
        return Settings(
            max_chars_per_line=max_chars.value,
            lines=lines.value,
            block_max_chars=block_max.value,
            whisper_model=whisper.value,
            filler_level=filler.value,
            custom_fillers=custom_fillers.value,
            user_dictionary=dictionary.value,
            use_gemini=use_gemini.value,
            gemini_model=gemini_model.value.strip() or Settings.gemini_model,
            gemini_assist_words=assist.value,
        )

    def on_run(_):
        run_btn.disabled = True
        run_btn.description = "処理中..."
        log_out.clear_output()
        result_out.clear_output()
        try:
            with log_out:
                _run(collect(), file_dd.value, force.value, result_out)
        finally:
            force.value = False
            run_btn.disabled = False
            run_btn.description = "SRT作成"

    run_btn.on_click(on_run)

    def section(title):
        return w.HTML(f'<div style="font-weight:bold;margin:10px 0 4px">{title}</div>')

    ui = w.VBox(
        [
            section("① 音声ファイル"),
            w.HBox([file_dd, refresh]),
            section("② 字幕の設定"),
            max_chars,
            lines,
            block_max,
            block_note,
            filler,
            use_gemini,
            advanced,
            section("③ 作成"),
            w.HBox([run_btn, force]),
            log_out,
            result_out,
        ]
    )
    display(ui)


def _run(s: Settings, path, force: bool, result_out) -> None:
    from IPython.display import HTML, display

    from .pipeline import Pipeline

    _save_settings(s)

    # ---- 事前チェック ----
    ac = checks.check_audio(path)
    print(_mark(ac.ok), ac.message)
    if not ac.ok:
        return
    gpu = checks.check_gpu()
    print(_mark(gpu.ok), gpu.message)
    if not gpu.ok:
        return

    api_key = None
    if s.use_gemini:
        api_key = _state.get("api_key")
        if not api_key:
            key, kc = checks.get_gemini_key()
            if key:
                vc = checks.verify_gemini(key, s.gemini_model)
                if not vc.ok:
                    print(_mark(False), vc.message)
                    return
                api_key = _state["api_key"] = key
            else:
                print(_mark(False), kc.message)
                return
        print("✅ Gemini校正:オン")
    else:
        print("ℹ️ Gemini校正:オフ")
    print()

    try:
        result = Pipeline(path).run(s, api_key=api_key, force=force)
    except Exception as e:
        print()
        print(f"❌ エラーで止まりました:{type(e).__name__}: {e}")
        print("もう一度「SRT作成」を押すと、完了した工程は飛ばして続きから再開します。")
        print("繰り返し失敗する場合は、以下の内容をXのDMでお知らせください。")
        traceback.print_exc()
        return

    with result_out:
        display(HTML(_download_link(result.srt_path, result.srt_text)))
        preview = html.escape(result.srt_text)
        display(
            HTML(
                '<details open style="margin-top:8px"><summary>プレビュー</summary>'
                f'<pre style="max-height:360px;overflow:auto;background:#f7f7f7;color:#222;padding:8px">{preview}</pre>'
                "</details>"
            )
        )
