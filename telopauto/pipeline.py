"""処理全体の流れと、工程ごとの保存・再利用。

工程:
  1. transcribe  文字起こし(Whisper)
  2. align       文字単位のタイムコード付け
  3. cleanup     文に分割 → フィラー除去 → ユーザー辞書置換
  4. proofread   Gemini 校正
  5. layout      ブロック分割・改行・表示時間 → SRT

各工程の結果は作業フォルダに JSON で保存する。保存時には
「その工程に関係する設定」と「前の工程の結果」から作った指紋を一緒に記録し、
次回実行時に指紋が同じなら保存済みの結果を使う。

そのため、設定を変えて「SRT作成」を押し直すだけで、
変更の影響がある工程からだけ自動でやり直される。
例:行数を 1→2 に変えた場合は 5. layout だけ、
   フィラーの強さを変えた場合は 3 以降だけがやり直しになる。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from . import __version__
from .chars import retime, strip_edges, text_of
from .cleanup import apply_dictionary, parse_dictionary, parse_word_list, remove_fillers
from .layout import LayoutSettings, layout, to_srt
from .sentences import build_sentences
from .settings import Settings

Log = Callable[[str], None]

# 文字タイムコード処理を直したら上げる(保存済みの結果を使わずにやり直させる)
ALIGN_VERSION = 2

STAGES = ["transcribe", "align", "cleanup", "proofread", "layout"]
STAGE_LABELS = {
    "transcribe": "文字起こし",
    "align": "文字タイムコード付け",
    "cleanup": "フィラー除去・辞書置換",
    "proofread": "Gemini校正",
    "layout": "字幕の分割・改行",
}


def default_work_root() -> str:
    return "/content/telopauto_work" if os.path.isdir("/content") else os.path.abspath("telopauto_work")


def file_fingerprint(path: str) -> str:
    """大きなファイルでも速く計算できるよう、先頭・末尾とサイズから作る。"""
    h = hashlib.sha1()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()[:10]


def _fp(*parts) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Result:
    srt_path: str
    srt_text: str
    block_count: int
    reran: List[str]
    reused: List[str]


class Pipeline:
    def __init__(self, audio_path: str, work_root: Optional[str] = None, log: Log = print):
        self.audio_path = audio_path
        self.log = log
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        self.stem = stem
        self.work_dir = os.path.join(work_root or default_work_root(), f"{stem}_{file_fingerprint(audio_path)}")
        os.makedirs(self.work_dir, exist_ok=True)
        self._audio = None

    # ---------- 保存 ----------

    def _path(self, stage: str) -> str:
        return os.path.join(self.work_dir, f"{STAGES.index(stage) + 1:02d}_{stage}.json")

    def load_stage(self, stage: str, fingerprint: str):
        path = self._path(stage)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            return None
        if saved.get("fingerprint") != fingerprint or not saved.get("complete", True):
            return None
        return saved["data"]

    def save_stage(self, stage: str, fingerprint: str, data, complete: bool = True) -> None:
        path = self._path(stage)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"fingerprint": fingerprint, "complete": complete, "version": __version__, "data": data},
                f,
                ensure_ascii=False,
            )
        os.replace(tmp, path)

    def clear(self) -> None:
        for stage in STAGES:
            p = self._path(stage)
            if os.path.exists(p):
                os.remove(p)

    # ---------- 共通 ----------

    def audio(self):
        if self._audio is None:
            from .audio import load_audio

            self.log("音声を読み込んでいます...")
            self._audio = load_audio(self.audio_path)
        return self._audio

    @staticmethod
    def device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    # ---------- 実行 ----------

    def run(
        self,
        settings: Settings,
        api_key: Optional[str] = None,
        force: bool = False,
        output_dir: Optional[str] = None,
        transcriber=None,
        aligner=None,
        gemini_client=None,
    ) -> Result:
        """全工程を実行する。transcriber / aligner / gemini_client はテスト用の差し替え口。"""
        s = settings.normalized()
        log = self.log
        if force:
            self.clear()
        reran, reused = [], []

        def step(stage, fp, compute):
            # 前の工程をやり直した場合は、それ以降もすべてやり直す
            data = None if reran else self.load_stage(stage, fp)
            if data is not None:
                reused.append(stage)
                log(f"▶ {STAGE_LABELS[stage]}:前回の結果を使います")
                return data
            log(f"▶ {STAGE_LABELS[stage]}")
            reran.append(stage)
            data, complete = compute()
            self.save_stage(stage, fp, data, complete)
            return data

        # 1. 文字起こし
        fp1 = _fp("transcribe", file_fingerprint(self.audio_path), s.whisper_model)

        def do_transcribe():
            if transcriber is not None:
                return transcriber(self), True
            from .transcribe import transcribe

            return transcribe(self.audio(), s.whisper_model, self.device(), log), True

        segments = step("transcribe", fp1, do_transcribe)

        # 2. 文字タイムコード
        fp2 = _fp("align", fp1, ALIGN_VERSION)

        def do_align():
            if aligner is not None:
                return aligner(self, segments), True
            from .align import align

            return align(self.audio(), segments, self.device(), log), True

        aligned = step("align", fp2, do_align)

        # 3. 文分割・フィラー・辞書
        dictionary = parse_dictionary(s.user_dictionary)
        custom_fillers = parse_word_list(s.custom_fillers)
        fp3 = _fp("cleanup", fp2, s.filler_level, custom_fillers, dictionary, __version__)

        def do_cleanup():
            sentences = []
            removed = 0
            for sent in build_sentences(aligned):
                before = len(sent)
                sent = remove_fillers(sent, s.filler_level, custom_fillers)
                removed += before - len(sent)
                sent = apply_dictionary(sent, dictionary)
                sent = strip_edges(sent)
                if text_of(sent).strip(" 、。"):
                    sentences.append(sent)
            log(f"  {len(sentences)}文(フィラー等で {removed}文字 削除)")
            return sentences, True

        sentences = step("cleanup", fp3, do_cleanup)

        # 4. Gemini 校正
        assist = parse_word_list(s.gemini_assist_words)
        fp4 = _fp("proofread", fp3, s.use_gemini, s.gemini_model, assist)

        def do_proofread():
            if not s.use_gemini:
                log("  Gemini校正は「しない」設定のためスキップしました。")
                return sentences, True
            client = gemini_client
            if client is None:
                if not api_key:
                    log("⚠️ Gemini APIキーが無いため、校正をスキップしました。")
                    return sentences, False
                from google import genai

                client = genai.Client(api_key=api_key)
            from .proofread import Proofreader

            cache_path = os.path.join(self.work_dir, "gemini_cache.json")
            cache = {}
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, encoding="utf-8") as f:
                        cache = json.load(f)
                except Exception:
                    cache = {}

            def save_cache():
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)

            pr = Proofreader(client, s.gemini_model, assist, cache, log)
            texts = [text_of(x) for x in sentences]
            corrected = pr.run(texts, on_batch_done=save_cache)
            save_cache()
            out, changed = [], 0
            for sent, new in zip(sentences, corrected):
                if new is not None and new != text_of(sent):
                    changed += 1
                    sent = strip_edges(retime(sent, new))
                out.append(sent)
            done = all(c is not None for c in corrected)
            log(f"  校正で修正された文:{changed}文")
            if not done:
                log("  ※校正できなかった文があります。次回「SRT作成」を押すと、その文だけ再度校正します。")
            return out, done

        proofread = step("proofread", fp4, do_proofread)

        # 5. 分割・改行
        protected = list(dictionary.values()) + assist
        ls = LayoutSettings(s.max_chars_per_line, s.lines, s.block_max_chars)
        fp5 = _fp("layout", fp4, ls.max_chars_per_line, ls.lines, ls.block_limit, protected, __version__)

        def do_layout():
            blocks = layout(proofread, ls, protected)
            return blocks, True

        blocks = step("layout", fp5, do_layout)

        srt_text = to_srt(blocks)
        out_dir = output_dir or (os.path.dirname(os.path.abspath(self.audio_path)))
        srt_path = os.path.join(out_dir, f"telopauto_{self.stem}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
        log(f"✅ 完成:{os.path.basename(srt_path)}({len(blocks)}ブロック)")
        return Result(srt_path, srt_text, len(blocks), reran, reused)
