"""Gemini による誤字・誤変換の校正。

v1.5 からの変更点:
・字幕ブロックではなく「文」単位で送る(改行は Gemini に任せず、後で機械的に決める)
・結果は JSON で受け取り、番号のずれや欠けを検出しやすくした
・言い換えすぎた結果(元の文と大きく違う)は採用せず元の文を残す
・校正結果をキャッシュに保存し、やり直し時や上限エラー後の再実行で同じ文を再送しない
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Sequence

Log = Callable[[str], None]

PROMPT_VERSION = "1.6-1"
DEFAULT_MODEL = "gemini-3.5-flash"
BATCH_SIZE = 60

# 元の文との類似度がこれ未満、または長さがこの範囲外なら採用しない
MIN_SIMILARITY = 0.6
LENGTH_RATIO = (0.75, 1.3)


class QuotaExceeded(Exception):
    pass


def build_prompt(items: List[dict], assist_words: Sequence[str]) -> str:
    words = "\n".join(f"・{w}" for w in assist_words) or "(なし)"
    data = json.dumps(items, ensure_ascii=False)
    return f"""あなたは日本語字幕の校正者です。
音声認識で作った字幕の文章リストを校正してください。

【やること】
・誤字脱字、音声認識の誤変換の修正
・漢字/かな表記の統一

【やらないこと】
・言い換え、要約、書き言葉への変換(話し言葉・口語はそのまま)
・フィラー(えー、あの 等)の削除
・句読点(、。)の追加・削除・移動
・文の結合や分割

【表記確認用の単語リスト】
次の語は正しい表記の参考です。文脈上その語の誤変換・表記揺れと判断できる場合だけ使ってください。
{words}

【入力】
各要素の id と text を持つ JSON 配列です。
{data}

【出力】
入力と同じ件数・同じ id の JSON 配列を出力してください。
各要素は {{"id": 番号, "text": "校正後の文"}} です。修正が無い文も元の text をそのまま入れてください。
JSON 以外は出力しないでください。"""


def _cache_key(model: str, assist_words: Sequence[str], text: str) -> str:
    raw = json.dumps([PROMPT_VERSION, model, sorted(assist_words), text], ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _parse(text: str) -> Dict[int, str]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    data = json.loads(text)
    result = {}
    for item in data:
        try:
            result[int(item["id"])] = str(item["text"])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def acceptable(original: str, corrected: str) -> bool:
    if not corrected.strip():
        return False
    ratio = len(corrected) / max(len(original), 1)
    if not (LENGTH_RATIO[0] <= ratio <= LENGTH_RATIO[1]):
        return False
    return SequenceMatcher(None, original, corrected).ratio() >= MIN_SIMILARITY


class Proofreader:
    def __init__(self, client, model: str, assist_words: Sequence[str], cache: dict, log: Log = print):
        self.client = client
        self.model = model
        self.assist_words = list(assist_words)
        self.cache = cache
        self.log = log

    def _call(self, prompt: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(response_mime_type="application/json")
        wait = 10
        for attempt in range(1, 6):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                text = resp.text
                if not text:
                    raise RuntimeError("Geminiの応答が空でした")
                return text
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    raise QuotaExceeded(msg) from e
                retryable = any(
                    k in msg for k in ("500", "502", "503", "504", "UNAVAILABLE", "timeout", "Timeout", "空")
                )
                if not retryable or attempt == 5:
                    raise
                self.log(f"  Geminiが混み合っています。{wait}秒後に再試行します({attempt}/5)")
                time.sleep(wait)
                wait = min(wait * 2, 120)
        raise RuntimeError("unreachable")

    def run(self, texts: List[str], on_batch_done: Optional[Callable[[], None]] = None) -> List[Optional[str]]:
        """各文の校正結果を返す。校正できなかった文は None。"""
        results: List[Optional[str]] = [None] * len(texts)
        pending = []
        for i, t in enumerate(texts):
            hit = self.cache.get(_cache_key(self.model, self.assist_words, t))
            if hit is not None:
                results[i] = hit
            else:
                pending.append(i)

        if len(pending) < len(texts):
            self.log(f"  保存済みの校正結果を {len(texts) - len(pending)}文 再利用しました。")
        if not pending:
            return results

        batches = [pending[k : k + BATCH_SIZE] for k in range(0, len(pending), BATCH_SIZE)]
        rejected = 0
        for b, idxs in enumerate(batches, start=1):
            self.log(f"  Gemini校正 {b}/{len(batches)}...")
            items = [{"id": n, "text": texts[i]} for n, i in enumerate(idxs, start=1)]
            prompt = build_prompt(items, self.assist_words)
            try:
                parsed = _parse(self._call(prompt))
            except QuotaExceeded:
                self.log("⚠️ Gemini APIの利用上限に達しました。残りの文は校正せずに続行します。")
                self.log("   時間をおいて「SRT作成」を押し直すと、校正済みの文はそのままで残りだけ校正します。")
                break
            except Exception as e:
                self.log(f"⚠️ このまとまりは校正できませんでした({type(e).__name__})。元の文のまま続行します。")
                continue

            for n, i in enumerate(idxs, start=1):
                corrected = parsed.get(n)
                if corrected is None:
                    continue
                if not acceptable(texts[i], corrected):
                    rejected += 1
                    corrected = texts[i]
                results[i] = corrected
                self.cache[_cache_key(self.model, self.assist_words, texts[i])] = corrected
            if on_batch_done:
                on_batch_done()
            if b < len(batches):
                time.sleep(2)

        if rejected:
            self.log(f"  元の文から変わりすぎた校正 {rejected}件 は採用せず、元の文を残しました。")
        return results
