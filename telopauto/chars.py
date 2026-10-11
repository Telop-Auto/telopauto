"""文字単位タイムコード付きテキストの共通処理。

Telop Auto の中間データはすべて「文字のリスト」で表す。
1文字ごとに {"c": 文字, "s": 開始秒, "e": 終了秒} を持ち、
文章の本文は常に "".join(c["c"] for c in chars) と一致する。

本文を書き換える処理(フィラー除去・辞書置換・Gemini校正)は
retime() で新しい本文へタイムコードを引き継ぐ。
これにより、どの工程の後でも文字とタイムコードの対応が崩れない。
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable, List

Char = dict  # {"c": str, "s": float, "e": float}

SENTENCE_END = set("。？！?!")


def text_of(chars: Iterable[Char]) -> str:
    return "".join(c["c"] for c in chars)


def make_chars(text: str, start: float, end: float) -> List[Char]:
    """タイムコードが無い本文に、区間を均等割りしたタイムコードを付ける。"""
    n = len(text)
    if n == 0:
        return []
    span = max(end - start, 0.0)
    step = span / n
    return [
        {"c": ch, "s": start + step * i, "e": start + step * (i + 1)}
        for i, ch in enumerate(text)
    ]


def retime(chars: List[Char], new_text: str) -> List[Char]:
    """chars の本文を new_text に置き換え、タイムコードを引き継ぐ。

    ・変わっていない文字はそのままのタイムコード
    ・置き換えられた文字は、元の文字が占めていた時間を均等割り
    ・挿入された文字は、直前の文字の終了時刻に長さ0で置く
    ・削除された文字の時間は捨てる
    """
    old_text = text_of(chars)
    if old_text == new_text:
        return [dict(c) for c in chars]
    if not chars:
        return []

    out: List[Char] = []
    sm = SequenceMatcher(None, old_text, new_text, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend(dict(c) for c in chars[i1:i2])
        elif tag == "replace":
            s = chars[i1]["s"]
            e = chars[i2 - 1]["e"]
            out.extend(make_chars(new_text[j1:j2], s, e))
        elif tag == "insert":
            if i1 > 0:
                t = chars[i1 - 1]["e"]
            else:
                t = chars[0]["s"]
            out.extend({"c": ch, "s": t, "e": t} for ch in new_text[j1:j2])
        # delete: 何もしない
    return out


def split_sentences(chars: List[Char]) -> List[List[Char]]:
    """句点・感嘆符・疑問符の直後で分ける(記号は前の文に含める)。"""
    sentences: List[List[Char]] = []
    cur: List[Char] = []
    for i, c in enumerate(chars):
        cur.append(c)
        nxt = chars[i + 1]["c"] if i + 1 < len(chars) else ""
        if c["c"] in SENTENCE_END and nxt not in SENTENCE_END and nxt not in "」』）)":
            sentences.append(cur)
            cur = []
        elif c["c"] in "」』）)" and len(cur) >= 2 and cur[-2]["c"] in SENTENCE_END:
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    return [s for s in sentences if text_of(s).strip()]


def strip_edges(chars: List[Char]) -> List[Char]:
    """前後の空白文字を取り除く。"""
    i, j = 0, len(chars)
    while i < j and chars[i]["c"].isspace():
        i += 1
    while j > i and chars[j - 1]["c"].isspace():
        j -= 1
    return chars[i:j]


def span(chars: List[Char]) -> tuple:
    """(開始秒, 終了秒)。"""
    return chars[0]["s"], chars[-1]["e"]
