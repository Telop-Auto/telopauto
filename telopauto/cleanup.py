"""フィラー除去とユーザー辞書置換(文単位・タイムコード維持)。"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

from .chars import Char, retime, strip_edges, text_of

FILLERS = {
    "弱": ["えーっと", "えーと", "ええと", "えっと", "えー"],
    "中": [
        "えーっと", "えーと", "ええと", "えっと", "えー",
        "あの、", "その、", "まぁ", "まあ、", "うーん",
    ],
    "強": [
        "えーっと", "えーと", "ええと", "えっと", "えー",
        "あの、", "その、", "まぁ", "まあ、", "うーん",
        "そうですね", "なんか、", "こう、",
    ],
}


# 形態素解析で「フィラー」と判定されたときだけ消す語
# (「あの店」の「あの」のように意味のある使い方は消さない)
TOKEN_FILLERS = {
    "弱": set(),
    "中": {"あの", "あのー", "その", "そのー", "まあ", "まぁ", "あー", "あ", "え"},
    "強": {"あの", "あのー", "その", "そのー", "まあ", "まぁ", "あー", "あ", "え", "そうですね", "なんか", "こう"},
}


def _remove_filler_tokens(text: str, level: str) -> str:
    targets = TOKEN_FILLERS.get(level, TOKEN_FILLERS["中"])
    if not targets or not text:
        return text
    from .linebreak import tokenize

    out = []
    for t in tokenize(text):
        if t.pos == "フィラー" and t.surface in targets:
            continue
        out.append(t.surface)
    return "".join(out)


def remove_fillers_text(text: str, level: str = "中", custom: Iterable[str] = ()) -> str:
    words = list(FILLERS.get(level, FILLERS["中"])) + [w for w in custom if w]
    for w in sorted(set(words), key=len, reverse=True):
        text = text.replace(w, "")
    # 「あ、」「え、」は文頭のみ削除(「ああ、」「ええ、」は残す)
    # 文頭に空白が残っていると判定できないため、先に取り除く
    text = text.lstrip()
    text = re.sub(r"^[あえ][、。]\s*", "", text)
    # 読点の付いていないフィラー(「あの例えば」の「あの」など)
    text = _remove_filler_tokens(text, level)
    text = text.lstrip()
    # フィラー削除で残った文頭の読点
    text = re.sub(r"^[、\s]+", "", text)
    text = re.sub(r"、{2,}", "、", text)
    text = re.sub(r"\s+", " ", text)
    return text


def remove_fillers(sentence: List[Char], level: str = "中", custom: Iterable[str] = ()) -> List[Char]:
    new_text = remove_fillers_text(text_of(sentence), level, custom)
    return strip_edges(retime(sentence, new_text))


def parse_dictionary(raw: str) -> Dict[str, str]:
    """「置換前=置換後」を改行または | 区切りで並べた文字列を辞書にする。"""
    result: Dict[str, str] = {}
    for item in re.split(r"[|\n]", raw or ""):
        item = item.strip()
        if "=" not in item:
            continue
        before, after = item.split("=", 1)
        before, after = before.strip(), after.strip()
        if before:
            result[before] = after
    return result


def parse_word_list(raw: str) -> List[str]:
    return [w.strip() for w in re.split(r"[,、，\n]", raw or "") if w.strip()]


def apply_dictionary(sentence: List[Char], dictionary: Dict[str, str]) -> List[Char]:
    text = text_of(sentence)
    new_text = text
    # 長い語から置換(短い語が先に当たって長い語が置換できなくなるのを防ぐ)
    for before in sorted(dictionary, key=len, reverse=True):
        new_text = new_text.replace(before, dictionary[before])
    if new_text == text:
        return sentence
    return retime(sentence, new_text)
