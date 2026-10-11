"""文の組み立て:Whisper の区間をつなぎ、「。」の無い文の切れ目を推定する。

Whisper は「。」を付けないことがあり、また話の途中で区間を切ることもある。
そこで
  1. 全区間の文字を1本につなぐ(長い無音のところだけは文の切れ目にする)
  2. 品詞と無音の長さから、句読点の無い文の切れ目を推定して「。」を補う
     (「〜です|私も」「〜よね|私も」「こんにちは|今日は」など)
  3. 応答の言葉の後ろ(「なるほど|平日の」「はい|例えば」)には「、」を補う
     (ここは改行・分割しやすい位置になる)
  4. 「。」「?」「!」で文に分ける

補った句読点は字幕には表示されない(最後に「、」「。」は削除される)。
"""

from __future__ import annotations

from typing import List

from .chars import Char, split_sentences, text_of
from .linebreak import tokenize

# この秒数以上の無音は、品詞にかかわらず文の切れ目とみなす
LONG_PAUSE = 1.0
# 「〜た」「〜る」などで終わる場合は、この秒数以上の無音があるときだけ文末とみなす
# (「合わせた|キーワード」のような名詞を修飾する形と区別するため)
PAUSE_FOR_AMBIGUOUS_END = 0.3
PAUSE_FOR_PLAIN_VERB_END = 0.6

# 感動詞だが応答・相づちではなく、前の言葉に続けて言う語
NOT_RESPONSE = {"ありがとう", "おめでとう", "すみません", "ごめん", "ごめんなさい"}

# 形態素解析で2つに分かれてしまう1語(この間には句読点を補わない)
JOINED = {"えっと", "ええと", "えーと", "えーっと", "じゃあ", "あのー", "そのー", "まあ", "うーん"}

GREETINGS = {"こんにちは", "こんばんは", "おはよう", "さようなら", "よろしく"}

PUNCT = set("、。，,．.？！?!「」『』（）() ")


def _insert(chars: List[Char], marks: dict) -> List[Char]:
    """marks = {位置: 記号}。位置 i の文字の直前に記号を挿入する(長さ0のタイムコード)。"""
    if not marks:
        return chars
    out: List[Char] = []
    for i, c in enumerate(chars):
        if i in marks and out:
            t = out[-1]["e"]
            out.append({"c": marks[i], "s": t, "e": t})
        out.append(c)
    return out


def join_segments(segments: List[dict]) -> List[Char]:
    """Whisper の区間を1本の文字列につなぐ。長い無音のところには「。」を補う。"""
    stream: List[Char] = []
    for seg in segments:
        chars = [dict(c) for c in seg.get("chars", [])]
        if not chars:
            continue
        chars[0]["b"] = 1  # Whisper の区間の切れ目(切れ目の推定で使う)
        if stream:
            last = stream[-1]["c"]
            gap = chars[0]["s"] - stream[-1]["e"]
            if last not in PUNCT and gap >= LONG_PAUSE:
                t = stream[-1]["e"]
                stream.append({"c": "。", "s": t, "e": t})
        stream.extend(chars)
    return stream


def _is_start(t) -> bool:
    """新しい文節(自立語)の始まりか。"""
    if t.pos in ("助詞", "助動詞", "記号"):
        return False
    if t.pos == "名詞" and t.pos1 in ("非自立", "接尾"):
        return False
    if t.pos in ("動詞", "形容詞") and t.pos1 in ("非自立", "接尾"):
        return False
    return True


def _gap(chars: List[Char], i: int) -> float:
    if i <= 0 or i >= len(chars):
        return 0.0
    return max(0.0, chars[i]["s"] - chars[i - 1]["e"])


def infer_marks(chars: List[Char]) -> dict:
    """句読点を補う位置と記号 {位置: "。" or "、"}。"""
    text = text_of(chars)
    if not text:
        return {}
    tokens = tokenize(text)
    marks = {}
    for k in range(1, len(tokens)):
        p, q = tokens[k - 1], tokens[k]
        pos = q.start
        if pos <= 0 or pos >= len(text):
            continue
        if text[pos - 1] in PUNCT or text[pos] in PUNCT:
            continue
        if not _is_start(q):
            continue
        gap = _gap(chars, pos)
        prev = tokens[k - 2] if k >= 2 else None

        mark = None
        if p.surface + q.surface in JOINED:
            pass  # 「えっ|と」「じゃ|あ」は1語(誤分割)
        elif p.surface in GREETINGS:
            mark = "。"  # 「こんにちは|えっと」「こんにちは|今日は」
        elif p.pos in ("感動詞", "フィラー") and q.pos in ("感動詞", "フィラー"):
            pass  # 感動詞どうしの間には入れない(「えっ|と」「じゃ|あ」の誤分割で句読点が入るのを防ぐ)
        elif q.surface in NOT_RESPONSE:
            pass  # 「お時間いただいて|ありがとうございます」は1文のまま
        elif q.pos in ("感動詞", "フィラー") and p.pos in ("助詞", "助動詞", "動詞", "形容詞"):
            # 言い切りの後の応答・相づち(「〜続かなくて|ああなるほど」)は話者が替わることが多い
            mark = "。"
        elif q.pos in ("感動詞", "フィラー"):
            mark = "、"  # 「ハッシュタグ|そうですね」
        elif p.surface == "か" and prev is not None and prev.pos == "助動詞":
            mark = "。"  # 疑問の「〜ますか|」「〜ですか|」
        elif q.pos == "接続詞" and p.pos == "助詞" and p.pos1 == "接続助詞":
            mark = "、"  # 「〜ありますし|でも」
        elif p.pos == "助詞" and p.pos1 == "終助詞":
            # 「〜ですよね|私も」「〜ですね|やっぱり」(「私はね|」のような間投の「ね」は除く)
            if prev is not None and prev.pos in ("助動詞", "動詞", "形容詞", "助詞") and prev.pos1 != "係助詞":
                mark = "。"
        elif p.pos == "助動詞" and p.base in ("です", "ます") and p.infl == "基本形":
            mark = "。"  # 「〜です|簡単で」「〜ございます|早速」
        elif p.pos == "助動詞" and p.base in ("た", "だ") and p.infl == "基本形":
            if gap >= PAUSE_FOR_AMBIGUOUS_END:
                mark = "。"
        elif p.pos in ("動詞", "形容詞") and p.infl == "基本形":
            if gap >= PAUSE_FOR_PLAIN_VERB_END:
                mark = "。"
        elif p.pos == "感動詞" and p.surface in GREETINGS:
            mark = "。"  # 「こんにちは|今日は」
        elif p.pos in ("感動詞", "フィラー"):
            mark = "、"  # 「なるほど|」「はい|」(短い相づちだけの1枚を作らないよう、文は分けない)
        elif p.pos == "接続詞" and gap >= PAUSE_FOR_AMBIGUOUS_END:
            mark = "、"

        if mark is None and chars[pos].get("b"):
            # Whisper の区間の切れ目では、言い切りの形なら文末、節の切れ目なら読点を補う
            if p.pos in ("助動詞", "動詞", "形容詞") and p.infl == "基本形":
                mark = "。"
            elif p.pos == "助詞" and p.pos1 == "接続助詞":
                mark = "、"

        if mark:
            marks[pos] = mark
    return marks


def build_sentences(segments: List[dict], prefilter=None) -> List[List[Char]]:
    """Whisper の区間リストから文のリストを作る。

    フィラー(「あ」「えっと」など)は話者が替わる目印になるので、
    切れ目の推定はフィラーを残したまま行い、その後で除去する。
    """
    stream = join_segments(segments)
    sentences: List[List[Char]] = []
    # 既存の句点で一度分けてから、それぞれの中で切れ目を推定する(長い文字列を一度に解析しない)
    for piece in split_sentences(stream):
        filled = _insert(piece, infer_marks(piece))
        for sent in split_sentences(filled):
            if prefilter is not None:
                sent = prefilter(sent)
            if text_of(sent).strip(" 、。"):
                sentences.append(sent)
    return sentences
