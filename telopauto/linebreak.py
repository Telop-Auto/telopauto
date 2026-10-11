"""改行・分割位置の評価。

文章中の「ここで区切ったらどれくらい不自然か」を位置ごとのコストで表す。
コストが低いほど自然な区切り位置。

基本の考え方は文節(意味のまとまり)単位で区切ること。
・文節の途中(「入ら|ない」「協力|して」「かけ|流し」など)は高コスト
・読点・接続助詞・助詞の後ろなど、文節の切れ目は低コスト
・ユーザー辞書やGemini補助辞書に登録した語の途中では区切らない

改行(1ブロック内の行分け)とブロック分割の両方がこのコストを使う。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

INF = float("inf")

# 字幕表示時に削除される文字(文字数に数えない)
HIDDEN = set("、。")

# 行頭に来てはいけない文字
NO_LINE_START = set("、。，,．.？！?!」』）)]ー〜～…・ゃゅょぁぃぅぇぉっャュョァィゥェォッ")
# 行末に来てはいけない文字
NO_LINE_END = set("「『（([")

# 単語の内部(形態素の途中)で区切る場合のコスト。最終手段。
INSIDE_WORD = 40.0

# 「て/で + 補助動詞」は区切ってもよい(協力して|もらって)
AUX_VERBS_OK = {"もらう", "いただく", "くれる", "くださる", "あげる"}

# サ変名詞の直後に来る動詞(この間では区切らない)
SURU_VERBS = {"する", "できる", "いたす", "なさる", "させる", "される"}

# 2行の長さの差1文字あたりのコスト
BALANCE_WEIGHT = 0.25


@dataclass
class Token:
    surface: str
    start: int
    end: int
    pos: str
    pos1: str
    infl: str
    base: str


_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from janome.tokenizer import Tokenizer

        _tokenizer = Tokenizer()
    return _tokenizer


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    for t in _get_tokenizer().tokenize(text):
        surface = t.surface
        start = text.find(surface, pos)
        if start < 0:
            start = pos
        end = start + len(surface)
        pos = end
        parts = t.part_of_speech.split(",")
        tokens.append(
            Token(
                surface=surface,
                start=start,
                end=end,
                pos=parts[0],
                pos1=parts[1] if len(parts) > 1 else "*",
                infl=getattr(t, "infl_form", "*") or "*",
                base=getattr(t, "base_form", surface) or surface,
            )
        )
    return tokens


def display_text(text: str) -> str:
    """字幕に表示される形(句読点を除き、前後の空白を除く)。"""
    return "".join(ch for ch in text if ch not in HIDDEN).strip()


def display_len(text: str) -> int:
    return len(display_text(text))


def _pair_cost(p: Token, q: Token) -> float:
    """トークン p の直後(q の直前)で区切るコスト。"""
    P, P1 = p.pos, p.pos1
    Q, Q1 = q.pos, q.pos1

    # ---- 記号 ----
    if P == "記号":
        if p.surface in ("、", "，", ","):
            return 0.0
        if p.surface in ("。", "？", "！", "?", "!"):
            return 0.0
        if p.surface in ("」", "』", "）", ")"):
            return 1.0
        if p.surface.isspace():
            return 1.0
        return 3.0
    if Q == "記号":
        if q.surface.isspace():
            return 1.0
        if q.surface in ("「", "『", "（", "("):
            return 1.0
        return INF if q.surface[:1] in NO_LINE_START else 3.0
    if P == "接頭詞":
        return 30.0

    # ---- 文節の途中(付属語が続く) ----
    if Q == "助詞":
        # 名詞と助詞の間は、どうしても必要な場合のみ(源泉かけ流し|の温泉)
        return 7.0 if P == "名詞" else 12.0
    if Q == "助動詞":
        return 12.0
    if Q == "名詞" and Q1 in ("接尾", "非自立"):
        return 12.0
    if Q in ("動詞", "形容詞") and Q1 == "非自立":
        if Q == "動詞" and p.surface in ("て", "で") and q.base in AUX_VERBS_OK:
            return 3.0  # 協力して|もらって
        return 12.0
    if Q == "動詞" and q.base in SURU_VERBS and P == "名詞":
        return 12.0  # 協力|して、投稿|して

    # ---- 複合語 ----
    if P == "名詞" and Q == "名詞":
        return 8.0  # 温泉女子|旅、観光|資源
    if P == "動詞" and p.infl.startswith("連用") and Q in ("名詞", "動詞"):
        return 8.0  # かけ|流し、組み|合わせる
    if P == "名詞" and Q == "動詞":
        return 5.0  # 助詞の省略(説得力|あります)や誤解析

    # ---- 文節の切れ目 ----
    if P in ("フィラー", "感動詞"):
        return 0.5
    if P == "接続詞":
        return 1.0
    if P == "助詞":
        if P1 == "接続助詞":
            return 1.0  # 〜て|、〜けど|、〜ので|
        if P1 == "終助詞":
            return 1.0
        if P1 == "係助詞":
            return 2.0 if p.surface == "も" else 1.5  # 〜は|、〜も|
        if P1 in ("格助詞", "並立助詞"):
            return 2.0  # 〜を|、〜に|、〜とか|
        if P1 == "連体化":
            return 4.0  # 旅館の|名前
        return 2.5
    if P == "副詞":
        return 2.5
    if P == "連体詞":
        return 6.0
    if Q == "名詞" and (P == "形容詞" or (P == "助動詞" and p.surface == "な")):
        return 6.0  # 短い修飾語と名詞(リアルな|声、大きい|声)
    if Q == "名詞" and P in ("動詞", "助動詞"):
        return 3.0  # 連体修飾節(入らない|状態、合わせた|キーワード)
    return 3.0


def _protected_spans(text: str, words: Iterable[str]) -> List[Tuple[int, int]]:
    spans = []
    for w in words:
        w = w.strip()
        if len(w) < 2:
            continue
        start = text.find(w)
        while start != -1:
            spans.append((start, start + len(w)))
            start = text.find(w, start + 1)
    return spans


def boundary_costs(text: str, protected_words: Sequence[str] = ()) -> List[float]:
    """位置 i(text[i] の直前)で区切るコストのリスト(長さ len(text)+1)。"""
    n = len(text)
    costs = [INSIDE_WORD] * (n + 1)
    costs[0] = 0.0
    costs[n] = 0.0
    if n == 0:
        return costs

    tokens = tokenize(text)
    for p, q in zip(tokens, tokens[1:]):
        if 0 < q.start < n:
            costs[q.start] = _pair_cost(p, q)

    # 行頭・行末の禁則
    for i in range(1, n):
        if text[i] in NO_LINE_START:
            costs[i] = INF
        if text[i - 1] in NO_LINE_END:
            costs[i] = INF

    # 登録語の途中では区切らない
    for a, b in _protected_spans(text, protected_words):
        for i in range(a + 1, b):
            costs[i] = INF

    return costs


def wrap(
    text: str,
    costs: Sequence[float],
    offset: int,
    max_chars: int,
    max_lines: int,
) -> Tuple[float, List[str]]:
    """text(全体の offset 文字目から始まる部分)を行に分ける。

    戻り値は (コスト, 行のリスト)。収まらない場合はコスト INF。
    """
    total = display_len(text)
    if total <= max_chars:
        return 0.0, [text]
    if max_lines < 2:
        return INF, [text]

    best: Tuple[float, Optional[int]] = (INF, None)
    for k in range(1, len(text)):
        c = costs[offset + k]
        if c == INF:
            continue
        l1 = display_len(text[:k])
        l2 = display_len(text[k:])
        if l1 == 0 or l2 == 0 or l1 > max_chars:
            continue
        if l2 > max_chars:
            if max_lines == 2:
                continue
            sub_cost, _ = wrap(text[k:], costs, offset + k, max_chars, max_lines - 1)
            score = c + sub_cost
        else:
            score = c + BALANCE_WEIGHT * abs(l1 - l2)
        if score < best[0]:
            best = (score, k)

    if best[1] is None:
        return INF, [text]
    k = best[1]
    first = text[:k]
    rest = text[k:]
    if display_len(rest) > max_chars:
        _, rest_lines = wrap(rest, costs, offset + k, max_chars, max_lines - 1)
    else:
        rest_lines = [rest]
    return best[0], [first] + rest_lines
