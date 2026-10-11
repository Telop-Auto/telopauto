"""文章 → 字幕ブロック(分割・改行・表示時間)→ SRT。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .chars import Char, strip_edges, text_of
from .linebreak import INF, boundary_costs, display_len, display_text, wrap

# ブロックを1つ増やすコスト(少ないブロックで収まるならそちらを選ぶ)
BLOCK_PENALTY = 4.0
# 1行の文字数 × この割合より短いブロックは避ける(「一人旅とか」だけの1枚など)
SHORT_BLOCK_RATIO = 0.4
SHORT_BLOCK_WEIGHT = 1.0
# ブロックの長さをそろえる重み(目標は上限の75%)。
# 同程度の区切り方が複数あるとき、極端に長い/短いブロックができない方を選ぶ
BLOCK_BALANCE_WEIGHT = 0.02
BLOCK_TARGET_RATIO = 0.75
# 区切り位置の無音1秒あたりのボーナス(無音のところで分けやすくする)
GAP_BONUS_PER_SEC = 4.0
GAP_BONUS_MAX_SEC = 1.0


@dataclass
class LayoutSettings:
    max_chars_per_line: int = 20
    lines: int = 2
    block_max_chars: int = 40

    @property
    def block_limit(self) -> int:
        limit = self.max_chars_per_line * self.lines
        return max(1, min(self.block_max_chars, limit))


def _gap(chars: List[Char], i: int) -> float:
    """chars[i-1] と chars[i] の間の無音(秒)。"""
    if i <= 0 or i >= len(chars):
        return 0.0
    return max(0.0, chars[i]["s"] - chars[i - 1]["e"])


def _relax(costs: List[float]) -> List[float]:
    return [100.0 if c == INF else c for c in costs]


def split_sentence(
    chars: List[Char],
    settings: LayoutSettings,
    protected_words: Sequence[str] = (),
) -> List[dict]:
    """1文をブロックに分け、各ブロックを改行する。

    戻り値: [{"lines": [...], "chars": [...]}]
    """
    chars = strip_edges(chars)
    text = text_of(chars)
    n = len(text)
    if n == 0:
        return []

    costs = boundary_costs(text, protected_words)
    result = _partition(chars, text, costs, settings)
    if result is None:
        # 登録語が長すぎるなどで収まらない場合は禁則を緩めて再計算
        result = _partition(chars, text, _relax(costs), settings)
    if result is None:
        result = _hard_cut(chars, text, settings)
    return result


def _partition(chars, text, costs, settings: LayoutSettings):
    n = len(text)
    limit = settings.block_limit
    max_chars = settings.max_chars_per_line
    max_lines = settings.lines

    short_block = max(4, round(max_chars * SHORT_BLOCK_RATIO))
    cuts = [i for i in range(1, n) if costs[i] != INF]
    points = [0] + cuts + [n]
    index_of = {p: k for k, p in enumerate(points)}

    best = [INF] * len(points)
    back = [-1] * len(points)
    lines_of = [None] * len(points)
    best[0] = 0.0

    for jk in range(1, len(points)):
        j = points[jk]
        for ik in range(jk - 1, -1, -1):
            i = points[ik]
            if best[ik] == INF:
                continue
            seg = text[i:j]
            length = display_len(seg)
            if length > limit:
                # これより前から始めるとさらに長くなるので打ち切り
                if display_len(text[i:j]) > limit * 2:
                    break
                continue
            if length == 0:
                continue
            wcost, wlines = wrap(seg, costs, i, max_chars, max_lines)
            if wcost == INF:
                continue
            cost = best[ik] + wcost + BLOCK_PENALTY
            # 短いブロックは避ける。ただし文末や句読点・応答の後で自然に切れる場合は除く(「ああなるほど」)
            natural_end = j == n or costs[j] == 0.0
            if length < short_block and not natural_end:
                cost += (short_block - length) * SHORT_BLOCK_WEIGHT
            cost += BLOCK_BALANCE_WEIGHT * (limit * BLOCK_TARGET_RATIO - length) ** 2
            if i > 0:
                cut = costs[i] * 1.5
                cut -= GAP_BONUS_PER_SEC * min(_gap(chars, i), GAP_BONUS_MAX_SEC)
                cost += cut
            if cost < best[jk]:
                best[jk] = cost
                back[jk] = ik
                lines_of[jk] = wlines

    if best[-1] == INF:
        return None

    blocks = []
    jk = len(points) - 1
    while jk > 0:
        ik = back[jk]
        i, j = points[ik], points[jk]
        blocks.append({"lines": lines_of[jk], "start": i, "end": j})
        jk = ik
    blocks.reverse()
    return [
        {
            "lines": [display_text(l) for l in b["lines"] if display_text(l)],
            "chars": chars[b["start"] : b["end"]],
        }
        for b in blocks
        if display_text(text[b["start"] : b["end"]])
    ]


def _hard_cut(chars, text, settings: LayoutSettings):
    """最終手段:最大文字数で機械的に区切る。"""
    out = []
    step = settings.max_chars_per_line
    per_block = step * settings.lines
    for a in range(0, len(text), per_block):
        seg = text[a : a + per_block]
        lines = [seg[k : k + step] for k in range(0, len(seg), step)]
        lines = [display_text(l) for l in lines if display_text(l)]
        if lines:
            out.append({"lines": lines, "chars": chars[a : a + per_block]})
    return out


def layout(
    sentences: List[List[Char]],
    settings: LayoutSettings,
    protected_words: Sequence[str] = (),
) -> List[dict]:
    """文章のリストから字幕ブロックのリストを作る。

    戻り値: [{"text": "1行目\\n2行目", "start": 秒, "end": 秒}]
    """
    blocks = []
    for sent in sentences:
        parts = []
        for b in split_sentence(sent, settings, protected_words):
            visible = [c for c in b["chars"] if c["c"] not in "、。" and not c["c"].isspace()]
            ref = visible or b["chars"]
            parts.append({"text": "\n".join(b["lines"]), "start": ref[0]["s"], "end": ref[-1]["e"]})
        # 同じ文の中で分けたテロップは、言い終わりで消さずに次のテロップまでつなぐ(v1.5 と同じ)
        for a, b in zip(parts, parts[1:]):
            a["end"] = max(a["end"], b["start"])
        blocks.extend(parts)
    blocks.sort(key=lambda b: b["start"])
    return fix_timing(blocks)


# 次の字幕までの隙間がこれより短ければ、表示をつないで隙間を作らない(一瞬消えるちらつき防止)
CLOSE_GAP_SEC = 1.0


def fix_timing(
    blocks: List[dict],
    sec_per_char: float = 0.3,
    min_duration: float = 0.8,
    close_gap: float = CLOSE_GAP_SEC,
) -> List[dict]:
    """最低表示時間の確保と、前後の字幕の重なり防止。

    ・文字数 × sec_per_char(最低 min_duration)より短い字幕は終了を延ばす
    ・次の字幕までの隙間が close_gap 秒未満なら、次の字幕の開始までつなぐ
    ・ただし次の字幕の開始時刻は越えない
    """
    out = [dict(b) for b in blocks]
    for k, b in enumerate(out):
        nxt = out[k + 1]["start"] if k + 1 < len(out) else None
        chars = len(b["text"].replace("\n", ""))
        required = max(chars * sec_per_char, min_duration)
        if b["end"] - b["start"] < required:
            want = b["start"] + required
            b["end"] = min(want, nxt) if nxt is not None else want
        if nxt is not None and 0 < nxt - b["end"] < close_gap:
            b["end"] = nxt
        if nxt is not None and b["end"] > nxt:
            b["end"] = nxt
        if b["end"] <= b["start"]:
            b["end"] = b["start"] + 0.1
    return out


def sec_to_srt_time(sec: float) -> str:
    ms = int(round(max(sec, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(blocks: List[dict]) -> str:
    parts = []
    k = 0
    for b in blocks:
        if not b["text"].strip():
            continue
        k += 1
        parts.append(
            f"{k}\n{sec_to_srt_time(b['start'])} --> {sec_to_srt_time(b['end'])}\n{b['text']}"
        )
    return "\n\n".join(parts) + "\n"
