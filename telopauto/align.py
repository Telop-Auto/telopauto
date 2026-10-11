"""文字単位のタイムコード付け(CTC 強制アライメント)。

WhisperX が日本語で使っているのと同じ音声認識モデル
(jonatasgrosman/wav2vec2-large-xlsr-53-japanese)を transformers で直接読み込み、
Whisper の本文を1文字ずつ音声に合わせる。

WhisperX 本体を使わないので、NumPy などのバージョン固定が不要。
また、アライメントに失敗した区間も本文を捨てず、
区間内で均等割りしたタイムコードを付けて残す。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .audio import SAMPLE_RATE
from .chars import Char, make_chars

ALIGN_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"

Log = Callable[[str], None]


def ctc_align(log_probs: np.ndarray, tokens: List[int], blank: int) -> Optional[List[tuple]]:
    """CTC の Viterbi 強制アライメント。

    log_probs: (フレーム数, 語彙数) の対数確率
    tokens:    並べたいトークンID列
    戻り値:    トークンごとの (開始フレーム, 終了フレーム[含まない])。
               フレームが足りないなどで合わせられない場合は None。
    """
    T = log_probs.shape[0]
    N = len(tokens)
    if N == 0:
        return []
    # 連続する同じトークンの間には blank が必要
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    if T < N + repeats:
        return None

    S = 2 * N + 1
    labels = np.full(S, blank, dtype=np.int64)
    labels[1::2] = tokens
    # s-2 → s へのスキップを許すか
    skip = np.zeros(S, dtype=bool)
    for s in range(3, S, 2):
        skip[s] = labels[s] != labels[s - 2]

    neg = -np.inf
    alpha = np.full(S, neg)
    alpha[0] = log_probs[0, blank]
    alpha[1] = log_probs[0, labels[1]]
    back = np.zeros((T, S), dtype=np.int8)  # 0: そのまま, 1: 1つ前から, 2: 2つ前から

    for t in range(1, T):
        stay = alpha
        prev1 = np.concatenate(([neg], alpha[:-1]))
        prev2 = np.concatenate(([neg, neg], alpha[:-2]))
        prev2 = np.where(skip, prev2, neg)
        stacked = np.stack([stay, prev1, prev2])
        choice = np.argmax(stacked, axis=0)
        alpha = stacked[choice, np.arange(S)] + log_probs[t, labels]
        back[t] = choice

    if alpha[S - 1] >= alpha[S - 2]:
        s = S - 1
    else:
        s = S - 2
    if not np.isfinite(alpha[s]):
        return None

    state_at = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        state_at[t] = s
        s -= back[t, s]

    spans = []
    for k in range(N):
        frames = np.nonzero(state_at == 2 * k + 1)[0]
        if len(frames) == 0:
            return None
        spans.append((int(frames[0]), int(frames[-1]) + 1))
    return spans


def _fill_unknown(chars: List[Char], known: List[bool], seg_start: float, seg_end: float) -> None:
    """辞書に無い文字(英字・記号など)に前後から補間したタイムコードを付ける。"""
    n = len(chars)
    i = 0
    while i < n:
        if known[i]:
            i += 1
            continue
        j = i
        while j < n and not known[j]:
            j += 1
        left = chars[i - 1]["e"] if i > 0 else seg_start
        right = chars[j]["s"] if j < n else seg_end
        if right < left:
            right = left
        step = (right - left) / (j - i)
        for k in range(i, j):
            chars[k]["s"] = left + step * (k - i)
            chars[k]["e"] = left + step * (k - i + 1)
        i = j


class Aligner:
    def __init__(self, device: str, log: Log = print):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        log("文字タイムコード用モデルを読み込んでいます...")
        self.torch = torch
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(ALIGN_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(ALIGN_MODEL).to(device).eval()
        vocab = self.processor.tokenizer.get_vocab()
        self.vocab: Dict[str, int] = {k.lower(): v for k, v in vocab.items()}
        self.blank = self.vocab.get("<pad>", self.vocab.get("[pad]", 0))

    def close(self) -> None:
        del self.model
        import gc

        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def _log_probs(self, wave: np.ndarray) -> np.ndarray:
        torch = self.torch
        inputs = torch.from_numpy(wave).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(inputs).logits[0]
            return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def align_segment(self, audio: np.ndarray, seg: dict) -> tuple:
        """1区間の本文に文字タイムコードを付ける。戻り値は (chars, 成功したか)。"""
        text = seg["text"]
        start, end = float(seg["start"]), float(seg["end"])
        a = max(0, int(start * SAMPLE_RATE))
        b = min(len(audio), int(end * SAMPLE_RATE))
        chars = make_chars(text, start, end)
        if b - a < SAMPLE_RATE // 10 or not text:
            return chars, False

        wave = audio[a:b]
        # wav2vec2 は正規化した波形を前提にしている
        wave = (wave - wave.mean()) / (wave.std() + 1e-7)
        log_probs = self._log_probs(wave.astype(np.float32))
        frame_sec = (b - a) / SAMPLE_RATE / log_probs.shape[0]

        known = [ch.lower() in self.vocab and not ch.isspace() for ch in text]
        tokens = [self.vocab[ch.lower()] for ch, ok in zip(text, known) if ok]
        if not tokens:
            return chars, False
        spans = ctc_align(log_probs, tokens, self.blank)
        if spans is None:
            return chars, False

        it = iter(spans)
        for ch, ok in zip(chars, known):
            if ok:
                f0, f1 = next(it)
                ch["s"] = start + f0 * frame_sec
                ch["e"] = start + f1 * frame_sec
        _fill_unknown(chars, known, start, end)
        return chars, True


def align(audio: np.ndarray, segments: List[dict], device: str, log: Log = print) -> List[dict]:
    """全区間に文字タイムコードを付ける。

    戻り値は [{"start", "end", "text", "chars", "aligned"}]。
    アライメントに失敗した区間も本文は残す(均等割りのタイムコード)。
    """
    aligner = Aligner(device, log)
    out = []
    failed = 0
    try:
        for k, seg in enumerate(segments, start=1):
            try:
                chars, ok = aligner.align_segment(audio, seg)
            except Exception:
                chars, ok = make_chars(seg["text"], seg["start"], seg["end"]), False
            if not ok:
                failed += 1
            out.append({**seg, "chars": chars, "aligned": ok})
            if k % 50 == 0 or k == len(segments):
                log(f"  文字タイムコード付け中... {k}/{len(segments)}")
    finally:
        aligner.close()
    if failed:
        log(f"⚠️ {failed}区間は文字単位で合わせられなかったため、区間内で均等にタイムコードを付けました。")
    return out
