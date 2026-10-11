import json

import numpy as np
import pytest

from telopauto.align import _fill_unknown, ctc_align
from telopauto.chars import make_chars, retime, split_sentences, text_of
from telopauto.cleanup import apply_dictionary, parse_dictionary, remove_fillers
from telopauto.layout import LayoutSettings, fix_timing, split_sentence, to_srt
from telopauto.linebreak import display_len
from telopauto.pipeline import Pipeline
from telopauto.settings import Settings


# ---------------- chars ----------------

def test_retime_keeps_unchanged_timings():
    chars = make_chars("厳選かけ流し", 0.0, 6.0)
    out = retime(chars, "源泉かけ流し")
    assert text_of(out) == "源泉かけ流し"
    # 置換部分は元の2文字分の時間を使う
    assert out[0]["s"] == 0.0 and out[1]["e"] == pytest.approx(2.0)
    # 変わっていない部分はそのまま
    assert out[2:] == chars[2:]


def test_retime_insert_and_delete():
    chars = make_chars("あいうえお", 0.0, 5.0)
    out = retime(chars, "あいXうお")
    assert text_of(out) == "あいXうお"
    assert all(a["s"] <= b["s"] for a, b in zip(out, out[1:]))


def test_split_sentences():
    chars = make_chars("こんにちは。今日は?はい!", 0, 12)
    assert [text_of(s) for s in split_sentences(chars)] == ["こんにちは。", "今日は?", "はい!"]


# ---------------- cleanup ----------------

def test_fillers_removed_with_timing():
    chars = make_chars("えーっと、あの、今日はですね。", 0, 15)
    out = remove_fillers(chars, "中")
    assert text_of(out) == "今日はですね。"
    assert out[0]["s"] == pytest.approx(chars[8]["s"])


def test_dictionary_across_old_line_break():
    # v1.5 では改行後に置換していたため「厳選かけ\n流し」が置換されなかった
    d = parse_dictionary("プレミア=Premiere | 厳選かけ流し=源泉かけ流し\nアイフォン=iPhone")
    assert d == {"プレミア": "Premiere", "厳選かけ流し": "源泉かけ流し", "アイフォン": "iPhone"}
    chars = make_chars("うちは厳選かけ流しの温泉です。", 0, 10)
    assert text_of(apply_dictionary(chars, d)) == "うちは源泉かけ流しの温泉です。"


# ---------------- align ----------------

def test_ctc_align_finds_token_positions():
    V, blank = 5, 0
    T = 20
    lp = np.full((T, V), np.log(0.01))
    lp[:, blank] = np.log(0.9)
    expected = {1: (3, 5), 2: (9, 11), 3: (15, 16)}
    for tok, (a, b) in expected.items():
        lp[a:b, tok] = np.log(0.95)
        lp[a:b, blank] = np.log(0.01)
    spans = ctc_align(lp, [1, 2, 3], blank)
    assert spans == [expected[1], expected[2], expected[3]]


def test_ctc_align_repeated_tokens_need_blank():
    lp = np.log(np.full((3, 3), 1 / 3))
    assert ctc_align(lp, [1, 1, 1], 0) is None  # 3トークン+間のblank2つ=5フレーム必要
    assert ctc_align(np.log(np.full((5, 3), 1 / 3)), [1, 1, 1], 0) is not None


def test_fill_unknown_interpolates():
    chars = [{"c": c, "s": 0.0, "e": 0.0} for c in "SNSも"]
    chars[3].update(s=2.0, e=2.2)
    _fill_unknown(chars, [False, False, False, True], 1.0, 3.0)
    assert [round(c["s"], 3) for c in chars[:3]] == [1.0, 1.333, 1.667]
    assert chars[2]["e"] == pytest.approx(2.0)


# ---------------- layout ----------------

ST2 = LayoutSettings(max_chars_per_line=20, lines=2, block_max_chars=40)


def _lines(text, settings=ST2, protected=()):
    return [b["lines"] for b in split_sentence(make_chars(text, 0, 10), settings, protected)]


def test_no_break_inside_word():
    # 「入ら|ない」のような単語途中の改行をしない
    (lines,) = _lines("特に平日はほとんど予約が入らない状態が続いてるんです")
    joined = "|".join(lines)
    assert "入ら|ない" not in joined


def test_protected_word_not_split():
    (lines,) = _lines("そうですねうちはやっぱり源泉かけ流しの温泉が一番の自慢なんですけど", protected=["源泉かけ流し"])
    assert "源泉かけ流し" in lines[0]  # 登録語の途中で切れない


def test_suru_verb_not_split():
    (lines,) = _lines("スタッフさんとかお客様に協力してもらって写真を撮ってもらったり")
    assert "協力|して" not in "|".join(lines)
    assert lines == ["スタッフさんとかお客様に協力してもらって", "写真を撮ってもらったり"]


def test_clause_boundary_preferred_over_balance():
    # 「〜ので|そこは」:節の切れ目を優先し、「そこは」を行末に取り残さない
    (lines,) = _lines("見てから予約する人が多いのでそこはやっぱり力を入れたいところですよね")
    assert lines == ["見てから予約する人が多いので", "そこはやっぱり力を入れたいところですよね"]


def test_blocks_in_sentence_are_connected():
    from telopauto.layout import layout

    sent = make_chars("はい例えば旅館の名前と温泉女子旅とか一人旅とかターゲットに合わせたキーワードを組み合わせると拡散しやすいです", 0, 16)
    # 「一人旅とか」の後に0.8秒の間を入れる
    for c in sent[24:]:
        c["s"] += 0.8
        c["e"] += 0.8
    blocks = layout([sent], LayoutSettings(20, 1, 20))
    for a, b in zip(blocks, blocks[1:]):
        assert a["end"] == pytest.approx(b["start"])  # 同じ文の中は隙間なくつながる
    assert all(len(b["text"]) >= 8 for b in blocks)  # 極端に短いブロックを作らない


def test_sentence_inference():
    from telopauto.sentences import build_sentences

    segs = []
    t = 0.0
    for text in ["こんにちは今日はお時間いただいてありがとうございます", "そうなんですよね私もやらなきゃと思ってはいるんですけど",
                 "考えたことなかったですなんか簡単で覚えやすいのがいいですよね", "投稿もなかなか続かなくて", "ああなるほど"]:
        segs.append({"chars": make_chars(text, t, t + len(text) * 0.15)})
        t += len(text) * 0.15 + 0.1
    sents = [text_of(s) for s in build_sentences(segs)]
    assert sents[0] == "こんにちは。"
    assert sents[1] == "今日はお時間いただいてありがとうございます。"
    assert sents[2] == "そうなんですよね。"
    assert any("いるんですけど、考えたことなかったです。" in x for x in sents)  # 区間の切れ目の「けど」には読点
    # Whisperの区間の切れ目でも、言い切りの後の相づちは別の文
    assert sents[-2].startswith("投稿もなかなか続かなくて")
    assert sents[-1].startswith("ああ")


def test_filler_without_comma():
    assert text_of(remove_fillers(make_chars("あの例えばですけど地域の", 0, 3), "中")) == "例えばですけど地域の"
    assert text_of(remove_fillers(make_chars("あの店に行った", 0, 3), "中")) == "あの店に行った"


def test_comma_is_preferred_and_hidden():
    (lines,) = _lines("最近はコロナの影響もあって、やっぱりお客様の数が減ってしまって")
    assert lines[0] == "最近はコロナの影響もあって"


@pytest.mark.parametrize("lines_setting", [1, 2])
def test_layout_invariants(lines_setting):
    st = LayoutSettings(max_chars_per_line=16, lines=lines_setting, block_max_chars=40)
    text = (
        "はい、例えば旅館の名前と温泉女子旅とか一人旅とか、ターゲットに合わせたキーワードを"
        "組み合わせると拡散しやすいですし、SNSでの反応も全然違ってくると思います。"
    )
    blocks = split_sentence(make_chars(text, 0, 30), st)
    shown = "".join("".join(b["lines"]) for b in blocks)
    assert shown == text.replace("、", "").replace("。", "")
    for b in blocks:
        assert len(b["lines"]) <= lines_setting
        assert all(display_len(l) <= 16 for l in b["lines"])


def test_fix_timing_and_srt():
    blocks = [
        {"text": "こんにちは", "start": 0.0, "end": 0.3},
        {"text": "今日は", "start": 1.0, "end": 1.5},
    ]
    fixed = fix_timing(blocks)
    assert fixed[0]["end"] == pytest.approx(1.0)  # 1.5秒欲しいが次の開始で止める
    srt = to_srt(fixed)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n\n2\n")


# ---------------- pipeline ----------------

class FakeGemini:
    def __init__(self):
        self.calls = 0
        self.models = self

    def generate_content(self, model, contents, config=None):
        self.calls += 1
        data = json.loads(contents[contents.index("[{"): contents.index("}]") + 2])
        out = [{"id": d["id"], "text": d["text"].replace("厳選", "源泉")} for d in data]

        class R:
            text = json.dumps(out, ensure_ascii=False)

        return R()


SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "えー、こんにちは。"},
    {"start": 2.0, "end": 9.0, "text": "うちはやっぱり厳選かけ流しの温泉が一番の自慢なんですけど、最近は予約が少ないんです。"},
]


def _fake_transcriber(calls):
    def f(pipe):
        calls.append("t")
        return SEGMENTS

    return f


def _fake_aligner(calls):
    def f(pipe, segs):
        calls.append("a")
        return [{**s, "chars": make_chars(s["text"], s["start"], s["end"]), "aligned": True} for s in segs]

    return f


def test_pipeline_reuses_stages(tmp_path):
    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"fake audio")
    logs = []
    pipe = Pipeline(str(audio), work_root=str(tmp_path / "work"), log=logs.append)
    calls = []
    gemini = FakeGemini()
    kw = dict(transcriber=_fake_transcriber(calls), aligner=_fake_aligner(calls), gemini_client=gemini,
              output_dir=str(tmp_path))

    s = Settings(user_dictionary="")
    r1 = pipe.run(s, **kw)
    assert r1.reran == ["transcribe", "align", "cleanup", "proofread", "layout"]
    assert "源泉かけ流し" in r1.srt_text  # Gemini校正が反映
    assert "えー" not in r1.srt_text
    assert gemini.calls == 1

    # 行数だけ変更 → 分割・改行だけやり直し
    s.lines = 1
    r2 = pipe.run(s, **kw)
    assert r2.reran == ["layout"]
    assert calls == ["t", "a"]
    assert gemini.calls == 1
    assert all(len(b.split("\n")) == 3 for b in r2.srt_text.strip().split("\n\n"))

    # フィラー設定を変更 → 3以降をやり直し。Gemini は保存済みの結果を再利用
    s.filler_level = "強"
    r3 = pipe.run(s, **kw)
    assert r3.reran == ["cleanup", "proofread", "layout"]
    assert gemini.calls <= 2

    # 最初からやり直し
    r4 = pipe.run(s, force=True, **kw)
    assert r4.reran[0] == "transcribe"
    assert calls == ["t", "a", "t", "a"]


def test_ctc_align_long_segment():
    # 64文字以上の区間で桁あふれして均等割りに落ちていた不具合の再発防止
    rng = np.random.default_rng(0)
    T, N, V = 1500, 180, 300
    lp = rng.standard_normal((T, V))
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    tokens = list(rng.integers(1, V, N))
    spans = ctc_align(lp, tokens, 0)
    assert spans is not None and len(spans) == N
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_filler_after_leading_space():
    # 「…どうですか? あ、それは」のように文頭に空白が残っていても「あ、」を消す
    chars = make_chars(" あ、それはいいかもしれないですね。", 0, 5)
    assert text_of(remove_fillers(chars, "中")) == "それはいいかもしれないですね。"


def test_fillers_removed_before_sentence_inference():
    # 「えっと」の途中に句読点が入ってフィラーが消えなくなる不具合の再発防止
    from telopauto.sentences import build_sentences

    f = lambda piece: remove_fillers(piece, "中", ["えっと", "ええと"])
    sents = [text_of(f(s)) for s in build_sentences([{"chars": make_chars("こんにちはえっと今日はお時間いただいて", 0, 5)}], prefilter=f)]
    assert "".join(sents).replace("。", "").replace("、", "") == "こんにちは今日はお時間いただいて"


def test_short_gap_between_sentences_is_closed():
    blocks = [
        {"text": "はい", "start": 10.0, "end": 10.3},
        {"text": "こちらこそありがとうございます", "start": 11.5, "end": 13.0},
        {"text": "長い沈黙の後", "start": 20.0, "end": 21.0},
    ]
    fixed = fix_timing(blocks)
    assert fixed[0]["end"] == pytest.approx(11.5)  # 1秒未満の隙間はつなぐ(最低表示0.8秒→10.8、隙間0.7秒)
    assert fixed[1]["end"] < 20.0  # 長い沈黙では消える


def test_filler_marks_speaker_change():
    # 「…ありますし あ でも…」の「あ」は話者交代の目印。消す前に文の切れ目として使う
    from telopauto.layout import layout
    from telopauto.sentences import build_sentences

    f = lambda piece: remove_fillers(piece, "中")
    text = "やっぱりお客様のリアルな声が一番説得力ありますしあでもそういうのお願いするのって失礼じゃないですかね"
    sents = build_sentences([{"chars": make_chars(text, 0, 10)}], prefilter=f)
    assert [text_of(s) for s in sents][0].endswith("ありますし。")
    blocks = [b["text"] for b in layout(sents, LayoutSettings(20, 1, 20))]
    assert blocks == ["やっぱりお客様のリアルな声が", "一番説得力ありますし", "でもそういうのお願いするのって", "失礼じゃないですかね"]
