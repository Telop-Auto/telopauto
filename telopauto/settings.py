"""設定値。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass
class Settings:
    # 字幕の形
    max_chars_per_line: int = 20
    lines: int = 2
    block_max_chars: int = 40

    # 文字起こし
    whisper_model: str = "large-v3"

    # フィラー・辞書
    filler_level: str = "中"
    custom_fillers: str = "あのー,えっと,ええと,なんか"
    user_dictionary: str = "プレミア=Premiere\nアイフォン=iPhone\n厳選かけ流し=源泉かけ流し"

    # Gemini
    use_gemini: bool = True
    gemini_model: str = "gemini-3.5-flash"
    gemini_assist_words: str = "iPhone,Premiere,Telop Design Lab"

    def normalized(self) -> "Settings":
        s = Settings(**asdict(self))
        s.max_chars_per_line = min(max(int(s.max_chars_per_line), 4), 60)
        s.lines = 1 if int(s.lines) == 1 else 2
        s.block_max_chars = min(max(int(s.block_max_chars), 4), 120)
        if s.filler_level not in ("弱", "中", "強"):
            s.filler_level = "中"
        return s

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Settings":
        data = json.loads(text)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
