#!/usr/bin/env python3
"""
Build lightweight, 100% open-source frontend pronunciation lexicons:
  1. speech/en-lexicon.json — Top ~16,000 English words mapped via real US-English
     IPA + primary stress (`*`) into exact `en.json` syllable keys.
     Source: Kokoro Misaki/CMUdict `lexicon-us-en.txt` (Apache-2.0 / BSD) + wordfreq.
  2. speech/ja-lexicon.json — High-frequency Japanese Kanji words (~15,000+) and
     single-Kanji fallback readings (~3,500+) mapped to Hiragana.
     Source: WorksApplications SudachiDict Core (Apache-2.0) + wordfreq.

Usage:
  python scripts/build-lexicons.py
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from typing import Dict, List, Set, Tuple

import sudachipy
import wordfreq

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
SPEECH_DIR = ROOT_DIR / "speech"
TTS_ROOT = pathlib.Path(os.environ.get("KOKORO_TTS_ROOT", str(ROOT_DIR / "models" / "tts")))
EN_IPA_PATH = TTS_ROOT / "kokoro-multi-lang-v1_1" / "lexicon-us-en.txt"

sys.path.insert(0, str(ROOT_DIR / "scripts"))
import importlib.util

_spec = importlib.util.spec_from_file_location("bvb", ROOT_DIR / "scripts" / "build-voice-bank.py")
_bvb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bvb)


# ---------- 1. English IPA -> en.json syllable key mapper ----------
EN_VOWELS = {
    "æ": "a", "ɑ": "aa", "ɒ": "aa", "ɔ": "o", "ɛ": "e", "ɪ": "i", "ᵻ": "i",
    "i": "ee", "ʌ": "u", "ə": "u", "ᵊ": "u", "u": "oo", "ʊ": "oo",
    "I": "ai", "A": "ei", "O": "ou", "W": "au", "Y": "oi", "ɜ": "er",
}

EN_CONSONANTS = {
    "b": "b", "d": "d", "ɾ": "t", "f": "f", "ɡ": "g", "g": "g", "h": "h",
    "ʤ": "j", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "ɹ": "r",
    "r": "r", "s": "s", "t": "t", "v": "v", "w": "w", "j": "y", "z": "z",
    "ʃ": "sh", "ʧ": "ch", "θ": "th", "ð": "dh", "ʒ": "sh", "ŋ": "n",
}

CLUSTERS = {
    "st", "tr", "kr", "pr", "br", "dr", "fr", "gr", "pl", "fl", "kl", "bl", "sl", "sp", "sk",
    "by", "ky", "my", "ny", "fy", "vy",
}


def ipa_to_en_syllables(ipa_str: str, en_keys: Set[str]) -> List[str]:
    """Split a space-separated Kokoro US-English IPA string into `en.json` keys with `*` stress."""
    tokens = ipa_str.split()
    out: List[str] = []
    onset_buf: List[str] = []
    stressed = False

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "ˈ":
            stressed = True
            i += 1
            continue
        if tok in ("ˌ", "ː", "ʔ"):
            i += 1
            continue

        if tok in EN_VOWELS:
            v_key = EN_VOWELS[tok]
            # Handle r-colored vowel (ɑɹ, ɔɹ, ɜɹ) or -ing (ɪŋ)
            if v_key == "er" and i + 1 < len(tokens) and tokens[i + 1] == "ɹ":
                i += 1
            elif v_key == "aa" and i + 1 < len(tokens) and tokens[i + 1] == "ɹ" and not onset_buf:
                v_key = "ar"
                i += 1
            elif v_key == "o" and i + 1 < len(tokens) and tokens[i + 1] == "ɹ" and not onset_buf:
                v_key = "or"
                i += 1
            elif v_key == "i" and i + 1 < len(tokens) and tokens[i + 1] == "ŋ":
                # Emit onset+i or pure ing
                if not onset_buf:
                    out.append("ing*" if stressed else "ing")
                    stressed = False
                    i += 2
                    continue

            # Pick best onset cluster or final consonant immediately preceding this vowel
            chosen = None
            if len(onset_buf) >= 2:
                pair = onset_buf[-2] + onset_buf[-1]
                if pair in CLUSTERS:
                    cands = [pair + v_key, pair + ("u" if v_key == "oo" else v_key[0])]
                    for c in cands:
                        if c in en_keys:
                            chosen = c
                            break
            if not chosen and onset_buf:
                # Keep th-cluster crisp (e.g. "through" θ ɹ u -> thoo instead of roo)
                c1 = "th" if len(onset_buf) >= 2 and onset_buf[-2] == "th" and onset_buf[-1] == "r" else onset_buf[-1]
                # Prefer exact onset+vowel, then graceful diphthong/vowel fallback
                cands = [
                    c1 + v_key,
                    c1 + ("o" if v_key == "aa" else "e" if v_key == "er" else "a" if v_key == "au" else "o" if v_key == "oi" else v_key),
                    c1 + "a",
                ]
                for c in cands:
                    if c in en_keys:
                        chosen = c
                        break
            if not chosen:
                chosen = v_key if v_key in en_keys else "a"

            # Merge trailing ŋ as -ing if single syllable ended on ɪŋ
            if tok == "ɪ" and i + 1 < len(tokens) and tokens[i + 1] == "ŋ" and i + 2 == len(tokens):
                # Keep crisp consonant+i plus optional ing feel
                pass

            out.append(f"{chosen}*" if stressed else chosen)
            onset_buf.clear()
            stressed = False
            i += 1
            continue

        if tok in EN_CONSONANTS:
            onset_buf.append(EN_CONSONANTS[tok])
        i += 1

    # Fallback for vowel-less abbreviations (e.g. "shh", "mr")
    if not out and onset_buf:
        c = onset_buf[0]
        out.append(c if c in en_keys else "a")

    return out


def build_en_lexicon() -> None:
    en_keys = set(_bvb.build_en_phonemes().keys())

    ipa_dict: Dict[str, str] = {}
    for line in EN_IPA_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            ipa_dict[parts[0].lower()] = parts[1]

    # Top 18,000 everyday English words + extra game/audio/greeting terms
    top_words = wordfreq.top_n_list("en", 20000)
    extra_words = [
        "animalese", "crossing", "villager", "villagers", "isabelle", "nook", "kk",
        "island", "hello", "welcome", "kitty", "meow", "purr", "synth", "formant",
        "prosody", "melody", "pitch", "audio", "studio", "guitar", "butterfly",
        "butterflies", "coffee", "espresso", "cookie", "cookies", "nintendo", "switch",
        "pixel", "retro", "github", "typescript", "javascript", "sandbox", "preview",
    ]

    result: Dict[str, str] = {}
    word_re = re.compile(r"^[a-z]{2,20}$")

    for w in list(top_words) + extra_words:
        wl = w.lower()
        if not word_re.match(wl) or wl in result:
            continue
        ipa = ipa_dict.get(wl)
        if not ipa:
            continue
        syls = ipa_to_en_syllables(ipa, en_keys)
        if syls:
            result[wl] = " ".join(syls)
        if len(result) >= 16000:
            break

    # Add curated manual entries for coined / game terms not in CMUdict
    result.setdefault("animalese", "a* ni ma lee")
    result.setdefault("xjbcat", "e* ji bi ka")
    result.setdefault("clarance", "kla* ra")
    result.setdefault("kokoro", "ko* ko rou")
    result.setdefault("i", "ai*")
    result.setdefault("a", "a")

    out_path = SPEECH_DIR / "en-lexicon.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024.0
    print(f"[OK] en-lexicon.json: {len(result)} words | {size_kb:.1f} KB")
    for sample in ("hello", "island", "welcome", "animal", "crossing", "voice", "beautiful", "knight", "through", "coffee", "animalese"):
        print(f"     {sample:12s} -> {result.get(sample)}")


# ---------- 2. Japanese SudachiDict (Apache-2.0) -> Kanji Words & Chars Hiragana Lexicon ----------
KANJI_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF々]")
KATA_TO_HIRA = str.maketrans(
    "".join(chr(c) for c in range(0x30A1, 0x30F7)),
    "".join(chr(c) for c in range(0x3041, 0x3097)),
)
VALID_HIRA_RE = re.compile(r"^[ぁ-んー]+$")


def kata_to_hira(s: str) -> str:
    return s.translate(KATA_TO_HIRA)


def build_ja_lexicon() -> None:
    tok = sudachipy.Dictionary().tokenizer()
    mode_c = sudachipy.SplitMode.C

    words: Dict[str, str] = {}
    chars: Dict[str, str] = {}

    # Seed with common Japanese conversational words & phrases (including okurigana forms)
    curated_words = {
        "今日": "きょう", "明日": "あした", "昨日": "きのう", "今": "いま", "天気": "てんき",
        "可愛い": "かわいい", "かわいい": "かわいい", "猫": "ねこ", "子猫": "こねこ", "小猫": "こねこ",
        "島": "しま", "海": "うみ", "海風": "うみかぜ", "風": "かぜ", "花": "はな", "星": "ほし",
        "夜": "よる", "朝": "あさ", "昼": "ひる", "夕方": "ゆうがた", "一緒": "いっしょ",
        "遊ぶ": "あそぶ", "遊ぼう": "あそぼう", "遊んで": "あそんで", "遊び": "あそび",
        "日本語": "にほんご", "日本": "にほん", "英語": "えいご", "中国語": "ちゅうごくご",
        "大丈夫": "だいじょうぶ", "本当": "ほんとう", "好き": "すき", "大好き": "だいすき",
        "元気": "げんき", "気持ち": "きもち", "楽しい": "たのしい", "嬉しい": "うれしい",
        "美味しい": "おいしい", "綺麗": "きれい", "素敵": "すてき", "最高": "さいこう",
        "友達": "ともだち", "音楽": "おんがく", "歌": "うた", "声": "こえ", "言葉": "ことば",
        "話": "はなし", "話す": "はなす", "時間": "じかん", "世界": "せかい", "森": "もり",
        "動物": "どうぶつ", "どうぶつの森": "どうぶつのもり", "虫": "むし", "蝶": "ちょう",
        "魚": "さかな", "釣る": "つる", "釣り": "つり", "木": "き", "雨": "あめ", "雪": "ゆき",
        "空": "そら", "雲": "くも", "太陽": "たいよう", "月": "つき", "光": "ひかり",
        "家": "いえ", "部屋": "へや", "学校": "がっこう", "仕事": "しごと", "休み": "やすみ",
        "毎日": "まいにち", "毎週": "まいしゅう", "今年": "ことし", "来年": "らいねん",
        "名前": "なまえ", "私": "わたし", "僕": "ぼく", "君": "きみ", "人": "ひと",
        "皆": "みんな", "本当": "ほんとう", "一番": "いちばん", "少し": "すこし",
        "全然": "ぜんぜん", "色々": "いろいろ", "時々": "ときどき", "大人": "おとな",
        "子供": "こども", "先生": "せんせい", "学生": "がくせい", "勉強": "べんきょう",
        "開発": "かいはつ", "実験": "じっけん", "音声": "おんせい", "合成": "ごうせい",
        "再生": "さいせい", "停止": "ていし", "設定": "せってい", "画面": "がめん",
        "一人": "ひとり", "二人": "ふたり", "一日": "いちにち", "二日": "ふつか",
        "行こう": "いこう", "行く": "いく", "来る": "くる", "見る": "みる", "聞く": "きく",
        "食べる": "たべる", "飲む": "飲む", "作る": "つくる", "思う": "おもう", "知る": "知る",
    }
    # Fix typo in two curated readings above
    curated_words["飲む"] = "のむ"
    curated_words["知る"] = "しる"
    words.update(curated_words)

    # Pull top 45,000 Japanese tokens from wordfreq and resolve readings via Apache-2.0 SudachiDict
    top_ja = wordfreq.top_n_list("ja", 55000)
    for w in top_ja:
        if len(w) > 6 or not KANJI_RE.search(w):
            continue
        # Only keep pure Japanese words (Kanji + optional Kana)
        if not re.match(r"^[\u4E00-\u9FFF\u3400-\u4DBF々ぁ-んァ-ヶー]+$", w):
            continue
        morphemes = tok.tokenize(w, mode_c)
        reading = "".join(kata_to_hira(m.reading_form()) for m in morphemes)
        if not reading or not VALID_HIRA_RE.match(reading):
            continue
        if len(w) == 1:
            chars.setdefault(w, reading)
        elif len(words) < 12000:
            words.setdefault(w, reading)
        if len(words) >= 12000 and len(chars) >= 2200:
            break

    # Populate single-Kanji fallback table across all common CJK unified ideographs known to SudachiDict
    for code in range(0x4E00, 0x9FFF + 1):
        ch = chr(code)
        if ch in chars:
            continue
        ms = tok.tokenize(ch, mode_c)
        if len(ms) == 1 and ms[0].is_oov() is False:
            r = kata_to_hira(ms[0].reading_form())
            if r and VALID_HIRA_RE.match(r) and len(r) <= 5:
                chars[ch] = r
        if len(chars) >= 4200:
            break

    # Ensure single-char curated readings override rare readings
    for k, v in curated_words.items():
        if len(k) == 1:
            chars[k] = v

    payload = {"words": words, "chars": chars}
    out_path = SPEECH_DIR / "ja-lexicon.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024.0
    print(f"[OK] ja-lexicon.json: {len(words)} words + {len(chars)} chars | {size_kb:.1f} KB")
    for s in ("今日", "天気", "可愛い", "猫", "島", "遊ぼう", "日本語", "一緒", "海風", "大丈夫"):
        print(f"     {s:6s} -> {words.get(s) or chars.get(s)}")


if __name__ == "__main__":
    build_en_lexicon()
    build_ja_lexicon()
