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
    "ʃ": "sh", "ʧ": "ch", "θ": "th", "ð": "dh", "ʒ": "sh", "ŋ": "ng",
}

CLUSTERS = {
    "st", "tr", "kr", "pr", "br", "dr", "fr", "gr", "pl", "fl", "kl", "bl", "sl", "sp", "sk",
    "sw", "sm", "sn", "tw", "kw", "th", "spr",
    "by", "ky", "my", "ny", "fy", "vy",
}


def ipa_to_en_syllables(ipa_str: str, en_keys: Set[str]) -> List[str]:
    """
    Split a space-separated Kokoro US-English IPA string into `en.json` keys with `*` stress,
    preserving r-colored nuclei, syllabic `-l/-n/-m` rimes, and word-final coda micro-units (`_s`, `_k`, `_t`...).
    """
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
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            nxt2 = tokens[i + 2] if i + 2 < len(tokens) else ""
            nxt_is_coda = nxt2 == "" or (nxt2 in EN_CONSONANTS and nxt2 != "ɹ")

            # 1. R-colored vowels (ɜɹ, ɑɹ, ɔɹ) both with and without onset consonants
            if v_key == "er" and nxt == "ɹ":
                i += 1
            elif v_key == "aa" and nxt == "ɹ" and nxt_is_coda:
                v_key = "ar"
                i += 1
            elif v_key == "o" and nxt == "ɹ" and nxt_is_coda:
                v_key = "or"
                i += 1
            # 2. Closed syllabic / unstressed rimes (`əl -> ul`, `ən -> un`, `ʌm -> um`, `ɛl -> el`, `ɔl -> ol`)
            elif tok in ("ə", "ᵊ") and nxt == "l" and nxt_is_coda:
                v_key = "ul"
                i += 1
            elif tok in ("ə", "ᵊ") and nxt == "n" and nxt_is_coda and onset_buf and (onset_buf[-1] + "un") in en_keys:
                v_key = "un"
                i += 1
            elif tok in ("ʌ", "ə") and nxt == "m" and nxt_is_coda and onset_buf and (onset_buf[-1] + "um") in en_keys:
                v_key = "um"
                i += 1
            elif tok == "ɛ" and nxt == "l" and nxt_is_coda and onset_buf and (onset_buf[-1] + "el") in en_keys:
                v_key = "el"
                i += 1
            elif tok == "ɔ" and nxt == "l" and nxt_is_coda and ((onset_buf and (onset_buf[-1] + "ol") in en_keys) or not onset_buf):
                v_key = "ol"
                i += 1
            elif v_key == "i" and nxt == "ŋ" and not onset_buf:
                out.append("ing*" if stressed else "ing")
                stressed = False
                i += 2
                continue
            elif not onset_buf and nxt in ("n", "ŋ") and nxt2 == "":
                fused = {"a": "an", "e": "en", "i": "ing" if nxt == "ŋ" else "in", "aa": "on", "o": "on", "u": "un"}.get(v_key)
                if fused and fused in en_keys:
                    out.append(f"{fused}*" if stressed else fused)
                    stressed = False
                    i += 2
                    continue

            # Gentle unstressed `hə` -> `he` (so "hello" sounds like "heh-low" rather than "huh-low")
            if tok == "ə" and onset_buf and onset_buf[-1] == "h":
                v_key = "e"

            # Pick best onset cluster or immediately preceding consonant
            chosen = None
            if len(onset_buf) >= 3:
                tri = onset_buf[-3] + onset_buf[-2] + onset_buf[-1]
                if tri in CLUSTERS and (tri + v_key) in en_keys:
                    chosen = tri + v_key
            if not chosen and len(onset_buf) >= 2:
                pair = onset_buf[-2] + onset_buf[-1]
                if pair == "th" and v_key == "oo":
                    pass
                if onset_buf[-2] == "th" and onset_buf[-1] == "r":
                    for c in ("three" if v_key == "ee" else "thoo" if v_key == "oo" else f"th{v_key}",):
                        if c in en_keys:
                            chosen = c
                            break
                elif pair in CLUSTERS:
                    cands = [pair + v_key, pair + ("u" if v_key == "oo" else v_key[0])]
                    for c in cands:
                        if c in en_keys:
                            chosen = c
                            break
            if not chosen and onset_buf:
                c1 = onset_buf[-1]
                if c1 == "ng":
                    c1 = "n"
                cands = [
                    c1 + v_key,
                    c1 + ("o" if v_key in ("aa", "or", "oi", "ol") else "a" if v_key in ("ar", "au") else "e" if v_key in ("er", "el") else "u" if v_key in ("ul", "um", "un") else v_key),
                    c1 + "a",
                ]
                for c in cands:
                    if c in en_keys:
                        chosen = c
                        break
            if not chosen:
                chosen = v_key if v_key in en_keys else "a"

            out.append(f"{chosen}*" if stressed else chosen)
            onset_buf.clear()
            stressed = False
            i += 1
            continue

        if tok in EN_CONSONANTS:
            onset_buf.append(EN_CONSONANTS[tok])
        i += 1

    # Handle word-final English coda consonant(s) (`voice -> voi* _s`, `mix -> mi* _ks`, `catch -> ka* _ch`)
    if out and onset_buf:
        # Filter out post-vocalic `r` (already colored into vowel or soft in Animalese)
        coda = [c for c in onset_buf if c not in ("r", "w", "y", "h")]
        if coda:
            coda_key = None
            if len(coda) >= 2:
                tail2 = coda[-2] + coda[-1]
                if tail2 == "ks" and "_ks" in en_keys:
                    coda_key = "_ks"
                elif tail2 == "ts" and "_ts" in en_keys:
                    coda_key = "_ts"
            if not coda_key:
                last_c = coda[-1]
                mapped = {"dh": "_d", "th": "_s"}.get(last_c, f"_{last_c}")
                if mapped in en_keys:
                    coda_key = mapped
            if coda_key:
                out.append(coda_key)
    elif not out and onset_buf:
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
    result["animalese"] = "a* ni ma lee _z"
    result.setdefault("xjbcat", "e* ji bi ka _t")
    result.setdefault("clarance", "kla* run _s")
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

    godan_rows: Dict[str, Tuple[str, ...]] = {
        "う": ("わ", "い", "う", "え", "お", "っ"),
        "く": ("か", "き", "く", "け", "こ", "い"),
        "ぐ": ("が", "ぎ", "ぐ", "げ", "ご", "い"),
        "す": ("さ", "し", "す", "せ", "そ"),
        "つ": ("た", "ち", "つ", "て", "と", "っ"),
        "ぬ": ("な", "に", "ぬ", "ね", "の", "ん"),
        "ぶ": ("ば", "び", "ぶ", "べ", "ぼ", "ん"),
        "む": ("ま", "み", "む", "め", "も", "ん"),
        "る": ("ら", "り", "る", "れ", "ろ", "っ"),
    }

    def expand_inflections(surface: str, reading: str, pos0: str, conj_type: str) -> None:
        if len(surface) < 2 or len(reading) < 2 or surface[-1] != reading[-1]:
            return
        stem_s, stem_r, tail = surface[:-1], reading[:-1], surface[-1]
        if not KANJI_RE.search(stem_s):
            return
        if pos0 == "動詞":
            if "一段" in conj_type and tail == "る":
                words.setdefault(stem_s, stem_r)
            elif "五段" in conj_type and tail in godan_rows:
                endings = godan_rows[tail]
                for e in endings:
                    actual_e = "っ" if (surface.endswith("行く") or surface == "行く") and e == "い" and tail == "く" else e
                    words.setdefault(stem_s + actual_e, stem_r + actual_e)
        elif pos0 == "形容詞" and tail == "い":
            for e in ("い", "く", "か", "け", "さ"):
                words.setdefault(stem_s + e, stem_r + e)

    # Expand curated verbs & i-adjectives first so core conversational vocabulary is always covered
    for cw, cr in list(curated_words.items()):
        ms = tok.tokenize(cw, mode_c)
        if len(ms) == 1:
            pos = ms[0].part_of_speech()
            expand_inflections(cw, cr, pos[0], pos[4])

    # Pull top 55,000 Japanese tokens from wordfreq and resolve readings via Apache-2.0 SudachiDict
    top_ja = wordfreq.top_n_list("ja", 55000)
    inflected_budget = 0
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
        else:
            if len(words) < 16500:
                words.setdefault(w, reading)
            if inflected_budget < 900 and len(morphemes) == 1:
                pos = morphemes[0].part_of_speech()
                if pos[0] in ("動詞", "形容詞"):
                    expand_inflections(w, reading, pos[0], pos[4])
                    inflected_budget += 1
        if len(words) >= 16500 and chars and len(chars) >= 2200:
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
