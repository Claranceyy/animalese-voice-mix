#!/usr/bin/env python3
"""
Build 100% open-source (Apache-2.0 Kokoro) pitch-flattened & time-compressed
trilingual Animal-Crossing syllable sprite banks:
  - zh.ogg: Kokoro-82M v1.1-zh (zf_001) — 407 Mandarin syllables
  - en.ogg: Kokoro-82M v1.0 (af_heart)  — ~240 native English CV + diphthong + cluster syllables
  - ja.ogg: Kokoro-82M v1.0 (jf_alpha)  — 106 native Japanese morae (Gojuon + Dakuten + Yoon)

All banks are pitch-flattened to an identical 364 Hz and loudness-equalized via
single-pass Praat PSOLA (PitchTier + DurationTier) so zh / en / ja blend seamlessly.

Usage:
  python scripts/build-voice-bank.py --langs en ja
  python scripts/build-voice-bank.py --langs zh en ja
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import tempfile
import time
from typing import Dict, List, Tuple

import numpy as np
import parselmouth
from parselmouth.praat import call as praat_call
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, resample_poly, sosfiltfilt
import soundfile as sf

TARGET_HZ = 364.0
TARGET_SR = 48000
GAP_SEC = 0.060
TARGET_RMS = 0.175
PEAK_LIMIT = 0.78

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
SPEECH_DIR = ROOT_DIR / "speech"
DEFAULT_TTS_ROOT = pathlib.Path(os.environ.get("KOKORO_TTS_ROOT", str(ROOT_DIR / "models" / "tts")))


# ---------- sherpa-onnx C-API ctypes binding for Kokoro v1.1-zh (Mandarin) ----------
class VitsConfig(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p), ("lexicon", ctypes.c_char_p), ("tokens", ctypes.c_char_p),
        ("data_dir", ctypes.c_char_p), ("noise_scale", ctypes.c_float), ("noise_scale_w", ctypes.c_float),
        ("length_scale", ctypes.c_float), ("dict_dir", ctypes.c_char_p),
    ]


class MatchaConfig(ctypes.Structure):
    _fields_ = [
        ("acoustic_model", ctypes.c_char_p), ("vocoder", ctypes.c_char_p), ("lexicon", ctypes.c_char_p),
        ("tokens", ctypes.c_char_p), ("data_dir", ctypes.c_char_p), ("noise_scale", ctypes.c_float),
        ("length_scale", ctypes.c_float), ("dict_dir", ctypes.c_char_p),
    ]


class KokoroConfig(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p), ("voices", ctypes.c_char_p), ("tokens", ctypes.c_char_p),
        ("data_dir", ctypes.c_char_p), ("length_scale", ctypes.c_float), ("dict_dir", ctypes.c_char_p),
        ("lexicon", ctypes.c_char_p), ("lang", ctypes.c_char_p),
    ]


class KittenConfig(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p), ("voices", ctypes.c_char_p), ("tokens", ctypes.c_char_p),
        ("data_dir", ctypes.c_char_p), ("length_scale", ctypes.c_float),
    ]


class ZipVoiceConfig(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.c_char_p), ("encoder", ctypes.c_char_p), ("decoder", ctypes.c_char_p),
        ("vocoder", ctypes.c_char_p), ("data_dir", ctypes.c_char_p), ("lexicon", ctypes.c_char_p),
        ("feat_scale", ctypes.c_float), ("t_shift", ctypes.c_float), ("target_rms", ctypes.c_float),
        ("guidance_scale", ctypes.c_float),
    ]


class PocketConfig(ctypes.Structure):
    _fields_ = [
        ("lm_flow", ctypes.c_char_p), ("lm_main", ctypes.c_char_p), ("encoder", ctypes.c_char_p),
        ("decoder", ctypes.c_char_p), ("text_conditioner", ctypes.c_char_p), ("vocab_json", ctypes.c_char_p),
        ("token_scores_json", ctypes.c_char_p), ("voice_embedding_cache_capacity", ctypes.c_int),
    ]


class SupertonicConfig(ctypes.Structure):
    _fields_ = [
        ("duration_predictor", ctypes.c_char_p), ("text_encoder", ctypes.c_char_p),
        ("vector_estimator", ctypes.c_char_p), ("vocoder", ctypes.c_char_p),
        ("tts_json", ctypes.c_char_p), ("unicode_indexer", ctypes.c_char_p), ("voice_style", ctypes.c_char_p),
    ]


class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("vits", VitsConfig), ("num_threads", ctypes.c_int), ("debug", ctypes.c_int),
        ("provider", ctypes.c_char_p), ("matcha", MatchaConfig), ("kokoro", KokoroConfig),
        ("kitten", KittenConfig), ("zipvoice", ZipVoiceConfig), ("pocket", PocketConfig),
        ("supertonic", SupertonicConfig),
    ]


class TtsConfig(ctypes.Structure):
    _fields_ = [
        ("model", ModelConfig), ("rule_fsts", ctypes.c_char_p), ("max_num_sentences", ctypes.c_int),
        ("rule_fars", ctypes.c_char_p), ("silence_scale", ctypes.c_float),
    ]


class GenAudio(ctypes.Structure):
    _fields_ = [("samples", ctypes.POINTER(ctypes.c_float)), ("n", ctypes.c_int), ("sample_rate", ctypes.c_int)]


class KokoroZhRunner:
    def __init__(self, tts_root: pathlib.Path, exact_lexicon: pathlib.Path):
        lib_dir = self._find_lib_dir(tts_root)
        model_dir = tts_root / "kokoro-multi-lang-v1_1"
        if not (model_dir / "model.onnx").exists():
            raise FileNotFoundError(f"Kokoro v1.1-zh model.onnx not found in {model_dir}")

        if os.name == "nt":
            os.add_dll_directory(str(lib_dir))
            for dll in ("onnxruntime_providers_shared.dll", "onnxruntime.dll"):
                p = lib_dir / dll
                if p.exists():
                    ctypes.CDLL(str(p))
            self.lib = ctypes.CDLL(str(lib_dir / "sherpa-onnx-c-api.dll"))
        else:
            self.lib = ctypes.CDLL(str(lib_dir / "libsherpa-onnx-c-api.so"))

        self.lib.SherpaOnnxCreateOfflineTts.argtypes = [ctypes.POINTER(TtsConfig)]
        self.lib.SherpaOnnxCreateOfflineTts.restype = ctypes.c_void_p
        self.lib.SherpaOnnxOfflineTtsSampleRate.argtypes = [ctypes.c_void_p]
        self.lib.SherpaOnnxOfflineTtsSampleRate.restype = ctypes.c_int
        self.lib.SherpaOnnxOfflineTtsGenerate.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_float
        ]
        self.lib.SherpaOnnxOfflineTtsGenerate.restype = ctypes.POINTER(GenAudio)
        self.lib.SherpaOnnxDestroyOfflineTtsGeneratedAudio.argtypes = [ctypes.POINTER(GenAudio)]
        self.lib.SherpaOnnxDestroyOfflineTts.argtypes = [ctypes.c_void_p]

        cfg = TtsConfig()
        cfg.model.vits = VitsConfig(b"", b"", b"", b"", 0.667, 0.8, 1.0, b"")
        cfg.model.matcha = MatchaConfig(b"", b"", b"", b"", b"", 0.667, 1.0, b"")
        cfg.model.kokoro = KokoroConfig(
            str(model_dir / "model.onnx").encode(),
            str(model_dir / "voices.bin").encode(),
            str(model_dir / "tokens.txt").encode(),
            str(model_dir / "espeak-ng-data").encode(),
            1.0,
            str(model_dir / "dict").encode(),
            str(exact_lexicon).encode(),
            b"",
        )
        cfg.model.kitten = KittenConfig(b"", b"", b"", b"", 1.0)
        cfg.model.zipvoice = ZipVoiceConfig(b"", b"", b"", b"", b"", b"", 0.1, 0.5, 0.1, 1.0)
        cfg.model.pocket = PocketConfig(b"", b"", b"", b"", b"", b"", b"", 50)
        cfg.model.supertonic = SupertonicConfig(b"", b"", b"", b"", b"", b"", b"")
        cfg.model.num_threads = max(2, min(6, (os.cpu_count() or 4) // 2))
        cfg.model.debug = 0
        cfg.model.provider = b"cpu"
        cfg.rule_fsts = b""
        cfg.max_num_sentences = 1
        cfg.rule_fars = b""
        cfg.silence_scale = 0.15

        self.handle = self.lib.SherpaOnnxCreateOfflineTts(ctypes.byref(cfg))
        if not self.handle:
            raise RuntimeError("SherpaOnnxCreateOfflineTts failed")
        self.sample_rate = int(self.lib.SherpaOnnxOfflineTtsSampleRate(self.handle))

    @staticmethod
    def _find_lib_dir(tts_root: pathlib.Path) -> pathlib.Path:
        for child in sorted(tts_root.glob("sherpa-onnx-v1.13*"), reverse=True):
            sub = child / "lib"
            if (sub / "sherpa-onnx-c-api.dll").exists() or (sub / "libsherpa-onnx-c-api.so").exists():
                return sub
            if (child / "sherpa-onnx-c-api.dll").exists():
                return child
        raise FileNotFoundError(f"sherpa-onnx runtime library not found under {tts_root}")

    def synth(self, slot_char: str, sid: int, speed: float = 1.18) -> Tuple[np.ndarray, int]:
        ptr = self.lib.SherpaOnnxOfflineTtsGenerate(self.handle, slot_char.encode("utf-8"), sid, speed)
        if not ptr or ptr.contents.n <= 0:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        arr = np.ctypeslib.as_array(ptr.contents.samples, shape=(ptr.contents.n,)).copy()
        sr = int(ptr.contents.sample_rate)
        self.lib.SherpaOnnxDestroyOfflineTtsGeneratedAudio(ptr)
        return arr, sr

    def close(self) -> None:
        if self.handle:
            self.lib.SherpaOnnxDestroyOfflineTts(self.handle)
            self.handle = None


# ---------- Audio DSP: Active Trim + Single-Pass Praat PSOLA (Pitch Flatten + Duration Compress) ----------
HP_SOS = butter(2, 95.0, btype="highpass", fs=TARGET_SR, output="sos")


def process_syllable(raw: np.ndarray, sr: int, unit_dur_sec: float = 0.115, target_hz: float = TARGET_HZ) -> np.ndarray:
    target_samples = int(round(unit_dur_sec * TARGET_SR))
    if raw.size < 200:
        return np.zeros(target_samples, dtype=np.float32)

    audio = raw.astype(np.float64)
    if sr != TARGET_SR:
        g = int(np.gcd(sr, TARGET_SR))
        audio = resample_poly(audio, TARGET_SR // g, sr // g)

    audio = sosfiltfilt(HP_SOS, audio)

    win = int(0.006 * TARGET_SR)
    kernel = np.ones(win, dtype=np.float64) / win
    env = np.sqrt(np.maximum(0.0, np.convolve(audio * audio, kernel, mode="same")))
    peak_env = float(np.max(env)) if env.size else 0.0
    if peak_env <= 1e-6:
        return np.zeros(target_samples, dtype=np.float32)

    onset_idx = np.where(env >= max(0.005, 0.06 * peak_env))[0]
    offset_idx = np.where(env >= max(0.008, 0.12 * peak_env))[0]
    start = max(0, int(onset_idx[0]) - int(0.010 * TARGET_SR)) if onset_idx.size else 0
    end = min(audio.size, int(offset_idx[-1]) + int(0.014 * TARGET_SR)) if offset_idx.size else audio.size
    active = audio[start:end]
    if active.size < int(0.028 * TARGET_SR):
        active = audio

    active_dur = active.size / TARGET_SR

    # Single-pass Praat PSOLA: flatten pitch to 364 Hz + smoothly compress full syllable into unit_dur_sec
    try:
        snd = parselmouth.Sound(active, sampling_frequency=TARGET_SR)
        manip = praat_call(snd, "To Manipulation", 0.006, 90, 620)

        pitch_tier = praat_call("Create PitchTier", "flat", 0.0, snd.duration)
        praat_call(pitch_tier, "Add point", 0.0, float(target_hz))
        praat_call(pitch_tier, "Add point", float(snd.duration), float(target_hz))
        praat_call([manip, pitch_tier], "Replace pitch tier")

        if active_dur > unit_dur_sec * 1.04:
            ratio = max(0.42, (unit_dur_sec * 0.97) / active_dur)
            dur_tier = praat_call("Create DurationTier", "compress", 0.0, snd.duration)
            praat_call(dur_tier, "Add point", 0.0, float(ratio))
            praat_call(dur_tier, "Add point", float(snd.duration), float(ratio))
            praat_call([manip, dur_tier], "Replace duration tier")

        resynth = praat_call(manip, "Get resynthesis (overlap-add)")
        proc = np.asarray(resynth.values[0], dtype=np.float64)
        if proc.size > 0 and np.isfinite(proc).all():
            active = proc
    except Exception:
        pass

    env2 = np.sqrt(np.maximum(0.0, np.convolve(active * active, kernel, mode="same")))
    p2 = float(np.max(env2)) if env2.size else 0.0
    if p2 > 1e-6:
        nz = np.where(env2 >= 0.05 * p2)[0]
        if nz.size:
            s2 = max(0, int(nz[0]) - int(0.004 * TARGET_SR))
            active = active[s2:]

    if active.size >= target_samples:
        unit = active[:target_samples].copy()
    else:
        unit = np.zeros(target_samples, dtype=np.float64)
        unit[: active.size] = active

    fade_in = min(int(0.0025 * TARGET_SR), unit.size // 6)
    fade_out = min(int(0.016 * TARGET_SR), unit.size // 3)
    if fade_in > 1:
        unit[:fade_in] *= 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_in)))
    if fade_out > 1:
        unit[-fade_out:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_out)))

    rms = float(np.sqrt(np.mean(unit * unit)))
    if rms > 1e-6:
        unit *= TARGET_RMS / rms
    unit = clean_envelope_limit(unit, ceiling=0.70, sr=TARGET_SR)

    return unit.astype(np.float32)


def clean_envelope_limit(unit: np.ndarray, ceiling: float = 0.70, sr: int = TARGET_SR) -> np.ndarray:
    """
    Distortion-free lookahead envelope limiter replacing sample-by-sample `np.tanh`.
    Smoothly attenuates transient spikes over a 5 ms window so waveforms never clip or buzz.
    """
    pk = float(np.max(np.abs(unit))) if unit.size else 0.0
    if pk <= ceiling:
        return unit
    win = max(5, int(0.005 * sr) | 1)
    env_pk = maximum_filter1d(np.abs(unit), size=win, mode="nearest")
    raw_gain = np.minimum(1.0, ceiling / np.maximum(ceiling, env_pk))
    hann = np.hanning(win)
    hann /= np.sum(hann)
    smooth_gain = np.convolve(raw_gain, hann, mode="same")
    out = unit * smooth_gain
    rem_pk = float(np.max(np.abs(out)))
    if rem_pk > ceiling:
        out *= ceiling / rem_pk
    return out


def pack_sprite(
    units_audio: List[Tuple[str, np.ndarray]],
    out_ogg: pathlib.Path,
    out_json: pathlib.Path,
    meta: Dict[str, str],
) -> None:
    gap = np.zeros(int(round(GAP_SEC * TARGET_SR)), dtype=np.float32)
    chunks: List[np.ndarray] = [gap]
    cursor = gap.size
    manifest_units: Dict[str, List[float]] = {}

    for key, wav in sorted(units_audio, key=lambda x: x[0]):
        off = round(cursor / TARGET_SR, 4)
        dur = round(wav.size / TARGET_SR, 4)
        manifest_units[key] = [off, dur]
        chunks.append(wav)
        chunks.append(gap)
        cursor += wav.size + gap.size

    full = np.concatenate(chunks)
    out_ogg.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_ogg), full, TARGET_SR, format="OGG", subtype="OPUS")

    manifest = {
        "hz": int(TARGET_HZ),
        "source": meta["source"],
        "license": meta["license"],
        "voice": meta["voice"],
        "units": manifest_units,
    }
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    rb, rsr = sf.read(str(out_ogg))
    f0s = []
    for key in list(manifest_units.keys())[:8]:
        off, dur = manifest_units[key]
        seg = rb[int(off * rsr) : int((off + dur) * rsr)]
        p = parselmouth.Sound(seg, sampling_frequency=rsr).to_pitch(pitch_floor=100, pitch_ceiling=600)
        v = p.selected_array["frequency"]
        v = v[v > 0]
        if v.size:
            f0s.append(float(np.median(v)))
    med_f0 = float(np.median(f0s)) if f0s else 0.0
    size_kb = out_ogg.stat().st_size / 1024.0
    print(
        f"[OK] {out_ogg.name}: {len(manifest_units)} units | {size_kb:.1f} KB | "
        f"dur={rb.size / rsr:.1f}s | median F0={med_f0:.1f} Hz"
    )


# ---------- Mandarin (zh) 407 Syllables ----------
def build_zh_phonemes(model_dir: pathlib.Path) -> List[Tuple[str, str]]:
    zh_lexicon_path = SPEECH_DIR / "zh-lexicon.json"
    raw_lex: Dict[str, str] = json.loads(zh_lexicon_path.read_text(encoding="utf-8"))

    kokoro_zh_chars: Dict[str, str] = {}
    for line in (model_dir / "lexicon-zh.txt").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2 and len(parts[0]) == 1:
            kokoro_zh_chars[parts[0]] = parts[1]

    by_syl: Dict[str, Dict[int, str]] = {}
    for k, chars in raw_lex.items():
        if len(k) < 2:
            continue
        tone = int(k[-1]) if k[-1].isdigit() else 1
        syl = k[:-1] if k[-1].isdigit() else k
        by_syl.setdefault(syl, {})[tone] = chars

    manual_bopo: Dict[str, str] = {
        "fiao": "ㄈ 要 1", "den": "ㄉ ㄣ 1", "lo": "ㄌ ㄛ 1", "nun": "ㄋ ㄨ ㄣ 1",
        "rua": "ㄖ ㄨ ㄚ 1", "chua": "ㄔ ㄨ ㄚ 1", "nou": "ㄋ ㄡ 1", "yo": "ㄧ ㄛ 1",
    }

    items: List[Tuple[str, str]] = []
    for syl in sorted(by_syl.keys()):
        chosen_char = None
        for tone in (1, 2, 4, 3, 5):
            for ch in by_syl[syl].get(tone, ""):
                if ch in kokoro_zh_chars and 0x4E00 <= ord(ch) <= 0x9FFF:
                    if tone == 1 or kokoro_zh_chars[ch].endswith(" 1"):
                        chosen_char = ch
                        break
            if chosen_char:
                break
        if not chosen_char:
            for tone in (1, 2, 4, 3, 5):
                for ch in by_syl[syl].get(tone, ""):
                    if ch in kokoro_zh_chars:
                        chosen_char = ch
                        break
                if chosen_char:
                    break

        if chosen_char:
            parts = kokoro_zh_chars[chosen_char].split()
            if parts and parts[-1] in ("1", "2", "3", "4", "5"):
                parts[-1] = "1"
            items.append((syl, " ".join(parts)))
        elif syl in manual_bopo:
            items.append((syl, manual_bopo[syl]))
    return items


# ---------- English (en) ~250 Crisp Phonetic Syllables (Kokoro v1.0 af_heart) ----------
def build_en_phonemes() -> Dict[str, str]:
    """
    Rich English phonetic syllable inventory designed to map real US-English IPA words
    (from lexicon-us-en.txt) cleanly: pure vowels, ALL English diphthongs, crisp CV grid,
    digraphs (sh/ch/th/ng), and high-frequency onset clusters.
    """
    en_ipa: Dict[str, str] = {
        # Pure vowels & diphthongs (short, crisp, unstretched)
        "a": "æ", "aa": "ɑ", "e": "ɛ", "i": "ɪ", "ee": "i", "o": "ɔ", "oo": "u", "u": "ʌ",
        "ai": "I", "ei": "A", "ou": "O", "au": "W", "oi": "Y",
        "er": "ɜɹ", "ar": "ɑɹ", "or": "ɔɹ",
        "ing": "ɪŋ", "en": "ɛn", "in": "ɪn", "an": "æn", "on": "ɑn", "un": "ʌn",
        "ion": "ʃən",
        # Single-letter fallbacks
        "b": "bi", "c": "si", "d": "di", "f": "ɛf", "g": "ʤi", "h": "hA", "j": "ʤA",
        "k": "kA", "l": "ɛl", "m": "ɛm", "n": "ɛn", "p": "pi", "q": "kju", "r": "ɑɹ",
        "s": "ɛs", "t": "ti", "v": "vi", "w": "wʌ", "x": "ɛks", "y": "wI", "z": "zi",
        "sh": "ʃi", "ch": "ʧi", "th": "θi", "dh": "ðə", "zh": "ʒə",
    }

    onsets = {
        "b": "b", "d": "d", "f": "f", "g": "ɡ", "h": "h", "j": "ʤ",
        "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "r": "ɹ",
        "s": "s", "t": "t", "v": "v", "w": "w", "y": "j", "z": "z",
        "sh": "ʃ", "ch": "ʧ", "th": "θ", "dh": "ð",
    }
    # 10 core English vowel/diphthong nuclei: a(æ), aa(ɑ), e(ɛ), i(ɪ), ee(i), o(ɔ), ou(O), u(ʌ), oo(u), ai(I), ei(A)
    core_vowels = {
        "a": "æ", "aa": "ɑ", "e": "ɛ", "i": "ɪ", "ee": "i",
        "o": "ɔ", "ou": "O", "u": "ʌ", "oo": "u", "ai": "I", "ei": "A",
    }
    for c_key, c_ipa in onsets.items():
        for v_key, v_ipa in core_vowels.items():
            en_ipa[f"{c_key}{v_key}"] = f"{c_ipa}{v_ipa}"

    # High-frequency diphthong W(ow), Y(oy), er(ɜɹ) + common English clusters
    extra_combos = {
        "bau": "bW", "dau": "dW", "hau": "hW", "kau": "kW", "lau": "lW", "mau": "mW",
        "nau": "nW", "pau": "pW", "rau": "ɹW", "sau": "sW", "tau": "tW", "wau": "wW",
        "boi": "bY", "joi": "ʤY", "koi": "kY", "loi": "lY", "noi": "nY", "poi": "pY",
        "roi": "ɹY", "soi": "sY", "toi": "tY", "voi": "vY",
        "ber": "bɜɹ", "der": "dɜɹ", "fer": "fɜɹ", "ger": "ɡɜɹ", "her": "hɜɹ", "ker": "kɜɹ",
        "ler": "lɜɹ", "mer": "mɜɹ", "ner": "nɜɹ", "per": "pɜɹ", "ser": "sɜɹ", "ter": "tɜɹ",
        "ver": "vɜɹ", "wer": "wɜɹ", "sher": "ʃɜɹ", "cher": "ʧɜɹ", "ther": "ðɜɹ",
        # Crisp consonant clusters (including long-vowel ee / ai / oo & sw/sm/sn/tw/kw)
        "sta": "stæ", "ste": "stɛ", "sti": "stɪ", "stee": "sti", "sto": "stɑ", "stou": "stO", "stu": "stu", "stai": "stI", "star": "stɑɹ",
        "tra": "tɹæ", "tre": "tɹɛ", "tri": "tɹɪ", "tree": "tɹi", "tro": "tɹɑ", "trou": "tɹO", "tru": "tɹu", "troo": "tɹu", "trai": "tɹI",
        "kra": "kɹæ", "kre": "kɹɛ", "kri": "kɹɪ", "kree": "kɹi", "kro": "kɹɔ", "krou": "kɹO", "krai": "kɹI",
        "pra": "pɹæ", "pre": "pɹɛ", "pri": "pɹɪ", "pree": "pɹi", "pro": "pɹO", "prai": "pɹI",
        "bra": "bɹæ", "bre": "bɹɛ", "bri": "bɹɪ", "bree": "bɹi", "bro": "bɹO", "broo": "bɹu", "brai": "bɹI",
        "dra": "dɹæ", "dre": "dɹɛ", "dri": "dɹɪ", "dree": "dɹi", "dro": "dɹɑ", "drai": "dɹI",
        "fra": "fɹæ", "fre": "fɹɛ", "fri": "fɹi", "free": "fɹi", "fro": "fɹɑ", "froo": "fɹu", "frai": "fɹI",
        "gra": "ɡɹæ", "gre": "ɡɹɛ", "gri": "ɡɹi", "gree": "ɡɹi", "gro": "ɡɹO", "groo": "ɡɹu",
        "pla": "plæ", "ple": "plA", "pli": "pli", "plo": "plɑ", "plai": "plI",
        "fla": "flæ", "fle": "flɛ", "fli": "flɪ", "flee": "fli", "flo": "flO", "flai": "flI",
        "kla": "klæ", "kle": "klɛ", "kli": "klɪ", "klo": "klO", "klai": "klI",
        "bla": "blæ", "ble": "blɛ", "bli": "blɪ", "blo": "blO", "bloo": "blu",
        "sla": "slæ", "sle": "slɛ", "sli": "slɪ", "slee": "sli", "slo": "slO",
        "spa": "spɑ", "spe": "spɛ", "spi": "spi", "spo": "spO", "spri": "spɹɪ",
        "ska": "skɑ", "ske": "skɛ", "ski": "ski", "sko": "skO", "skai": "skI",
        "swa": "swæ", "swe": "swɛ", "swi": "swɪ", "swee": "swi", "swo": "swɔ", "swai": "swI",
        "sma": "smɑ", "sme": "smɛ", "smi": "smɪ", "smai": "smI", "smo": "smɔ",
        "sna": "snæ", "sne": "snɛ", "sni": "snɪ", "snou": "snO",
        "twa": "twɑ", "twe": "twɛ", "twi": "twɪ",
        "kwa": "kwɑ", "kwe": "kwɛ", "kwi": "kwɪ", "kwai": "kwI",
        "three": "θɹi", "thoo": "θɹu",
        "byu": "bju", "kyu": "kju", "myu": "mju", "nyu": "nju", "fyu": "fju", "vyu": "vju",
        # R-colored CV syllables (fixes morning, market, star, park, shore, warm...)
        "bar": "bɑɹ", "dar": "dɑɹ", "far": "fɑɹ", "gar": "ɡɑɹ", "har": "hɑɹ",
        "kar": "kɑɹ", "lar": "lɑɹ", "mar": "mɑɹ", "par": "pɑɹ", "sar": "sɑɹ", "tar": "tɑɹ",
        "bor": "bɔɹ", "dor": "dɔɹ", "for": "fɔɹ", "gor": "ɡɔɹ", "hor": "hɔɹ",
        "kor": "kɔɹ", "lor": "lɔɹ", "mor": "mɔɹ", "nor": "nɔɹ", "por": "pɔɹ",
        "sor": "sɔɹ", "tor": "tɔɹ", "wor": "wɔɹ", "shor": "ʃɔɹ",
        # Syllabic -l / -n / -m & common closed rimes (fixes welcome, island, apple, beetle, ocean, some)
        "ul": "əl", "bul": "bəl", "pul": "pəl", "tul": "təl", "dul": "dəl",
        "kul": "kəl", "gul": "ɡəl", "ful": "fəl", "mul": "məl", "sul": "səl", "zul": "zəl",
        "wel": "wɛl", "fel": "fɛl", "tel": "tɛl", "sel": "sɛl", "hel": "hɛl", "bel": "bɛl",
        "ol": "ɔl", "kol": "kɔl", "bol": "bɔl", "fol": "fɔl",
        "shun": "ʃən", "zhun": "ʒən", "chun": "ʧən", "lun": "lən", "run": "ɹən", "mun": "mən", "nun": "nən",
        "kum": "kʌm", "sum": "sʌm", "hum": "hʌm",
        # Soft word-final English coda micro-units (`_` prefix, tucked tightly onto preceding vowel)
        "_s": "s", "_z": "z", "_sh": "ʃ", "_ch": "ʧ", "_j": "ʤ",
        "_f": "f", "_v": "v", "_t": "t", "_d": "d", "_k": "k", "_p": "p",
        "_m": "m", "_n": "n", "_ng": "ŋ", "_l": "l", "_ks": "ks", "_ts": "ts",
    }
    en_ipa.update(extra_combos)
    return dict(sorted(en_ipa.items()))


# ---------- Japanese (ja) 106 Genuine Native Morae (Kokoro v1.0 jf_alpha) ----------
def build_ja_phonemes() -> Dict[str, str]:
    """Genuine native Japanese IPA morae trained directly in Kokoro v1.0 (jf_alpha)."""
    ja_ipa: Dict[str, str] = {
        # Pure 5 Japanese monophthongs (`ɑ` guarantees open Japanese あ without English schwa drift)
        "a": "ɑ", "i": "i", "u": "ɯ", "e": "e", "o": "o",
        # K
        "ka": "ka", "ki": "ki", "ku": "kɯ", "ke": "ke", "ko": "ko",
        # G
        "ga": "ɡa", "gi": "ɡi", "gu": "ɡɯ", "ge": "ɡe", "go": "ɡo",
        # S
        "sa": "sa", "shi": "ɕi", "su": "sɯ", "se": "se", "so": "so",
        # Z / J
        "za": "za", "ji": "ʥi", "zu": "ʣɯ", "ze": "ze", "zo": "zo",
        # T
        "ta": "ta", "chi": "ʨi", "tsu": "ʦɯ", "te": "te", "to": "to",
        # D
        "da": "da", "de": "de", "do": "do",
        # N
        "na": "na", "ni": "ɲi", "nu": "nɯ", "ne": "ne", "no": "no",
        # H / F
        "ha": "ha", "hi": "çi", "fu": "ɸɯ", "he": "he", "ho": "ho",
        # B
        "ba": "ba", "bi": "bi", "bu": "bɯ", "be": "be", "bo": "bo",
        # P
        "pa": "pa", "pi": "pi", "pu": "pɯ", "pe": "pe", "po": "po",
        # M
        "ma": "ma", "mi": "mi", "mu": "mɯ", "me": "me", "mo": "mo",
        # Y
        "ya": "ja", "yu": "jɯ", "yo": "jo",
        # R (Japanese alveolar flap ɾ)
        "ra": "ɾa", "ri": "ɾi", "ru": "ɾɯ", "re": "ɾe", "ro": "ɾo",
        # W & Moraic N
        "wa": "wa", "wo": "o", "n": "ɴ",
        # Yoon (palatalized morae using Kokoro v1.0 native ʲ palatalization for crisp voicing)
        "kya": "kʲa", "kyu": "kʲɯ", "kyo": "kʲo",
        "gya": "ɡʲa", "gyu": "ɡʲɯ", "gyo": "ɡʲo",
        "sha": "ɕa", "shu": "ɕɯ", "sho": "ɕo",
        "ja": "ʥa", "ju": "ʥɯ", "jo": "ʥo",
        "cha": "ʨa", "chu": "ʨɯ", "cho": "ʨo",
        "nya": "ɲa", "nyu": "ɲɯ", "nyo": "ɲo",
        "hya": "ça", "hyu": "çɯ", "hyo": "ço",
        "bya": "bʲa", "byu": "bʲɯ", "byo": "bʲo",
        "pya": "pʲa", "pyu": "pʲɯ", "pyo": "pʲo",
        "mya": "mʲa", "myu": "mʲɯ", "myo": "mʲo",
        "rya": "ɾʲa", "ryu": "ɾʲɯ", "ryo": "ɾʲo",
        # Extended Katakana loanword morae (ファ/フィ/フェ/フォ/ティ/ディ/トゥ/ドゥ/シェ/チェ/ジェ/ウィ/ウェ)
        "fa": "ɸa", "fi": "ɸi", "fe": "ɸe", "fo": "ɸo",
        "ti": "ti", "di": "di", "tu": "tɯ", "du": "dɯ",
        "she": "ɕe", "che": "ʨe", "je": "ʥe",
        "wi": "wi", "we": "we",
    }
    return dict(sorted(ja_ipa.items()))


def extract_ja_carrier_mora(k: object, v_style: np.ndarray, ph: str) -> Tuple[np.ndarray, int]:
    """
    Synthesize a Japanese mora inside an utterance-initial carrier followed by a
    geminate voiceless stop (`{ph}ː ʔtote.`) using a mid-sentence style vector (`v_style[18]`).

    Why this matters:
    - Passing isolated 1-2 phoneme strings into Kokoro hits degenerate `v_style[0..1]`
      vectors and triggers Japanese high-vowel devoicing (`き/し/ち/す/つ/ひ/ふ`), collapsing
      vowel formants toward a central schwa.
    - Lengthening `{ph}ː` before the mandatory silent closure of `ʔt` guarantees
      100% clear Japanese vowel formants AND a clean zero-energy right boundary.
    """
    tok = "ɴː" if ph == "ɴ" else f"{ph}ː"
    carrier = f"{tok} ʔtote."
    tokens = k.tokenizer.tokenize(carrier)
    style_row = v_style[min(18, len(v_style) - 1)]
    raw, _ = k._infer(tokens, style_row, 1.0)
    sr = 24000

    w = raw.astype(np.float64)
    win = int(0.006 * sr)
    env = np.sqrt(np.maximum(0.0, np.convolve(w * w, np.ones(win) / win, mode="same")))
    first_half = env[: max(win, int(0.58 * len(env)))]
    pk1 = float(np.max(first_half)) if first_half.size else 0.0
    if pk1 <= 1e-6:
        return raw, sr

    # 1. Locate main vowel nucleus peak of `{tok}`, then capture full initial consonant burst/frication
    vowel_hits = np.where(first_half >= 0.30 * pk1)[0]
    v_peak_start = int(vowel_hits[0]) if vowel_hits.size else int(np.argmax(first_half))
    pre_hits = np.where(env[: v_peak_start + 1] >= max(0.0025, 0.020 * pk1))[0]
    start_idx = max(0, int(pre_hits[0]) - int(0.008 * sr)) if pre_hits.size else max(0, v_peak_start - int(0.020 * sr))

    # 2. Scan forward after vowel establishment for the silent `ʔt` geminate closure (>= 28 ms quiet valley)
    search_from = min(len(env) - 1, v_peak_start + int(0.055 * sr))
    min_gap = int(0.028 * sr)
    quiet = env[search_from:] < (0.038 * pk1)
    end_idx = min(len(w), v_peak_start + int(0.28 * sr))
    run_len = 0
    for off, is_q in enumerate(quiet):
        if is_q:
            run_len += 1
            if run_len >= min_gap:
                gap_start = search_from + off - run_len + 1
                end_idx = min(len(w), gap_start + int(0.008 * sr))
                break
        else:
            run_len = 0

    return raw[start_idx:end_idx].copy(), sr


def process_ja_mora(raw: np.ndarray, sr: int, unit_dur_sec: float = 0.112, target_hz: float = TARGET_HZ) -> np.ndarray:
    """
    Two-zone consonant-preserving PSOLA normalizer tailored for Japanese Doubutsugo:
    - Preserves crisp initial consonant articulation (`18-28 ms`) without excessive sibilant hiss;
    - Dedicates a full `80-90 ms` steady voiced vowel core flattened to `364 Hz` with zero hard truncation.
    """
    target_samples = int(round(unit_dur_sec * TARGET_SR))
    if raw.size < 200:
        return np.zeros(target_samples, dtype=np.float32)

    audio = raw.astype(np.float64)
    if sr != TARGET_SR:
        g = int(np.gcd(sr, TARGET_SR))
        audio = resample_poly(audio, TARGET_SR // g, sr // g)
    audio = sosfiltfilt(HP_SOS, audio)

    win = int(0.005 * TARGET_SR)
    kernel = np.ones(win, dtype=np.float64) / win
    env = np.sqrt(np.maximum(0.0, np.convolve(audio * audio, kernel, mode="same")))
    pk = float(np.max(env)) if env.size else 0.0
    if pk <= 1e-6:
        return np.zeros(target_samples, dtype=np.float32)

    nz = np.where(env >= max(0.0025, 0.022 * pk))[0]
    nz_tail = np.where(env >= max(0.005, 0.08 * pk))[0]
    s0 = max(0, int(nz[0]) - int(0.006 * TARGET_SR)) if nz.size else 0
    e0 = min(audio.size, int(nz_tail[-1]) + int(0.012 * TARGET_SR)) if nz_tail.size else audio.size
    active = audio[s0:e0].copy()
    if active.size < int(0.040 * TARGET_SR):
        active = audio.copy()

    total_budget = unit_dur_sec * 0.965
    voiced_start_out = 0.018

    try:
        snd = parselmouth.Sound(active, sampling_frequency=TARGET_SR)
        dur = float(snd.duration)
        pit = snd.to_pitch(time_step=0.004, pitch_floor=100.0, pitch_ceiling=620.0)
        freqs = pit.selected_array["frequency"]
        xs = pit.xs()
        v_idx = np.where(freqs > 0)[0]
        v_start = float(xs[v_idx[0]]) if v_idx.size else min(0.025, dur * 0.2)
        v_start = max(0.0, min(dur - 0.035, v_start))

        # Gently tame overly loud voiceless sibilant hiss relative to vowel nucleus before PSOLA
        if v_start > 0.018:
            c_end = int(v_start * TARGET_SR)
            c_rms = float(np.sqrt(np.mean(active[:c_end] ** 2))) if c_end > 0 else 0.0
            v_rms = float(np.sqrt(np.mean(active[c_end:] ** 2))) if c_end < active.size else c_rms
            if v_rms > 1e-6 and c_rms > 0.58 * v_rms:
                active[:c_end] *= (0.58 * v_rms) / c_rms
                snd = parselmouth.Sound(active, sampling_frequency=TARGET_SR)

        manip = praat_call(snd, "To Manipulation", 0.005, 95, 620)
        pitch_tier = praat_call("Create PitchTier", "flat", 0.0, dur)
        praat_call(pitch_tier, "Add point", 0.0, float(target_hz))
        praat_call(pitch_tier, "Add point", dur, float(target_hz))
        praat_call([manip, pitch_tier], "Replace pitch tier")

        dur_tier = praat_call("Create DurationTier", "twozone", 0.0, dur)
        if v_start >= 0.014 and (dur - v_start) >= 0.040:
            target_c = min(0.027, max(0.016, v_start * 0.46))
            target_v = max(0.068, total_budget - target_c)
            ratio_c = max(0.24, min(1.05, target_c / v_start))
            ratio_v = max(0.22, min(1.05, target_v / (dur - v_start)))
            t_mid1 = max(0.002, v_start - 0.003)
            t_mid2 = min(dur - 0.002, v_start + 0.003)
            praat_call(dur_tier, "Add point", 0.0, float(ratio_c))
            praat_call(dur_tier, "Add point", float(t_mid1), float(ratio_c))
            praat_call(dur_tier, "Add point", float(t_mid2), float(ratio_v))
            praat_call(dur_tier, "Add point", dur, float(ratio_v))
            voiced_start_out = target_c
        else:
            ratio_all = max(0.24, min(1.05, total_budget / max(0.04, dur)))
            praat_call(dur_tier, "Add point", 0.0, float(ratio_all))
            praat_call(dur_tier, "Add point", dur, float(ratio_all))
            voiced_start_out = v_start * ratio_all

        praat_call([manip, dur_tier], "Replace duration tier")
        resynth = praat_call(manip, "Get resynthesis (overlap-add)")
        proc = np.asarray(resynth.values[0], dtype=np.float64)
        if proc.size > 0 and np.isfinite(proc).all():
            active = proc
    except Exception:
        pass

    env2 = np.sqrt(np.maximum(0.0, np.convolve(active * active, kernel, mode="same")))
    p2 = float(np.max(env2)) if env2.size else 0.0
    if p2 > 1e-6:
        nz2 = np.where(env2 >= 0.03 * p2)[0]
        if nz2.size:
            s2 = max(0, int(nz2[0]) - int(0.003 * TARGET_SR))
            active = active[s2:]

    if active.size >= target_samples:
        unit = active[:target_samples].copy()
    else:
        unit = np.zeros(target_samples, dtype=np.float64)
        unit[: active.size] = active

    fade_in = min(int(0.0022 * TARGET_SR), unit.size // 8)
    fade_out = min(int(0.015 * TARGET_SR), unit.size // 4)
    if fade_in > 1:
        unit[:fade_in] *= 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_in)))
    if fade_out > 1:
        unit[-fade_out:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_out)))

    # Equalize loudness primarily on the voiced vowel core so every mora projects equally clearly
    v_s = min(unit.size // 2, max(0, int(voiced_start_out * TARGET_SR)))
    v_e = max(v_s + int(0.030 * TARGET_SR), unit.size - fade_out)
    vowel_slice = unit[v_s:v_e]
    rms = float(np.sqrt(np.mean(vowel_slice * vowel_slice))) if vowel_slice.size else 0.0
    if rms <= 1e-6:
        rms = float(np.sqrt(np.mean(unit * unit)))
    if rms > 1e-6:
        unit *= (TARGET_RMS * 1.02) / rms
    unit = clean_envelope_limit(unit, ceiling=0.68, sr=TARGET_SR)

    return unit.astype(np.float32)


EN_VOWEL_CHARS = set("æɑɛɪiɔuʌIAOWYəɜ")


def lengthen_en_ipa(ph: str) -> str:
    """Insert `ː` right after the primary vowel nucleus so carrier synthesis yields a clear vowel body."""
    for idx in range(len(ph) - 1, -1, -1):
        if ph[idx] in EN_VOWEL_CHARS:
            return ph[: idx + 1] + "ː" + ph[idx + 1 :]
    return ph + "ː"


def build_en_coda_unit(k: object, v_style: np.ndarray, key: str, ph: str, unit_dur_sec: float = 0.064) -> np.ndarray:
    """
    Synthesize a soft, non-intrusive English word-final coda micro-unit (`_s`, `_k`, `_t`, `_sh`, `_m`...):
    - Sonorant codas (`_m`, `_n`, `_ng`, `_l`) keep a gentle 364Hz voiced murmur;
    - Obstruent codas (`_s`, `_z`, `_sh`, `_ch`, `_t`, `_k`, `_ks`, `_ts`...) extract the crisp post-vowel
      release from `"hˈɑ{ph}."` with smooth cosine edges and zero waveshaping distortion.
    """
    target_samples = int(round(unit_dur_sec * TARGET_SR))
    sr = 24000
    style_row = v_style[min(18, len(v_style) - 1)]

    if key in ("_m", "_n", "_ng", "_l"):
        tokens = k.tokenizer.tokenize(f"ˈəː{ph}ː ʔtote.")
        raw, _ = k._infer(tokens, style_row, 1.0)
        w = raw.astype(np.float64)
        win = int(0.006 * sr)
        env = np.sqrt(np.maximum(0.0, np.convolve(w * w, np.ones(win) / win, mode="same")))
        pk1 = float(np.max(env[: int(0.58 * len(env))])) if env.size else 0.0
        vh = np.where(env[: int(0.58 * len(env))] >= 0.25 * pk1)[0]
        vs = int(vh[0]) if vh.size else 0
        q_s = min(len(env) - win, vs + int(0.08 * sr))
        q_e = min(len(env), vs + int(0.38 * sr))
        q_w = int(0.022 * sr)
        e_idx = q_s + int(np.argmin(np.convolve(env[q_s:q_e], np.ones(q_w) / q_w, mode="valid"))) if q_e - q_s > q_w else q_e
        s_idx = max(vs, e_idx - int(0.11 * sr))
        tail = raw[s_idx:e_idx].copy()
        unit = process_ja_mora(tail, sr, unit_dur_sec=unit_dur_sec, target_hz=TARGET_HZ).astype(np.float64)
        unit *= 0.54
        return clean_envelope_limit(unit, ceiling=0.44, sr=TARGET_SR).astype(np.float32)

    tokens = k.tokenizer.tokenize(f"hˈɑ{ph}.")
    raw, _ = k._infer(tokens, style_row, 1.0)
    audio = resample_poly(raw.astype(np.float64), TARGET_SR // 24000, 1)
    audio = sosfiltfilt(HP_SOS, audio)

    snd = parselmouth.Sound(audio, sampling_frequency=TARGET_SR)
    pit = snd.to_pitch(time_step=0.004, pitch_floor=100.0, pitch_ceiling=550.0)
    freqs = pit.selected_array["frequency"]
    xs = pit.xs()
    v_idx = np.where(freqs > 0)[0]
    coda_start_sec = float(xs[v_idx[-1]]) - 0.006 if v_idx.size else snd.duration * 0.45
    s_i = max(0, int(coda_start_sec * TARGET_SR))

    win = int(0.004 * TARGET_SR)
    env = np.sqrt(np.maximum(0.0, np.convolve(audio * audio, np.ones(win) / win, mode="same")))
    sub_env = env[s_i:]
    pk = float(np.max(sub_env)) if sub_env.size else 0.0
    nz = np.where(sub_env >= max(0.0015, 0.04 * pk))[0]
    if nz.size:
        seg = audio[s_i + max(0, int(nz[0]) - int(0.004 * TARGET_SR)) : s_i + int(nz[-1]) + int(0.008 * TARGET_SR)].copy()
    else:
        seg = audio[s_i : s_i + target_samples].copy()

    if seg.size > target_samples:
        # Keep onset transition + crisp release without exceeding budget
        idx_grid = np.linspace(0, seg.size - 1, target_samples)
        unit = np.interp(idx_grid, np.arange(seg.size), seg)
    else:
        unit = np.zeros(target_samples, dtype=np.float64)
        unit[: seg.size] = seg

    fade_in = min(int(0.006 * TARGET_SR), unit.size // 4)
    fade_out = min(int(0.016 * TARGET_SR), unit.size // 3)
    if fade_in > 1:
        unit[:fade_in] *= 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_in)))
    if fade_out > 1:
        unit[-fade_out:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_out)))

    rms = float(np.sqrt(np.mean(unit * unit)))
    if rms > 1e-6:
        unit *= (TARGET_RMS * 0.36) / rms
    return clean_envelope_limit(unit, ceiling=0.38, sr=TARGET_SR).astype(np.float32)


def extract_en_carrier_syllable(k: object, v_style: np.ndarray, ph: str) -> Tuple[np.ndarray, int]:
    """Synthesize an English syllable inside `'ˈ{phː} ʔtote.'` using mid-sentence style vector `v_style[20]`."""
    tok = lengthen_en_ipa(ph)
    carrier = f"ˈ{tok} ʔtote."
    tokens = k.tokenizer.tokenize(carrier)
    style_row = v_style[min(20, len(v_style) - 1)]
    raw, _ = k._infer(tokens, style_row, 1.0)
    sr = 24000

    w = raw.astype(np.float64)
    win = int(0.006 * sr)
    env = np.sqrt(np.maximum(0.0, np.convolve(w * w, np.ones(win) / win, mode="same")))
    first_half = env[: max(win, int(0.58 * len(env)))]
    pk1 = float(np.max(first_half)) if first_half.size else 0.0
    if pk1 <= 1e-6:
        return raw, sr

    vowel_hits = np.where(first_half >= 0.28 * pk1)[0]
    v_peak_start = int(vowel_hits[0]) if vowel_hits.size else int(np.argmax(first_half))
    pre_hits = np.where(env[: v_peak_start + 1] >= max(0.002, 0.018 * pk1))[0]
    start_idx = max(0, int(pre_hits[0]) - int(0.008 * sr)) if pre_hits.size else max(0, v_peak_start - int(0.018 * sr))

    q_start = min(len(env) - win, v_peak_start + int(0.065 * sr))
    q_end = min(len(env), v_peak_start + int(0.35 * sr))
    q_win = int(0.024 * sr)
    if q_end - q_start > q_win:
        sm = np.convolve(env[q_start:q_end], np.ones(q_win) / q_win, mode="valid")
        end_idx = min(len(w), q_start + int(np.argmin(sm)) + int(0.006 * sr))
    else:
        end_idx = min(len(w), v_peak_start + int(0.24 * sr))

    return raw[start_idx:end_idx].copy(), sr


def build_v1_0_bank(
    lang: str,
    phonemes_map: Dict[str, str],
    voice: str,
    unit_dur: float,
    speed: float,
    tts_root: pathlib.Path,
) -> None:
    import kokoro_onnx

    v1_dir = tts_root / "kokoro-v1_0"
    model_path = v1_dir / "kokoro-v1.0.int8.onnx"
    voices_path = v1_dir / "voices-v1.0.bin"
    if not model_path.exists() or not voices_path.exists():
        raise FileNotFoundError(f"Kokoro v1.0 files not found in {v1_dir}")

    k = kokoro_onnx.Kokoro(str(model_path), str(voices_path))
    v_style = k.get_voice_style(voice)
    vocab = k.tokenizer.vocab
    for key, ph in phonemes_map.items():
        for ch in ph:
            if ch not in vocab:
                raise ValueError(f"Unknown Kokoro v1.0 phoneme char {ch!r} in {lang}:{key} ({ph})")

    print(f"\n=== Building {lang}.ogg ({len(phonemes_map)} native units, Kokoro v1.0 voice={voice}) ===")
    built: List[Tuple[str, np.ndarray]] = []
    t0 = time.perf_counter()
    items = list(phonemes_map.items())
    for idx, (key, ph) in enumerate(items, 1):
        if lang == "ja":
            wav, sr = extract_ja_carrier_mora(k, v_style, ph)
            unit_wav = process_ja_mora(wav, sr, unit_dur_sec=unit_dur, target_hz=TARGET_HZ)
        elif lang == "en":
            if key.startswith("_"):
                unit_wav = build_en_coda_unit(k, v_style, key, ph, unit_dur_sec=0.064)
            else:
                wav, sr = extract_en_carrier_syllable(k, v_style, ph)
                unit_wav = process_ja_mora(wav, sr, unit_dur_sec=unit_dur, target_hz=TARGET_HZ)
        else:
            wav, sr = k.create(ph, voice=voice, speed=speed, is_phonemes=True)
            unit_wav = process_syllable(wav, sr, unit_dur_sec=unit_dur, target_hz=TARGET_HZ)
        built.append((key, unit_wav))
        if idx % 60 == 0 or idx == len(items):
            print(f"  [{lang}] {idx}/{len(items)} ({time.perf_counter() - t0:.1f}s)")

    pack_sprite(
        built,
        SPEECH_DIR / f"{lang}.ogg",
        SPEECH_DIR / f"{lang}.json",
        meta={
            "source": "Kokoro-82M v1.0 (hexgrad/Kokoro-82M)",
            "license": "Apache-2.0",
            "voice": voice,
        },
    )


def build_zh_bank(tts_root: pathlib.Path, zh_sid: int) -> None:
    model_dir = tts_root / "kokoro-multi-lang-v1_1"
    valid_tokens = set()
    for line in (model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            valid_tokens.add(parts[0])

    items = build_zh_phonemes(model_dir)
    lex_lines: List[str] = []
    slots: List[Tuple[str, str]] = []
    for idx, (syl_key, phoneme_str) in enumerate(items):
        for tok in phoneme_str.split():
            if tok not in valid_tokens:
                raise ValueError(f"Unknown Kokoro v1.1-zh token {tok!r} for zh:{syl_key}")
        slot_char = chr(0x4E00 + idx)
        lex_lines.append(f"{slot_char} {phoneme_str}")
        slots.append((syl_key, slot_char))

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write("\n".join(lex_lines) + "\n")
        exact_lex = pathlib.Path(tmp.name)

    runner = KokoroZhRunner(tts_root, exact_lexicon=exact_lex)
    try:
        print(f"\n=== Building zh.ogg ({len(slots)} Mandarin units, Kokoro v1.1-zh zf_001) ===")
        built: List[Tuple[str, np.ndarray]] = []
        t0 = time.perf_counter()
        for idx, (syl_key, slot_char) in enumerate(slots, 1):
            raw, sr = runner.synth(slot_char, sid=zh_sid, speed=1.18)
            unit_wav = process_syllable(raw, sr, unit_dur_sec=0.115, target_hz=TARGET_HZ)
            built.append((syl_key, unit_wav))
            if idx % 80 == 0 or idx == len(slots):
                print(f"  [zh] {idx}/{len(slots)} ({time.perf_counter() - t0:.1f}s)")

        pack_sprite(
            built,
            SPEECH_DIR / "zh.ogg",
            SPEECH_DIR / "zh.json",
            meta={
                "source": "Kokoro-82M-v1.1 (hexgrad/Kokoro-82M-v1.1-zh)",
                "license": "Apache-2.0",
                "voice": "zf_001",
            },
        )
    finally:
        runner.close()
        exact_lex.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Apache-2.0 Kokoro trilingual Animalese sprite banks")
    parser.add_argument("--tts-root", type=pathlib.Path, default=DEFAULT_TTS_ROOT)
    parser.add_argument("--langs", nargs="+", default=["en", "ja"], choices=["zh", "en", "ja"])
    parser.add_argument("--zh-sid", type=int, default=3)
    parser.add_argument("--en-voice", type=str, default="af_heart")
    parser.add_argument("--ja-voice", type=str, default="jf_alpha")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if "zh" in args.langs:
        build_zh_bank(args.tts_root, args.zh_sid)
    if "en" in args.langs:
        build_v1_0_bank("en", build_en_phonemes(), args.en_voice, unit_dur=0.105, speed=1.32, tts_root=args.tts_root)
    if "ja" in args.langs:
        build_v1_0_bank("ja", build_ja_phonemes(), args.ja_voice, unit_dur=0.112, speed=1.0, tts_root=args.tts_root)

    print(f"\nAll done in {time.perf_counter() - t0:.1f}s!")


if __name__ == "__main__":
    main()
