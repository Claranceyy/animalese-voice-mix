import {
  isAsciiLetter,
  mapEnglishChunk,
  mapLatinChunk,
  tokenizeEnglishWord,
} from "./tokenizer/english";
import {
  KANA_DIGRAPH_MAP,
  KANA_JA_MAP,
  KANA_ZH_FALLBACK,
  assignJapanesePitchAccents,
  computeJapaneseClauseFlags,
  hiraganaToMorae,
  isJapaneseEParticle,
  isJapaneseEncliticParticle,
  isJapanesePoliteTail,
  isJapaneseWaParticle,
  moraVowel,
  normalizeKana,
} from "./tokenizer/japanese";
import {
  CJK_RE,
  DIGIT_EN_SYLS,
  DIGIT_JA_KANA,
  DIGIT_ZH_KEYS,
  DIGIT_ZH_TONES,
  PHRASE_MARKS,
  WEAK_PARTICLES,
  parseHanziLexicon,
  type RawToken,
  type TokenizerOptions,
} from "./tokenizer/shared";
import type { SpeechUnit } from "./types";

export { WEAK_PARTICLES, PHRASE_MARKS, parseHanziLexicon, mapEnglishChunk, mapLatinChunk };
export type { TokenizerOptions };

/**
 * Convert raw text (Mandarin, English/Latin with real IPA lexicon, Japanese with
 * Kanji->Kana dictionary + Sokuon/Choonpu, and digits) into phrase-grouped `SpeechUnit[]`.
 */
export function tokenizeSpeech(text: string, options: TokenizerOptions = {}): SpeechUnit[] {
  if (!text) return [];
  const { lexicon, enLexicon, jaLexicon, languageMode = "auto", validKeys, validKeysByBank } = options;
  const zhKeys = validKeysByBank?.zh ?? validKeys;
  const enKeys = validKeysByBank?.en;
  const jaKeys = validKeysByBank?.ja;
  const hasZh = (k: string) => (!zhKeys || zhKeys.size === 0 ? true : zhKeys.has(k));

  const isJaAt = computeJapaneseClauseFlags(text, languageMode);
  const raw: RawToken[] = [];
  let i = 0;
  let lastJaMora = "a";
  let pendingJaSokuon = false;
  let prevWasKatakana = false;

  const pushJaMora = (
    at: number,
    surface: string,
    mora: string,
    weak: boolean,
    extra: { preGapSec?: number; jaWordBreak?: boolean; jaParticle?: boolean } = {}
  ) => {
    const useJa = Boolean(jaKeys && jaKeys.has(mora));
    const fallbackZh = KANA_ZH_FALLBACK[mora] ?? "a";
    if (useJa || hasZh(fallbackZh)) {
      const gap = extra.preGapSec ?? (pendingJaSokuon ? 0.048 : undefined);
      pendingJaSokuon = false;
      raw.push({
        at,
        surface,
        bank: useJa ? "ja" : "zh",
        key: useJa ? mora : fallbackZh,
        tone: 0,
        script: "kana",
        stress: false,
        weak,
        preGapSec: gap,
        jaWordBreak: extra.jaWordBreak,
        jaParticle: extra.jaParticle,
      });
      lastJaMora = mora;
    }
  };

  while (i < text.length) {
    const ch = text[i];

    // 1. Japanese clause Kanji -> Kana (longest-match compound words, then single-Kanji readings)
    if (isJaAt[i] && CJK_RE.test(ch) && jaLexicon) {
      prevWasKatakana = false;
      let matchedWord: string | null = null;
      let reading: string | null = null;
      const maxLen = Math.min(6, text.length - i);
      for (let len = maxLen; len >= 2; len--) {
        const sub = text.slice(i, i + len);
        const hit = jaLexicon.words[sub];
        if (hit) {
          matchedWord = sub;
          reading = hit;
          break;
        }
      }
      if (!matchedWord && jaLexicon.chars[ch]) {
        matchedWord = ch;
        reading = jaLexicon.chars[ch];
      }
      if (matchedWord && reading) {
        const { morae, trailingSokuon } = hiraganaToMorae(reading, lastJaMora);
        const wordLen = matchedWord.length;
        morae.forEach((m, idx) => {
          const charOffset = Math.min(wordLen - 1, Math.floor((idx * wordLen) / Math.max(1, morae.length)));
          pushJaMora(i + charOffset, matchedWord![charOffset], m.mora, m.weak, {
            preGapSec: m.preGapSec,
            jaWordBreak: idx === 0,
            jaParticle: false,
          });
        });
        if (trailingSokuon && !matchedWord.endsWith("っ") && !matchedWord.endsWith("ッ")) {
          pendingJaSokuon = true;
        }
        i += wordLen;
        continue;
      }
    }

    // 2. Direct Japanese Hiragana / Katakana / Choonpu / Sokuon
    if (ch === "っ" || ch === "ッ") {
      i++;
      continue;
    }
    const prevJaTok = raw.length > 0 && raw[raw.length - 1].bank === "ja" ? raw[raw.length - 1] : null;
    const isAdjacentToPrevJa = Boolean(prevJaTok && prevJaTok.at + prevJaTok.surface.length === i);

    if (ch === "ー") {
      const v = isAdjacentToPrevJa ? moraVowel(lastJaMora) : "a";
      pushJaMora(i, ch, v, true, { jaParticle: false });
      i++;
      continue;
    }
    const isKatakana = ch.charCodeAt(0) >= 0x30a1 && ch.charCodeAt(0) <= 0x30f6;
    const hira = normalizeKana(ch);
    if (KANA_JA_MAP[hira] !== undefined) {
      let consumed = 1;
      let surface = ch;
      let mora = KANA_JA_MAP[hira];
      let isParticle =
        !isKatakana &&
        isJapaneseEncliticParticle(text, i, hira, Boolean(isAdjacentToPrevJa && prevJaTok?.jaParticle));
      const wordBreak =
        (isKatakana && !prevWasKatakana) ||
        (!isAdjacentToPrevJa && raw.length > 0) ||
        Boolean(isAdjacentToPrevJa && prevJaTok?.jaParticle && !isParticle);

      if (i + 1 < text.length) {
        const pair = hira + normalizeKana(text[i + 1]);
        if (KANA_DIGRAPH_MAP[pair]) {
          mora = KANA_DIGRAPH_MAP[pair];
          surface = text.slice(i, i + 2);
          consumed = 2;
          isParticle = false;
        }
      }

      let isClitic = isParticle;
      let isLongU = false;
      if (consumed === 1 && !isKatakana) {
        if (hira === "は" && isJapaneseWaParticle(text, i)) {
          mora = "wa";
          isParticle = true;
          isClitic = true;
        } else if (hira === "へ" && isJapaneseEParticle(text, i)) {
          mora = "e";
          isParticle = true;
          isClitic = true;
        } else if (
          hira === "う" &&
          isAdjacentToPrevJa &&
          moraVowel(lastJaMora) === "o" &&
          text[i + 1] !== "ー"
        ) {
          mora = "o";
          isLongU = true;
          isParticle = false;
          isClitic = false;
        } else if (isJapanesePoliteTail(text, i, hira)) {
          isClitic = true;
        }
      }

      pushJaMora(i, surface, mora, isParticle || isLongU, {
        jaWordBreak: wordBreak,
        jaParticle: isClitic,
      });
      prevWasKatakana = isKatakana;
      i += consumed;
      continue;
    }
    prevWasKatakana = false;
    pendingJaSokuon = false;

    // 3. Mandarin Hanzi -> zh bank
    const han = lexicon?.get(ch);
    if (han && hasZh(han.key)) {
      raw.push({
        at: i,
        surface: ch,
        bank: "zh",
        key: han.key,
        tone: han.tone,
        script: "han",
        stress: false,
        weak: WEAK_PARTICLES.has(ch) || han.tone === 5,
      });
      i++;
      continue;
    }

    // 4. Digits 0-9 (language-aware)
    if (ch >= "0" && ch <= "9") {
      const idx = ch.charCodeAt(0) - 48;
      if (isJaAt[i] && jaKeys && jaKeys.size > 0) {
        const { morae } = hiraganaToMorae(DIGIT_JA_KANA[idx]);
        for (const m of morae) pushJaMora(i, ch, m.mora, false);
      } else if (languageMode === "en" && enKeys && enKeys.size > 0) {
        for (const k of DIGIT_EN_SYLS[idx].split(" ")) {
          if (enKeys.has(k)) {
            raw.push({ at: i, surface: ch, bank: "en", key: k, tone: 0, script: "digit", stress: false, weak: false });
          }
        }
      } else {
        const key = DIGIT_ZH_KEYS[idx];
        if (hasZh(key)) {
          raw.push({
            at: i,
            surface: ch,
            bank: "zh",
            key,
            tone: DIGIT_ZH_TONES[idx],
            script: "digit",
            stress: false,
            weak: false,
          });
        }
      }
      i++;
      continue;
    }

    // 5. English / Latin words
    if (isAsciiLetter(ch)) {
      const start = i;
      while (i < text.length && isAsciiLetter(text[i])) i++;
      const word = text.slice(start, i);
      raw.push(...tokenizeEnglishWord(word, start, enLexicon, enKeys, zhKeys));
      continue;
    }

    i++;
  }

  return groupTokensIntoPhrases(text, raw);
}

/** Group raw tokens into punctuation-delimited phrases and attach Japanese bunsetsu pitch roles. */
function groupTokensIntoPhrases(text: string, raw: RawToken[]): SpeechUnit[] {
  const units: SpeechUnit[] = [];
  let phraseStart = 0;
  let phraseIndex = 0;

  while (phraseStart < raw.length) {
    let phraseEnd = phraseStart;
    let closingPunct = "";

    while (phraseEnd < raw.length) {
      const nextCharIdx = phraseEnd + 1 < raw.length ? raw[phraseEnd + 1].at : text.length;
      const scanFrom = raw[phraseEnd].at + raw[phraseEnd].surface.length;
      let marks = "";
      for (let m = scanFrom; m < nextCharIdx; m++) {
        if (PHRASE_MARKS.has(text[m])) marks += text[m];
      }
      phraseEnd++;
      if (marks.length > 0) {
        closingPunct = marks;
        break;
      }
    }

    const count = phraseEnd - phraseStart;
    const phraseSlice = raw.slice(phraseStart, phraseEnd);
    assignJapanesePitchAccents(phraseSlice);

    for (let k = 0; k < count; k++) {
      const item = phraseSlice[k];
      units.push({
        id: units.length,
        at: item.at,
        surface: item.surface,
        bank: item.bank,
        key: item.key,
        tone: item.tone,
        script: item.script,
        stress: item.stress,
        weak: item.weak,
        jaAccent: item.jaAccent,
        preGapSec: item.preGapSec,
        pos: k,
        of: count,
        phraseIndex,
        closingPunct,
        isPhraseFinal: k >= count - 2,
      });
    }

    phraseStart = phraseEnd;
    phraseIndex++;
  }

  return units;
}
