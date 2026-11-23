import type { LanguageMode } from "../types";
import { CJK_RE, PHRASE_MARKS, type RawToken } from "./shared";

/** Native Japanese Hiragana -> Mora key (matching ja.json). */
export const KANA_JA_MAP: Readonly<Record<string, string>> = {
  あ: "a", い: "i", う: "u", え: "e", お: "o",
  か: "ka", き: "ki", く: "ku", け: "ke", こ: "ko",
  が: "ga", ぎ: "gi", ぐ: "gu", げ: "ge", ご: "go",
  さ: "sa", し: "shi", す: "su", せ: "se", そ: "so",
  ざ: "za", じ: "ji", ず: "zu", ぜ: "ze", ぞ: "zo",
  た: "ta", ち: "chi", つ: "tsu", て: "te", と: "to",
  だ: "da", ぢ: "ji", づ: "zu", で: "de", ど: "do",
  な: "na", に: "ni", ぬ: "nu", ね: "ne", の: "no",
  は: "ha", ひ: "hi", ふ: "fu", へ: "he", ほ: "ho",
  ば: "ba", び: "bi", ぶ: "bu", べ: "be", ぼ: "bo",
  ぱ: "pa", ぴ: "pi", ぷ: "pu", ぺ: "pe", ぽ: "po",
  ま: "ma", み: "mi", む: "mu", め: "me", も: "mo",
  や: "ya", ゆ: "yu", よ: "yo",
  ら: "ra", り: "ri", る: "ru", れ: "re", ろ: "ro",
  わ: "wa", を: "wo", ん: "n",
  ぁ: "a", ぃ: "i", ぅ: "u", ぇ: "e", ぉ: "o",
  ゃ: "ya", ゅ: "yu", ょ: "yo",
};

/** Palatalized Japanese Yoon & Katakana loanword digraphs (after Katakana -> Hiragana). */
export const KANA_DIGRAPH_MAP: Readonly<Record<string, string>> = {
  きゃ: "kya", きゅ: "kyu", きょ: "kyo",
  ぎゃ: "gya", ぎゅ: "gyu", ぎょ: "gyo",
  しゃ: "sha", しゅ: "shu", しょ: "sho", しぇ: "she",
  じゃ: "ja", じゅ: "ju", じょ: "jo",
  ちゃ: "cha", ちゅ: "chu", ちょ: "cho", ちぇ: "che",
  にゃ: "nya", にゅ: "nyu", にょ: "nyo",
  ひゃ: "hya", ひゅ: "hyu", ひょ: "hyo",
  びゃ: "bya", びゅ: "byu", びょ: "byo",
  ぴゃ: "pya", ぴゅ: "pyu", ぴょ: "pyo",
  みゃ: "mya", みゅ: "myu", みょ: "myo",
  りゃ: "rya", りゅ: "ryu", りょ: "ryo",
  ふぁ: "fa", ふぃ: "fi", ふぇ: "fe", ふぉ: "fo",
  てぃ: "ti", でぃ: "di",
};

export const JA_WEAK_KANA = new Set(["は", "が", "の", "を", "に", "で", "て", "ね", "よ", "と", "も"]);

export const KANA_ZH_FALLBACK: Readonly<Record<string, string>> = {
  a: "a", i: "yi", u: "wu", e: "ei", o: "ou",
  ka: "ka", ki: "qi", ku: "ku", ke: "ke", ko: "kou",
  ga: "ga", gi: "ji", gu: "gu", ge: "ge", go: "gou",
  sa: "sa", shi: "xi", su: "su", se: "se", so: "sou",
  za: "za", ji: "ji", zu: "zu", ze: "ze", zo: "zou",
  ta: "ta", chi: "qi", tsu: "cu", te: "te", to: "tou",
  da: "da", de: "de", do: "dou",
  na: "na", ni: "ni", nu: "nu", ne: "ne", no: "nou",
  ha: "ha", hi: "xi", fu: "fu", he: "he", ho: "hou",
  ba: "ba", bi: "bi", bu: "bu", be: "bei", bo: "bo",
  pa: "pa", pi: "pi", pu: "pu", pe: "pei", po: "po",
  ma: "ma", mi: "mi", mu: "mu", me: "mei", mo: "mo",
  ya: "ya", yu: "yu", yo: "yo",
  ra: "la", ri: "li", ru: "lu", re: "le", ro: "luo",
  wa: "wa", wo: "wo", n: "en",
  kya: "qia", kyu: "qiu", kyo: "qiao", gya: "jia", gyu: "jiu", gyo: "jiao",
  sha: "xia", shu: "xu", sho: "xiao", she: "xie", ja: "jia", ju: "ju", jo: "jiao",
  cha: "qia", chu: "qu", cho: "qiao", che: "qie", nya: "nia", nyu: "niu", nyo: "niao",
  hya: "xia", hyu: "xiu", hyo: "xiao", bya: "bia", byu: "biu", byo: "biao",
  pya: "pia", pyu: "piu", pyo: "piao", mya: "mia", myu: "miu", myo: "miao",
  rya: "lia", ryu: "liu", ryo: "liao", fa: "fa", fi: "fei", fe: "fei", fo: "fo", ti: "ti", di: "di",
};

const KANA_CHAR_RE = /[\u3041-\u3096\u30A1-\u30FAー]/;
const JA_HINT_KANJI_RE = /[々ヶ気駅広桜沢辺込辻働畑込]/;

const JA_WA_PREFIXES = [
  "こんにちは", "こんばんは",
  "これ", "それ", "あれ", "どれ", "ここ", "そこ", "あそこ",
  "わたし", "ぼく", "きみ", "あなた", "きょう", "あした", "きのう", "いま", "みんな",
  "から", "まで", "より", "ほど", "だけ", "しか",
] as const;

export function normalizeKana(ch: string): string {
  const code = ch.charCodeAt(0);
  if (code >= 0x30a1 && code <= 0x30f6) {
    return String.fromCharCode(code - 0x60);
  }
  return ch;
}

export function moraVowel(moraKey: string): string {
  const last = moraKey[moraKey.length - 1];
  if (last === "a" || last === "i" || last === "u" || last === "e" || last === "o") return last;
  return "u";
}

/** Expand a Hiragana string (from Kana text or Japanese Kanji reading) into morae with o+う long vowels & sokuon gaps. */
export function hiraganaToMorae(
  hiraText: string,
  prevMora = "a"
): { morae: Array<{ mora: string; weak: boolean; preGapSec?: number; isLong?: boolean }>; trailingSokuon: boolean } {
  const out: Array<{ mora: string; weak: boolean; preGapSec?: number; isLong?: boolean }> = [];
  let p = 0;
  let lastMora = prevMora;
  let pendingSokuon = false;

  while (p < hiraText.length) {
    const ch = normalizeKana(hiraText[p]);
    if (ch === "っ") {
      pendingSokuon = true;
      p++;
      continue;
    }
    if (ch === "ー") {
      const v = moraVowel(lastMora);
      out.push({ mora: v, weak: true, isLong: true, preGapSec: pendingSokuon ? 0.055 : undefined });
      pendingSokuon = false;
      lastMora = v;
      p++;
      continue;
    }
    if (p + 1 < hiraText.length) {
      const pair = ch + normalizeKana(hiraText[p + 1]);
      if (KANA_DIGRAPH_MAP[pair]) {
        const m = KANA_DIGRAPH_MAP[pair];
        out.push({ mora: m, weak: false, preGapSec: pendingSokuon ? 0.055 : undefined });
        pendingSokuon = false;
        lastMora = m;
        p += 2;
        continue;
      }
    }
    if (ch === "う" && out.length > 0 && moraVowel(lastMora) === "o") {
      out.push({ mora: "o", weak: true, isLong: true, preGapSec: pendingSokuon ? 0.055 : undefined });
      pendingSokuon = false;
      lastMora = "o";
      p++;
      continue;
    }
    if (KANA_JA_MAP[ch]) {
      const m = KANA_JA_MAP[ch];
      out.push({ mora: m, weak: JA_WEAK_KANA.has(ch), preGapSec: pendingSokuon ? 0.055 : undefined });
      pendingSokuon = false;
      lastMora = m;
    }
    p++;
  }
  return { morae: out, trailingSokuon: pendingSokuon };
}

/** Detect whether raw `は` at index `i` is the Japanese topic particle (pronounced "wa"). */
export function isJapaneseWaParticle(text: string, i: number): boolean {
  if (i <= 0) return false;
  const prev = text[i - 1];
  const next = text[i + 1] ?? "";
  const uptoHere = text.slice(Math.max(0, i - 4), i + 1);
  if (uptoHere.endsWith("こんにちは") || uptoHere.endsWith("こんばんは")) return true;

  const prevIsBoundary = PHRASE_MARKS.has(prev) || /\s/.test(prev);
  if (prevIsBoundary) return false;

  if (prev === "お" && next === "よ") return false;
  if (prev === "ご" && next === "ん") return false;

  if (CJK_RE.test(prev) || /[\u30A1-\u30FAー]/.test(prev) || /[にでてとの]/.test(prev)) {
    return true;
  }

  const before = text.slice(Math.max(0, i - 4), i);
  for (const prefix of JA_WA_PREFIXES) {
    if (prefix.length <= 4 && before.endsWith(prefix)) return true;
  }
  return false;
}

/** Detect whether raw `へ` at index `i` is the Japanese direction particle (pronounced "e"). */
export function isJapaneseEParticle(text: string, i: number): boolean {
  if (i <= 0) return false;
  const prev = text[i - 1];
  if (CJK_RE.test(prev) || /[\u30A1-\u30FAー]/.test(prev)) return true;
  const before = text.slice(Math.max(0, i - 3), i);
  return before.endsWith("ここ") || before.endsWith("そこ") || before.endsWith("あそこ") || before.endsWith("どこ");
}

/** Detect polite copula/auxiliary tail morae (です/でした/ます/ました/ません) so pitch stays smoothly lowered. */
export function isJapanesePoliteTail(text: string, i: number, hira: string): boolean {
  const prev = i > 0 ? normalizeKana(text[i - 1]) : "";
  const prev2 = i > 1 ? normalizeKana(text[i - 2]) : "";
  const next = i + 1 < text.length ? normalizeKana(text[i + 1]) : "";
  if (hira === "で" && (next === "す" || next === "し")) return true;
  if ((hira === "す" || hira === "し") && (prev === "で" || prev === "ま")) return true;
  if (hira === "た" && prev === "し" && (prev2 === "で" || prev2 === "ま")) return true;
  if (hira === "ま" && i > 0 && !PHRASE_MARKS.has(text[i - 1]) && (next === "す" || next === "し" || next === "せ")) {
    return true;
  }
  if (hira === "せ" && prev === "ま" && next === "ん") return true;
  if (hira === "ん" && prev === "せ" && prev2 === "ま") return true;
  return false;
}

/** Assign per-bunsetsu Tokyo pitch-accent roles (low -> rise -> high -> fall -> drop) across a Japanese phrase. */
export function assignJapanesePitchAccents(phraseTokens: RawToken[]): void {
  let start = 0;
  while (start < phraseTokens.length) {
    if (phraseTokens[start].bank !== "ja") {
      start++;
      continue;
    }
    let end = start + 1;
    let seenParticle = Boolean(phraseTokens[start].jaParticle);
    while (end < phraseTokens.length && phraseTokens[end].bank === "ja") {
      const tok = phraseTokens[end];
      if (tok.jaWordBreak && end > start) break;
      if (seenParticle && !tok.jaParticle) break;
      if (tok.jaParticle) seenParticle = true;
      end++;
    }

    const slice = phraseTokens.slice(start, end);
    const contentEnd = slice.findIndex((t) => t.jaParticle);
    const contentLen = contentEnd === -1 ? slice.length : contentEnd;

    for (let k = 0; k < slice.length; k++) {
      const tok = slice[k];
      if (tok.jaParticle) {
        tok.jaAccent = "drop";
      } else if (contentLen === 1) {
        tok.jaAccent = "rise";
      } else if (k === 0) {
        tok.jaAccent = "low";
      } else if (k === 1) {
        tok.jaAccent = "rise";
      } else if (k === contentLen - 1 && (contentEnd !== -1 || contentLen >= 3)) {
        tok.jaAccent = "fall";
      } else {
        tok.jaAccent = "high";
      }
    }

    start = end;
  }
}

/** Detect whether the clause containing index `pos` should read Kanji as Japanese. */
export function computeJapaneseClauseFlags(text: string, mode: LanguageMode): boolean[] {
  const flags = new Array<boolean>(text.length).fill(false);
  if (mode === "zh" || mode === "en") return flags;
  if (mode === "ja") {
    flags.fill(true);
    return flags;
  }

  let start = 0;
  while (start < text.length) {
    let end = start;
    while (end < text.length && !PHRASE_MARKS.has(text[end])) end++;
    while (end < text.length && PHRASE_MARKS.has(text[end])) end++;
    const slice = text.slice(start, end);
    if (KANA_CHAR_RE.test(slice) || JA_HINT_KANJI_RE.test(slice)) {
      for (let k = start; k < end; k++) flags[k] = true;
    }
    start = end;
  }

  return flags;
}
