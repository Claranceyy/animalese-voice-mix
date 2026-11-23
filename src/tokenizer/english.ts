import type { EnglishLexicon } from "../types";
import type { RawToken } from "./shared";

const SINGLE_LETTER_ZH_FALLBACK: Record<string, string> = {
  a: "a", b: "bi", c: "si", d: "di", e: "ei", f: "fa", g: "ga",
  h: "ha", i: "yi", j: "ji", k: "ka", l: "la", m: "mi", n: "ni",
  o: "ou", p: "pi", q: "qi", r: "ri", s: "si", t: "ti", u: "wu",
  v: "wei", w: "wa", x: "xi", y: "ya", z: "zi",
};

const VOWEL_GRAPH_MAP: readonly [string, string][] = [
  ["ee", "ee"], ["ea", "ee"], ["oo", "oo"], ["ai", "ei"], ["ay", "ei"],
  ["igh", "ai"], ["ie", "ai"], ["oa", "ou"], ["ow", "au"], ["ou", "au"],
  ["oi", "oi"], ["oy", "oi"], ["ar", "ar"], ["or", "or"], ["er", "er"], ["ir", "er"], ["ur", "er"],
];

const ONSET_DIGRAPHS = [
  "spr", "thr",
  "sh", "ch", "th", "wh", "ph", "qu",
  "st", "tr", "kr", "cr", "pr", "br",
  "dr", "fr", "gr", "pl", "fl", "kl", "cl", "bl", "sl", "sp", "sk",
  "sw", "sm", "sn", "tw", "kw",
] as const;

export function isAsciiLetter(ch: string): boolean {
  const c = ch.charCodeAt(0);
  return (c >= 65 && c <= 90) || (c >= 97 && c <= 122);
}

/** Map an English/Latin chunk onto the native English bank (`en.json`) when available. */
export function mapEnglishChunk(sub: string, enKeys: ReadonlySet<string>): string {
  const s = sub.toLowerCase();
  const candidates: string[] = [];

  if (s.includes("tion") || s.includes("sion")) candidates.push("ion");
  if (s.includes("ing")) candidates.push("ing");

  let onset = "";
  let rest = s;
  for (const dg of ONSET_DIGRAPHS) {
    if (s.startsWith(dg)) {
      onset =
        dg === "wh"
          ? "w"
          : dg === "ph"
            ? "f"
            : dg === "qu"
              ? "kw"
              : dg === "thr"
                ? "th"
                : dg === "cr"
                  ? "kr"
                  : dg === "cl"
                    ? "kl"
                    : dg;
      rest = s.slice(dg.length);
      break;
    }
  }
  if (!onset && s.length > 0 && !/[aeiouy]/.test(s[0])) {
    const c0 = s[0];
    onset = c0 === "c" ? (/[eiy]/.test(s[1] ?? "") ? "s" : "k") : c0 === "q" ? "k" : c0 === "x" ? "s" : c0;
    rest = s.slice(1);
  }

  let vKey = "";
  for (const [pat, vk] of VOWEL_GRAPH_MAP) {
    if (rest.includes(pat)) {
      vKey = vk;
      break;
    }
  }
  if (!vKey) {
    const m = /[aeiouy]/.exec(rest);
    if (m) {
      const vc = m[0];
      vKey = vc === "y" ? "ee" : vc;
    }
  }

  if (onset && vKey) {
    candidates.push(`${onset}${vKey}`, `${onset}${vKey[0]}`);
    if (onset.length === 2) candidates.push(`${onset[1]}${vKey}`, `${onset[0]}${vKey}`);
  }
  if (vKey) candidates.push(vKey);
  if (onset) candidates.push(`${onset}a`, `${onset}e`, `${onset}i`, onset);
  candidates.push(s.slice(0, 2), s[0] ?? "a", "a");

  for (const cand of candidates) {
    if (enKeys.has(cand)) return cand;
  }
  return "a";
}

/** Split an out-of-vocabulary English word on phonetic vowel clusters (ignoring silent trailing -e). */
export function splitEnglishWordHeuristic(lower: string): string[] {
  const clean = lower.length > 3 && lower.endsWith("e") && !/[aeiou]e$/.test(lower) ? lower.slice(0, -1) : lower;
  const parts = clean.match(/[^aeiouy]*[aeiouy]+(?:[^aeiouy](?=[^aeiouy]|$))?/g);
  if (!parts || parts.length === 0) return [lower];
  return parts;
}

/** Lookup English word in `en-lexicon.json` (with light inflection fallback). */
export function lookupEnglishLexicon(
  lower: string,
  lexicon: EnglishLexicon
): Array<{ key: string; stress: boolean }> | null {
  const parse = (raw: string) =>
    raw.split(" ").map((tok) =>
      tok.endsWith("*") ? { key: tok.slice(0, -1), stress: true } : { key: tok, stress: false }
    );

  if (lexicon[lower]) return parse(lexicon[lower]);

  if (lower.endsWith("ing") && lower.length > 5) {
    const stem = lower.slice(0, -3);
    const hit = lexicon[stem] ?? lexicon[stem + "e"];
    if (hit) return [...parse(hit), { key: "ing", stress: false }];
  }
  if (lower.endsWith("es") && lower.length > 4) {
    const hit = lexicon[lower.slice(0, -2)] ?? lexicon[lower.slice(0, -1)];
    if (hit) {
      const base = parse(hit);
      const lastKey = base[base.length - 1]?.key ?? "";
      if (!lastKey.startsWith("_")) base.push({ key: "_z", stress: false });
      return base;
    }
  }
  if (lower.endsWith("s") && lower.length > 3 && !lower.endsWith("ss")) {
    const hit = lexicon[lower.slice(0, -1)];
    if (hit) {
      const base = parse(hit);
      const lastKey = base[base.length - 1]?.key ?? "";
      const sibilantEnd = lastKey === "_s" || lastKey === "_z" || lastKey === "_ks" || lastKey === "_ts";
      if (!sibilantEnd) {
        const voiceless = lastKey === "_p" || lastKey === "_t" || lastKey === "_k" || lastKey === "_f";
        if (lastKey === "_k") base[base.length - 1] = { key: "_ks", stress: false };
        else if (lastKey === "_t") base[base.length - 1] = { key: "_ts", stress: false };
        else base.push({ key: voiceless ? "_s" : "_z", stress: false });
      }
      return base;
    }
  }
  if (lower.endsWith("ed") && lower.length > 4) {
    const hit = lexicon[lower.slice(0, -2)] ?? lexicon[lower.slice(0, -1)];
    if (hit) {
      const base = parse(hit);
      const lastKey = base[base.length - 1]?.key ?? "";
      if (lastKey !== "_t" && lastKey !== "_d") {
        const voiceless =
          lastKey === "_p" || lastKey === "_k" || lastKey === "_f" || lastKey === "_s" || lastKey === "_sh" || lastKey === "_ch";
        base.push({ key: voiceless ? "_t" : "_d", stress: false });
      }
      return base;
    }
  }
  if (lower.endsWith("ly") && lower.length > 4) {
    const hit = lexicon[lower.slice(0, -2)];
    if (hit) return [...parse(hit), { key: "lee", stress: false }];
  }

  return null;
}

/**
 * Fallback mapper from English/Latin chunks to verified Mandarin syllable keys
 * (used when only `zh.ogg` is loaded without `en.ogg`).
 */
export function mapLatinChunk(sub: string, validKeys?: ReadonlySet<string>): string {
  const s = sub.toLowerCase();
  const has = (k: string) => (!validKeys || validKeys.size === 0 ? true : validKeys.has(k));

  const candidates: string[] = [];
  if (s.length >= 3) candidates.push(s.slice(0, 3));
  if (s.length >= 2) {
    const two = s.slice(0, 2);
    candidates.push(two);
    if (two === "th") candidates.push("si", "da");
    if (two === "ph") candidates.push("fa", "fu");
    if (two === "wh") candidates.push("wa", "wu");
    if (two === "oo") candidates.push("wu");
    if (two === "ee") candidates.push("yi");
    if (two === "ck") candidates.push("ka", "ke");
    if (two === "qu") candidates.push("kua", "qu");
    if (two === "tr") candidates.push("chu", "ta");
    if (two === "dr") candidates.push("zhu", "da");
  }

  const c0 = s[0] ?? "a";
  if (c0 === "a") candidates.push("a", "ai");
  else if (c0 === "e") candidates.push("ei", "e");
  else if (c0 === "i") candidates.push("yi", "ai");
  else if (c0 === "o") candidates.push("ou", "o");
  else if (c0 === "u") candidates.push("wu", "you");
  else {
    const lead = c0 === "c" ? "k" : c0 === "v" ? "w" : c0;
    candidates.push(`${lead}a`, `${lead}i`, `${lead}u`, `${lead}e`, `${lead}o`);
  }

  const single = SINGLE_LETTER_ZH_FALLBACK[c0];
  if (single) candidates.push(single);
  candidates.push("la", "mi", "a");

  for (const cand of candidates) {
    if (has(cand)) return cand;
  }
  return "a";
}

/** Tokenize a contiguous ASCII word into `RawToken` slices using IPA lexicon or heuristic G2P. */
export function tokenizeEnglishWord(
  word: string,
  start: number,
  enLexicon: EnglishLexicon | null | undefined,
  enKeys: ReadonlySet<string> | undefined,
  zhKeys: ReadonlySet<string> | undefined
): RawToken[] {
  const out: RawToken[] = [];
  const lower = word.toLowerCase();
  const hasEnBank = Boolean(enKeys && enKeys.size > 0);
  const lexHit = hasEnBank && enLexicon ? lookupEnglishLexicon(lower, enLexicon) : null;

  if (lexHit && lexHit.length > 0 && enKeys) {
    const count = lexHit.length;
    for (let c = 0; c < count; c++) {
      const isCoda = lexHit[c].key.startsWith("_");
      const segStart = isCoda ? Math.max(0, word.length - 1) : Math.floor((c * word.length) / count);
      const segEnd = Math.floor(((c + 1) * word.length) / count);
      const surface = word.slice(segStart, Math.max(segStart + 1, segEnd));
      const key = enKeys.has(lexHit[c].key) ? lexHit[c].key : mapEnglishChunk(surface, enKeys);
      out.push({
        at: start + segStart,
        surface,
        bank: "en",
        key,
        tone: 0,
        script: "latin",
        stress: isCoda ? false : lexHit[c].stress,
        weak: isCoda || (!lexHit[c].stress && count > 1 && word.length <= 4),
      });
    }
    return out;
  }

  if (hasEnBank && enKeys) {
    const chunks = splitEnglishWordHeuristic(lower);
    let cursor = 0;
    chunks.forEach((chunk, c) => {
      const at = start + Math.min(word.length - 1, cursor);
      const surface = word.slice(cursor, Math.min(word.length, cursor + chunk.length)) || word[word.length - 1];
      cursor += chunk.length;
      out.push({
        at,
        surface,
        bank: "en",
        key: mapEnglishChunk(chunk, enKeys),
        tone: 0,
        script: "latin",
        stress: c === 0 && chunks.length > 1,
        weak: c > 0 && lower.length <= 4,
      });
    });
    return out;
  }

  const chunks = Math.max(1, Math.ceil(lower.length / 3));
  for (let c = 0; c < chunks; c++) {
    const segStart = Math.floor((c * lower.length) / chunks);
    const segEnd = Math.floor(((c + 1) * lower.length) / chunks);
    const surface = word.slice(segStart, Math.max(segStart + 1, segEnd));
    out.push({
      at: start + segStart,
      surface,
      bank: "zh",
      key: mapLatinChunk(lower.slice(segStart, Math.min(lower.length, segStart + 3)), zhKeys),
      tone: 0,
      script: "latin",
      stress: c === 0 && chunks > 1,
      weak: c > 0 && lower.length <= 4,
    });
  }
  return out;
}
