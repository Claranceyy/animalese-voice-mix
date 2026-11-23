import type {
  EnglishLexicon,
  HanziLookupMap,
  JaPitchRole,
  JapaneseLexicon,
  LanguageMode,
  RawHanziLexicon,
  ToneNumber,
  UnitScript,
  VoiceBankId,
} from "../types";

export const WEAK_PARTICLES = new Set([
  "的", "了", "吗", "呢", "啊", "呀", "吧", "哦", "啦", "么", "着", "过", "喵", "～",
  "地", "得", "嘛", "哇", "哟", "噢",
]);

export const PHRASE_MARKS = new Set([
  ",", ".", "!", "?", ";", ":", "…", "~",
  "，", "。", "！", "？", "；", "：", "、", "～", "\n",
]);

export const DIGIT_ZH_KEYS = ["ling", "yi", "er", "san", "si", "wu", "liu", "qi", "ba", "jiu"] as const;
export const DIGIT_ZH_TONES: ToneNumber[] = [2, 1, 4, 1, 4, 3, 4, 1, 1, 3];
export const DIGIT_JA_KANA = ["ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"] as const;
export const DIGIT_EN_SYLS = ["zee rou", "wu", "too", "thee", "fo", "fai", "si", "se vu", "ei", "nai"] as const;

export const CJK_RE = /[\u4E00-\u9FFF\u3400-\u4DBF々]/;

export interface RawToken {
  at: number;
  surface: string;
  bank: VoiceBankId;
  key: string;
  tone: ToneNumber;
  script: UnitScript;
  stress: boolean;
  weak: boolean;
  jaAccent?: JaPitchRole;
  preGapSec?: number;
  jaWordBreak?: boolean;
  jaParticle?: boolean;
}

export interface TokenizerOptions {
  lexicon?: HanziLookupMap | null;
  enLexicon?: EnglishLexicon | null;
  jaLexicon?: JapaneseLexicon | null;
  languageMode?: LanguageMode;
  /** Legacy single-bank key set (treated as `zh` keys) */
  validKeys?: ReadonlySet<string>;
  /** Per-language available bank keys (`zh`, `en`, `ja`) */
  validKeysByBank?: Partial<Record<VoiceBankId, ReadonlySet<string>>>;
}

/** Invert `zh-lexicon.json` (`{ "ni3": "你妳..." }`) into a fast char -> (key, tone) map. */
export function parseHanziLexicon(raw: RawHanziLexicon): HanziLookupMap {
  const map: HanziLookupMap = new Map();
  for (const [k, chars] of Object.entries(raw)) {
    if (k.length < 2) continue;
    const last = k[k.length - 1];
    const toneNum = last >= "1" && last <= "5" ? (Number(last) as ToneNumber) : 1;
    const syllable = last >= "1" && last <= "5" ? k.slice(0, -1) : k;
    for (const ch of chars) {
      if (!map.has(ch)) {
        map.set(ch, { key: syllable, tone: toneNum });
      }
    }
  }
  return map;
}
