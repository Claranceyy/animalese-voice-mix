import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_CONFIG,
  VoiceBankStore,
  VoiceClock,
  computeEqualPowerGains,
  mapLatinChunk,
  parseHanziLexicon,
  resolveFormantSpec,
  simulateVoiceClock,
  solvePitchContour,
  textRevealSteps,
  tokenizeSpeech,
} from "../src/index";
import type { EnglishLexicon, JapaneseLexicon, RawHanziLexicon, VoiceBankManifest } from "../src/index";

const speechDir = path.resolve(process.cwd(), "speech");
const rawLexicon = JSON.parse(fs.readFileSync(path.join(speechDir, "zh-lexicon.json"), "utf8")) as RawHanziLexicon;
const enLexicon = JSON.parse(fs.readFileSync(path.join(speechDir, "en-lexicon.json"), "utf8")) as EnglishLexicon;
const jaLexicon = JSON.parse(fs.readFileSync(path.join(speechDir, "ja-lexicon.json"), "utf8")) as JapaneseLexicon;
const zhManifest = JSON.parse(fs.readFileSync(path.join(speechDir, "zh.json"), "utf8")) as VoiceBankManifest;
const enManifest = JSON.parse(fs.readFileSync(path.join(speechDir, "en.json"), "utf8")) as VoiceBankManifest;
const jaManifest = JSON.parse(fs.readFileSync(path.join(speechDir, "ja.json"), "utf8")) as VoiceBankManifest;
const lexicon = parseHanziLexicon(rawLexicon);
const validKeys = new Set(Object.keys(zhManifest.units));
const enKeys = new Set(Object.keys(enManifest.units));
const jaKeys = new Set(Object.keys(jaManifest.units));

describe("VoiceClock dual-clock scheduler", () => {
  it("caps syllable onset rate by unitsPerSecond and applies punctuation gaps", () => {
    const clock = new VoiceClock(0);
    const d1 = clock.place(0, { id: 0, at: 0, surface: "你", key: "ni" }, {
      weak: false,
      isPhraseFinal: false,
      unitsPerSecond: 10,
      durationSec: 0.1,
      maxLag: 1,
    });
    expect(d1.dropped).toBe(false);
    expect(d1.scheduledAt).toBeCloseTo(0, 5);

    // Next char arrives at 0.02s, must wait until 0.10s (1/10s)
    const d2 = clock.place(0.02, { id: 1, at: 1, surface: "好", key: "hao" }, {
      weak: false,
      isPhraseFinal: false,
      unitsPerSecond: 10,
      durationSec: 0.1,
      maxLag: 1,
    });
    expect(d2.scheduledAt).toBeCloseTo(0.1, 5);
    expect(d2.lag).toBeCloseTo(0.08, 5);

    // Comma adds 0.14s after clock.end (0.20s -> free at 0.34s)
    const pause = clock.mark("，", 2, 0.05);
    expect(pause).not.toBeNull();
    expect(pause?.freeAfterAt).toBeCloseTo(0.34, 5);

    const d3 = clock.place(0.1, { id: 2, at: 3, surface: "呀", key: "ya" }, {
      weak: true,
      isPhraseFinal: true,
      unitsPerSecond: 10,
      durationSec: 0.1,
      maxLag: 1,
    });
    expect(d3.scheduledAt).toBeCloseTo(0.34, 5);
    expect(d3.gapApplied).toBeCloseTo(0.14, 5);
  });

  it("drops weak units past maxLag and non-final units past 2*maxLag, while protecting phrase finals", () => {
    const clock = new VoiceClock(0);
    // Push voice end out to 0.50s
    clock.place(0, { id: 0, at: 0, surface: "哈", key: "ha" }, {
      weak: false,
      isPhraseFinal: false,
      unitsPerSecond: 2,
      durationSec: 0.1,
      maxLag: 0.3,
    });

    // Next candidate would be 0.50s while revealedAt = 0.10s -> lag = 0.40s > maxLag (0.30s)
    const weakDrop = clock.place(0.1, { id: 1, at: 1, surface: "的", key: "de" }, {
      weak: true,
      isPhraseFinal: false,
      unitsPerSecond: 2,
      durationSec: 0.1,
      maxLag: 0.3,
    });
    expect(weakDrop.dropped).toBe(true);
    expect(weakDrop.dropReason).toBe("lag-weak");

    // Non-weak at lag 0.40s (< 0.60s) still plays
    const strongKeep = clock.place(0.1, { id: 2, at: 2, surface: "猫", key: "mao" }, {
      weak: false,
      isPhraseFinal: false,
      unitsPerSecond: 2,
      durationSec: 0.1,
      maxLag: 0.3,
    });
    expect(strongKeep.dropped).toBe(false);
    expect(strongKeep.scheduledAt).toBeCloseTo(0.5, 5);

    // Next candidate is 1.00s while revealedAt = 0.12s -> lag = 0.88s > 2*maxLag (0.60s)
    const hardDrop = clock.place(0.12, { id: 3, at: 3, surface: "快", key: "kuai" }, {
      weak: false,
      isPhraseFinal: false,
      unitsPerSecond: 2,
      durationSec: 0.1,
      maxLag: 0.3,
    });
    expect(hardDrop.dropped).toBe(true);
    expect(hardDrop.dropReason).toBe("lag-hard");

    // Phrase-final is ALWAYS preserved even when lag > 2*maxLag
    const finalKeep = clock.place(0.12, { id: 4, at: 4, surface: "啦", key: "la" }, {
      weak: true,
      isPhraseFinal: true,
      unitsPerSecond: 2,
      durationSec: 0.1,
      maxLag: 0.3,
    });
    expect(finalKeep.dropped).toBe(false);
    expect(finalKeep.scheduledAt).toBeCloseTo(1.0, 5);
  });

  it("simulates a stress sentence and triggers weak shedding deterministically", () => {
    const text = "我觉得这个算法真的太好玩了呀，的了吗呢啊呀吧哦啦都会丢掉，最后两个字还在！";
    const units = tokenizeSpeech(text, { lexicon, validKeys });
    const sim = simulateVoiceClock(text, units, {
      typingSpeed: 2.0,
      unitsPerSecond: 7.5,
      maxLag: 0.3,
      unitDurationFor: () => 0.1,
    });
    expect(sim.decisions.length).toBe(units.length);
    const droppedWeak = sim.decisions.filter((d) => d.dropReason === "lag-weak");
    const finals = sim.decisions.filter((d) => d.isPhraseFinal);
    expect(droppedWeak.length).toBeGreaterThan(0);
    expect(finals.every((d) => !d.dropped)).toBe(true);
  });

  it("advances Latin letters faster than Hanzi", () => {
    expect(textRevealSteps("a")).toBeLessThan(textRevealSteps("猫"));
    expect(textRevealSteps("！")).toBeGreaterThan(textRevealSteps("猫"));
  });
});

describe("ProsodyTokenizer", () => {
  it("tokenizes Mandarin, digits, Latin, and Japanese kana with valid bank keys", () => {
    const text = "你好 2026! Hello world, ありがとう？";
    const units = tokenizeSpeech(text, { lexicon, validKeys });
    expect(units.length).toBeGreaterThan(10);
    for (const u of units) {
      expect(validKeys.has(u.key)).toBe(true);
    }
    expect(units.some((u) => u.script === "han")).toBe(true);
    expect(units.some((u) => u.script === "digit")).toBe(true);
    expect(units.some((u) => u.script === "latin")).toBe(true);
    expect(units.some((u) => u.script === "kana")).toBe(true);

    const lastUnit = units[units.length - 1];
    expect(lastUnit.closingPunct).toContain("？");
    expect(lastUnit.isPhraseFinal).toBe(true);
  });

  it("never drops tricky English consonant clusters", () => {
    for (const chunk of ["str", "chr", "th", "ph", "ck", "xyz", "qu"]) {
      const mapped = mapLatinChunk(chunk, validKeys);
      expect(validKeys.has(mapped)).toBe(true);
    }
  });

  it("routes Mandarin, English, and Japanese (including Yoon digraphs) to native zh/en/ja banks", () => {
    const text = "Hello Island! 欢迎来到小岛，ありがとうにゃん！";
    const units = tokenizeSpeech(text, {
      lexicon,
      validKeysByBank: { zh: validKeys, en: enKeys, ja: jaKeys },
    });

    const enUnits = units.filter((u) => u.bank === "en");
    const zhUnits = units.filter((u) => u.bank === "zh");
    const jaUnits = units.filter((u) => u.bank === "ja");

    expect(enUnits.length).toBeGreaterThanOrEqual(4);
    expect(zhUnits.length).toBe(6);
    expect(jaUnits.length).toBeGreaterThanOrEqual(6);

    for (const u of enUnits) expect(enKeys.has(u.key)).toBe(true);
    for (const u of zhUnits) expect(validKeys.has(u.key)).toBe(true);
    for (const u of jaUnits) expect(jaKeys.has(u.key)).toBe(true);

    // Palatalized digraph にゃ -> nya
    expect(jaUnits.some((u) => u.surface === "にゃ" && u.key === "nya")).toBe(true);
    expect(jaUnits.some((u) => u.key === "n")).toBe(true);
  });

  it("reads Japanese Kanji as Japanese Kana in Japanese clauses and uses real English IPA with stress", () => {
    const jaText = "今日は天気がいいですね！可愛い猫と一緒に遊ぼう！";
    const jaUnits = tokenizeSpeech(jaText, {
      lexicon,
      enLexicon,
      jaLexicon,
      languageMode: "auto",
      validKeysByBank: { zh: validKeys, en: enKeys, ja: jaKeys },
    });
    // Every single unit (including 今日, 天気, 可愛い, 猫, 一緒, 遊) must route to ja, NEVER Mandarin zh!
    expect(jaUnits.every((u) => u.bank === "ja")).toBe(true);
    const jaSeq = jaUnits.map((u) => u.key);
    // 今日 -> kyo, o (o-row + う long vowel!); は -> wa (topic particle!); 天気 -> te, n, ki; 猫 -> ne, ko
    expect(jaSeq.slice(0, 3)).toEqual(["kyo", "o", "wa"]);
    expect(jaSeq).toContain("te");
    expect(jaSeq).toContain("ki");
    expect(jaSeq).toContain("ne");
    expect(jaSeq).toContain("ko");
    // Sokuon っ inside 一緒 emits preGapSec=0.055 on sho instead of "tsu"
    expect(jaSeq).not.toContain("tsu");
    const shoUnit = jaUnits.find((u) => u.key === "sho");
    expect(shoUnit?.preGapSec).toBeCloseTo(0.055, 3);
    // Japanese pitch accent produces varied low / rise / drop contour roles, and polite tail す stays dropped (not rising!)
    expect(jaUnits.some((u) => u.jaAccent === "low")).toBe(true);
    expect(jaUnits.some((u) => u.jaAccent === "rise")).toBe(true);
    expect(jaUnits.some((u) => u.jaAccent === "drop")).toBe(true);
    const suUnit = jaUnits.find((u) => u.surface === "す");
    expect(suUnit?.jaAccent).toBe("drop");

    // こんにちは / こんばんは pronounce trailing は as "wa"
    const greetJa = tokenizeSpeech("こんにちは！わたしはねこです。", {
      lexicon,
      jaLexicon,
      languageMode: "auto",
      validKeysByBank: { zh: validKeys, en: enKeys, ja: jaKeys },
    });
    expect(greetJa.slice(0, 5).map((u) => u.key)).toEqual(["ko", "n", "ni", "chi", "wa"]);
    expect(greetJa.slice(5, 9).map((u) => u.key)).toEqual(["wa", "ta", "shi", "wa"]);

    // Pure Kanji with forced Japanese mode
    const pureKanjiJa = tokenizeSpeech("日本語", {
      lexicon,
      jaLexicon,
      languageMode: "ja",
      validKeysByBank: { zh: validKeys, en: enKeys, ja: jaKeys },
    });
    expect(pureKanjiJa.every((u) => u.bank === "ja")).toBe(true);
    expect(pureKanjiJa.map((u) => u.key)).toEqual(["ni", "ho", "n", "go"]);

    // English real IPA + lexical stress
    const enUnits = tokenizeSpeech("Hello Island! Coffee?", {
      lexicon,
      enLexicon,
      validKeysByBank: { zh: validKeys, en: enKeys, ja: jaKeys },
    });
    expect(enUnits.every((u) => u.bank === "en" && enKeys.has(u.key))).toBe(true);
    // "Hello" stresses second syllable ("lou"), "Island" starts with stressed "ai"
    expect(enUnits[1].key).toBe("lou");
    expect(enUnits[1].stress).toBe(true);
    expect(enUnits[2].key).toBe("ai");
    expect(enUnits[2].stress).toBe(true);
  });

  it("tokenizes representative ZH, JA, and EN sentences cleanly to their native banks", () => {
    const bankKeys = { zh: validKeys, en: enKeys, ja: jaKeys };
    const cases: Array<{ bank: "zh" | "ja" | "en"; lines: string[] }> = [
      {
        bank: "zh",
        lines: [
          "你好呀！今天岛上的海风好舒服，要不要一起去海边捡贝壳？",
          "妈麻马骂！花画化话？风声轻轻的，树叶沙沙响。",
          "明天就是烟花大会啦，记得换上最喜欢的浴衣！",
        ],
      },
      {
        bank: "ja",
        lines: [
          "今日は天気がいいですね！可愛い猫と一緒に島で遊ぼう？",
          "こんにちは！熱いコーヒーを入れましたよ、ミルクも入れますか？",
          "砂浜で小さなガラスのびんを見つけました！中にお手紙が入っていますよ。",
        ],
      },
      {
        bank: "en",
        lines: [
          "Good morning! The ocean breeze feels wonderful today, shall we go fishing?",
          "Look up at the night sky! A meteor shower is starting, make a wish!",
          "Fresh apples and sweet fruit are ready to pick under the trees.",
        ],
      },
    ];
    for (const c of cases) {
      for (const line of c.lines) {
        const units = tokenizeSpeech(line, {
          lexicon,
          enLexicon,
          jaLexicon,
          languageMode: "auto",
          validKeysByBank: bankKeys,
        });
        expect(units.length).toBeGreaterThan(8);
        for (const u of units) {
          expect(u.bank).toBe(c.bank);
          expect(bankKeys[u.bank].has(u.key)).toBe(true);
        }
      }
    }
  });
});

describe("Prosody 8-point pitch contour & formant specs", () => {
  it("shapes Mandarin Tone 2 upward, Tone 4 downward, and question endings upward", () => {
    const [uMa2, uMa4] = tokenizeSpeech("麻骂？", { lexicon, validKeys });
    const c2 = solvePitchContour(uMa2, 364, 0.115, { ...DEFAULT_CONFIG, toneAmount: 1.0 });
    const c4 = solvePitchContour(uMa4, 364, 0.115, { ...DEFAULT_CONFIG, toneAmount: 1.0 });

    // Tone 2 rises from start to end
    expect(c2.semitones[7]).toBeGreaterThan(c2.semitones[0]);
    // Question final adds +3st bend on top of Tone 4
    expect(c4.bendSemitones).toBe(3.0);
    expect(c4.playbackRates).toHaveLength(8);
    expect(c4.frequenciesHz).toHaveLength(8);

    // Manual override shifts all 8 contour points identically
    const shifted = solvePitchContour(uMa2, 364, 0.115, DEFAULT_CONFIG, { pitchOffsetSt: 4 });
    const base = solvePitchContour(uMa2, 364, 0.115, DEFAULT_CONFIG);
    expect(shifted.semitones[0] - base.semitones[0]).toBeCloseTo(4, 5);
    expect(shifted.semitones[7] - base.semitones[7]).toBeCloseTo(4, 5);

    // rawDry bypasses all pitch curves and outputs strict 1:1 playbackRate = 1.0
    const dry = solvePitchContour(uMa4, 364, 0.115, { ...DEFAULT_CONFIG, basePitchHz: 520, rawDry: true });
    expect(dry.playbackRates.every((r) => r === 1)).toBe(true);
    expect(dry.semitones.every((s) => s === 0)).toBe(true);
    expect(dry.gainScale).toBe(1);
  });

  it("resolves distinct formants and consonant onsets for Mandarin syllables", () => {
    const sa = resolveFormantSpec("sa");
    const ba = resolveFormantSpec("ba");
    const mi = resolveFormantSpec("mi");
    const qu = resolveFormantSpec("qu");

    expect(sa.onset).toBe("fricative");
    expect(ba.onset).toBe("plosive");
    expect(mi.onset).toBe("nasal");
    // 'qu' uses ü formant (higher F2 than back-rounded 'ku')
    const ku = resolveFormantSpec("ku");
    expect(qu.start[1]).toBeGreaterThan(ku.start[1] + 500);
  });

  it("isolates VoiceBankStore and equal-power audio mix calculations cleanly", async () => {
    const store = new VoiceBankStore();
    store.setHanziLexicon(rawLexicon);
    await store.loadCustomBank(zhManifest, new ArrayBuffer(16), "custom-zh.ogg", "zh");

    const summary = store.getSummary();
    expect(summary.bankLoaded).toBe(true);
    expect(summary.loadedBanks).toEqual(["zh"]);
    expect(summary.bankUnitCount).toBe(Object.keys(zhManifest.units).length);
    expect(store.getBankHz("zh")).toBe(364);

    // Equal-power law preserves constant total power (sample^2 + formant^2 = 1)
    const half = computeEqualPowerGains(0.5, true);
    expect(half.sampleGain ** 2 + half.formantGain ** 2).toBeCloseTo(1.0, 5);
    expect(computeEqualPowerGains(0.0, false)).toEqual({ sampleGain: 0, formantGain: 1 });
  });
});
