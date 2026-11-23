import type { JaPitchRole, PitchContour, ProsodyConfig, SpeechUnit, SyllableOverride, ToneNumber } from "./types";

/**
 * Relative semitone contours for the 4 Mandarin lexical tones + neutral tone (5).
 * Interpolated across 8 control points and scaled by `config.toneAmount`.
 */
export const TONE_CURVES: Readonly<Record<ToneNumber, readonly number[]>> = {
  0: [0, 0],
  1: [0.8, 0.8],
  2: [-0.6, 1.6],
  3: [-1.0, -1.8, -0.6],
  4: [1.6, -1.2],
  5: [-0.8, -0.8],
};

const JA_ROLE_BASE_ST: Readonly<Record<JaPitchRole, number>> = {
  low: -0.36,
  rise: 0.44,
  high: 0.34,
  fall: 0.10,
  drop: -0.42,
};

const JA_ROLE_CURVE: Readonly<Record<JaPitchRole, readonly number[]>> = {
  low: [-0.05, 0.05],
  rise: [-0.04, 0.10],
  high: [0.03, 0.0],
  fall: [0.06, -0.16],
  drop: [-0.04, -0.12],
};

export const CONTOUR_POINTS = 8;

/**
 * Deterministic per-unit micro-jitter in [-0.2, +0.2] semitones seeded by unit index,
 * avoiding robotic pitch uniformity while keeping solved contours strictly reproducible.
 */
export function deterministicJitter(unitId: number, seed = 1337): number {
  let x = (unitId * 374761393 + seed * 668265263) >>> 0;
  x = ((x ^ (x >>> 13)) * 1274126177) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return (x / 0xffffffff - 0.5) * 0.4;
}

/**
 * Solve the 8-point semitone, playbackRate, and Hz pitch contour for a single syllable.
 */
export function solvePitchContour(
  unit: SpeechUnit,
  bankHz: number,
  rawUnitDurSec: number,
  config: ProsodyConfig,
  override?: SyllableOverride
): PitchContour {
  const safeBankHz = Math.max(80, bankHz);

  // True 1:1 raw dry bypass: completely flat playbackRate = 1.0, zero prosody or tone curves
  if (config.rawDry) {
    return {
      semitones: new Array<number>(CONTOUR_POINTS).fill(0),
      playbackRates: new Array<number>(CONTOUR_POINTS).fill(1),
      frequenciesHz: new Array<number>(CONTOUR_POINTS).fill(safeBankHz),
      baseSemitone: 0,
      bendSemitones: 0,
      gainScale: 1.0,
      durationSec: rawUnitDurSec,
    };
  }

  const safeBaseHz = Math.max(80, config.basePitchHz);
  const baseRate = safeBaseHz / safeBankHz;

  const isQuestion = /[?？]/.test(unit.closingPunct);
  const isExclamation = /[!！]/.test(unit.closingPunct);
  const isEllipsis = /…/.test(unit.closingPunct);
  const isLast = unit.pos === unit.of - 1;
  const isPenultimate = unit.pos === unit.of - 2;
  const stressed = override?.stress !== undefined ? override.stress : unit.stress;

  // Language-aware phrase intonation envelope
  let phraseShape = (unit.pos === 0 ? 0.8 : 0) - 0.06 * unit.pos;
  let gainScale = isExclamation ? 1.14 : 1.0;
  const isEnCoda = unit.bank === "en" && unit.key.startsWith("_");

  if (unit.bank === "ja") {
    // Japanese Doubutsugo cadence: steady flat-topped mora vowels with subtle bunsetsu steps
    const role: JaPitchRole = unit.jaAccent ?? (unit.pos === 0 ? "low" : unit.weak ? "drop" : "high");
    phraseShape = JA_ROLE_BASE_ST[role] - 0.022 * Math.min(12, unit.pos);
    if (unit.surface === "ー") {
      phraseShape -= 0.14;
      gainScale *= 0.90;
    }
    if (unit.key === "n") {
      gainScale *= 0.88;
    } else if (role === "drop") {
      gainScale *= 0.93;
    } else if (role === "rise") {
      gainScale *= 1.04;
    }
  } else if (unit.bank === "en") {
    if (isEnCoda) {
      phraseShape = -0.35 - 0.025 * Math.min(12, unit.pos);
      gainScale *= 0.62;
    } else {
      // Single clean lexical stress contrast without double-counting stressSemitones
      phraseShape = (unit.pos === 0 ? 0.35 : 0) - 0.028 * Math.min(12, unit.pos) + (stressed ? 0.0 : -0.32);
      if (stressed) gainScale *= 1.05;
    }
  }

  const jitterScale = unit.bank === "ja" || isEnCoda ? 0.55 : 1.0;
  const stressBoost = stressed ? config.stressSemitones * (unit.bank === "en" ? 0.78 : 1.0) : 0;
  let st =
    phraseShape +
    deterministicJitter(unit.id) * jitterScale +
    (isExclamation ? (unit.bank === "en" ? 0.65 : 1.0) : 0) +
    stressBoost +
    (override?.pitchOffsetSt ?? 0);

  let bend = 0;
  if (isQuestion && isPenultimate) st += unit.bank === "ja" ? 0.9 : 1.5;
  if (isQuestion && isLast) {
    st += unit.bank === "ja" ? 1.4 : 2.0;
    bend = unit.bank === "ja" ? 2.1 : 3.0;
  }
  if (isEllipsis && isLast) st -= 0.8;

  const useJaCurve = unit.bank === "ja" && unit.tone === 0;
  const jaRole: JaPitchRole = unit.jaAccent ?? (unit.pos === 0 ? "low" : unit.weak ? "drop" : "high");
  const toneCurve = useJaCurve ? JA_ROLE_CURVE[jaRole] : TONE_CURVES[unit.tone] ?? TONE_CURVES[0];
  const curveScale = useJaCurve ? Math.max(0.4, config.toneAmount) : config.toneAmount;
  const semitones: number[] = [];
  const playbackRates: number[] = [];
  const frequenciesHz: number[] = [];

  for (let k = 0; k < CONTOUR_POINTS; k++) {
    const p = k / (CONTOUR_POINTS - 1);
    const pos = p * (toneCurve.length - 1);
    const a = Math.floor(pos);
    const b = Math.min(toneCurve.length - 1, a + 1);
    const toneSt = (toneCurve[a] + (toneCurve[b] - toneCurve[a]) * (pos - a)) * curveScale;
    const totalSt = st + toneSt + bend * p;
    const ratio = 2 ** (totalSt / 12);
    semitones.push(totalSt);
    playbackRates.push(baseRate * ratio);
    frequenciesHz.push(safeBaseHz * ratio);
  }

  const moraDurScale = unit.bank === "ja" && unit.key === "n" ? 0.86 : isEnCoda ? 0.78 : 1.0;
  return {
    semitones,
    playbackRates,
    frequenciesHz,
    baseSemitone: st,
    bendSemitones: bend,
    gainScale,
    durationSec: (rawUnitDurSec / baseRate) * moraDurScale,
  };
}
