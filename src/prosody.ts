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
  low: -0.9,
  rise: 1.45,
  high: 1.05,
  fall: 0.85,
  drop: -0.95,
};

const JA_ROLE_CURVE: Readonly<Record<JaPitchRole, readonly number[]>> = {
  low: [-0.4, 0.25],
  rise: [-0.15, 0.55],
  high: [0.2, 0.0],
  fall: [0.45, -1.15],
  drop: [-0.15, -0.55],
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
  let gainScale = isExclamation ? 1.26 : 1.0;

  if (unit.bank === "ja") {
    // Authentic Japanese per-word Tokyo pitch accent (L -> H -> H -> accent drop) + gentle phrase tilt
    const role: JaPitchRole = unit.jaAccent ?? (unit.pos === 0 ? "low" : unit.weak ? "drop" : "high");
    phraseShape = JA_ROLE_BASE_ST[role] - 0.045 * unit.pos;
    if (role === "drop") gainScale *= 0.86;
    if (role === "rise") gainScale *= 1.06;
  } else if (unit.bank === "en") {
    // English lexical stress bounces clearly above unstressed schwas
    phraseShape = (unit.pos === 0 ? 0.5 : 0) - 0.05 * unit.pos + (stressed ? 0.7 : -0.45);
    if (stressed) gainScale *= 1.08;
  }

  let st =
    phraseShape +
    deterministicJitter(unit.id) +
    (isExclamation ? 1.0 : 0) +
    (stressed ? config.stressSemitones : 0) +
    (override?.pitchOffsetSt ?? 0);

  let bend = 0;
  if (isQuestion && isPenultimate) st += 1.5;
  if (isQuestion && isLast) {
    st += 2.0;
    bend = 3.0;
  }
  if (isEllipsis && isLast) st -= 0.8;

  const useJaCurve = unit.bank === "ja" && unit.tone === 0;
  const jaRole: JaPitchRole = unit.jaAccent ?? (unit.pos === 0 ? "low" : unit.weak ? "drop" : "high");
  const toneCurve = useJaCurve ? JA_ROLE_CURVE[jaRole] : TONE_CURVES[unit.tone] ?? TONE_CURVES[0];
  const curveScale = useJaCurve ? Math.max(0.55, config.toneAmount * 1.45) : config.toneAmount;
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

  return {
    semitones,
    playbackRates,
    frequenciesHz,
    baseSemitone: st,
    bendSemitones: bend,
    gainScale,
    durationSec: rawUnitDurSec / baseRate,
  };
}
