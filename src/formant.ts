import type { PitchContour } from "./types";

export interface FormantSpec {
  /** Starting formant frequencies [F1, F2, F3] in Hz */
  start: [number, number, number];
  /** Ending formant frequencies [F1, F2, F3] in Hz (for diphthongs / glides) */
  end: [number, number, number];
  /** Consonant onset class */
  onset: "vowel" | "plosive" | "fricative" | "nasal" | "affricate";
  /** Characteristic noise band frequency for consonant transient */
  burstHz: number;
}

const INITIALS = [
  "zh", "ch", "sh",
  "b", "p", "m", "f", "d", "t", "n", "l",
  "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w",
] as const;

const VOWEL_FORMANTS: Record<string, { start: [number, number, number]; end?: [number, number, number] }> = {
  a: { start: [760, 1240, 2600] },
  o: { start: [520, 920, 2480] },
  e: { start: [500, 1460, 2520] },
  i: { start: [300, 2280, 3020] },
  u: { start: [330, 780, 2360] },
  v: { start: [310, 1880, 2360] },
  er: { start: [490, 1340, 1740] },
  ai: { start: [740, 1220, 2580], end: [340, 2180, 2920] },
  ei: { start: [480, 1760, 2620], end: [310, 2260, 2980] },
  ao: { start: [720, 1140, 2520], end: [400, 840, 2400] },
  ou: { start: [510, 920, 2460], end: [340, 780, 2360] },
  an: { start: [720, 1320, 2560], end: [520, 1520, 2500] },
  en: { start: [490, 1440, 2500], end: [410, 1560, 2480] },
  ang: { start: [740, 1120, 2520], end: [560, 1020, 2440] },
  eng: { start: [490, 1320, 2480], end: [420, 1160, 2420] },
  ong: { start: [420, 820, 2380], end: [360, 760, 2320] },
  ia: { start: [310, 2220, 2960], end: [740, 1280, 2600] },
  ie: { start: [310, 2220, 2960], end: [520, 1860, 2680] },
  iao: { start: [310, 2200, 2940], end: [420, 880, 2420] },
  iu: { start: [310, 2200, 2940], end: [350, 820, 2380] },
  ian: { start: [320, 2180, 2920], end: [580, 1660, 2580] },
  in: { start: [300, 2260, 3000], end: [340, 1980, 2780] },
  iang: { start: [320, 2160, 2920], end: [680, 1180, 2520] },
  ing: { start: [300, 2260, 2980], end: [360, 1780, 2680] },
  iong: { start: [310, 2140, 2860], end: [370, 820, 2360] },
  ua: { start: [340, 800, 2380], end: [750, 1220, 2580] },
  uo: { start: [350, 800, 2380], end: [520, 940, 2480] },
  uai: { start: [350, 820, 2380], end: [360, 2120, 2860] },
  ui: { start: [350, 820, 2380], end: [320, 2200, 2920] },
  uan: { start: [350, 820, 2380], end: [620, 1460, 2540] },
  un: { start: [340, 800, 2380], end: [420, 1460, 2480] },
  uang: { start: [350, 820, 2380], end: [680, 1120, 2500] },
  ve: { start: [310, 1860, 2380], end: [520, 1780, 2560] },
  ue: { start: [310, 1860, 2380], end: [520, 1780, 2560] },
};

/** Resolve acoustic formant targets and consonant burst class for any pinyin syllable key. */
export function resolveFormantSpec(key: string): FormantSpec {
  const k = key.toLowerCase();
  let initial = "";
  let final = k;

  for (const cand of INITIALS) {
    if (k.startsWith(cand) && k.length > cand.length) {
      initial = cand;
      final = k.slice(cand.length);
      break;
    }
  }

  // Handle j/q/x/y + u -> ü ("v") acoustic color, and y/w glides
  if (initial === "y") {
    if (final === "i" || final === "in" || final === "ing") {
      // keep i-nucleus
    } else if (final === "u" || final === "ue" || final === "uan" || final === "un") {
      final = final === "u" ? "v" : final === "ue" ? "ve" : "v";
    } else if (!final.startsWith("i")) {
      final = `i${final}`;
    }
  } else if (initial === "w") {
    if (final === "u") {
      // keep u
    } else if (!final.startsWith("u")) {
      final = `u${final}`;
    }
  } else if ((initial === "j" || initial === "q" || initial === "x") && final.startsWith("u")) {
    final = final === "u" ? "v" : final === "ue" ? "ve" : "v";
  }

  const vowel =
    VOWEL_FORMANTS[final] ??
    VOWEL_FORMANTS[final.slice(-2)] ??
    VOWEL_FORMANTS[final.slice(-1)] ??
    VOWEL_FORMANTS.a;

  let onset: FormantSpec["onset"] = "vowel";
  let burstHz = 1800;

  if (["b", "p", "d", "t", "g", "k"].includes(initial)) {
    onset = "plosive";
    burstHz = initial === "b" || initial === "p" ? 1100 : initial === "d" || initial === "t" ? 2900 : 1850;
  } else if (["f", "s", "sh", "x", "h"].includes(initial)) {
    onset = "fricative";
    burstHz = initial === "s" ? 4800 : initial === "sh" || initial === "x" ? 3100 : 2100;
  } else if (["z", "c", "zh", "ch", "j", "q"].includes(initial)) {
    onset = "affricate";
    burstHz = initial === "z" || initial === "c" ? 4200 : 2800;
  } else if (["m", "n", "l", "r"].includes(initial)) {
    onset = "nasal";
    burstHz = 950;
  }

  return {
    start: [...vowel.start] as [number, number, number],
    end: [...(vowel.end ?? vowel.start)] as [number, number, number],
    onset,
    burstHz,
  };
}

let cachedNoiseBuf: AudioBuffer | null = null;
let cachedNoiseRate = 0;

function getNoiseBuffer(ctx: AudioContext): AudioBuffer {
  if (cachedNoiseBuf && cachedNoiseRate === ctx.sampleRate) return cachedNoiseBuf;
  const len = Math.floor(ctx.sampleRate * 0.4);
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < len; i++) {
    data[i] = Math.random() * 2 - 1;
  }
  cachedNoiseBuf = buf;
  cachedNoiseRate = ctx.sampleRate;
  return buf;
}

/**
 * Synthesize one cute Animal-Crossing-style syllable purely with Web Audio API
 * (zero external audio files), tracking the exact 8-point `contour.frequenciesHz`.
 */
export function scheduleFormantSyllable(
  ctx: AudioContext,
  destination: AudioNode,
  key: string,
  contour: PitchContour,
  startAt: number,
  gainLevel: number,
  basePitchHz: number
): AudioNode[] {
  const nodes: AudioNode[] = [];
  if (gainLevel <= 0.0005) return nodes;

  const spec = resolveFormantSpec(key);
  const dur = Math.max(0.055, Math.min(0.22, contour.durationSec));
  const endAt = startAt + dur;
  // Gentle vocal-tract scaling so high-pitched cat voices stay crisp and bright
  const tractScale = Math.max(0.76, Math.min(1.38, (basePitchHz / 364) ** 0.24));

  const outGain = ctx.createGain();
  const amp = gainLevel * contour.gainScale * 0.44;
  const attack = spec.onset === "plosive" ? 0.008 : spec.onset === "fricative" ? 0.016 : 0.011;
  const release = Math.min(0.028, dur * 0.34);

  outGain.gain.setValueAtTime(0.0001, startAt);
  outGain.gain.exponentialRampToValueAtTime(Math.max(0.001, amp), startAt + attack);
  outGain.gain.setValueAtTime(Math.max(0.001, amp), Math.max(startAt + attack + 0.004, endAt - release));
  outGain.gain.exponentialRampToValueAtTime(0.0001, endAt);
  outGain.connect(destination);
  nodes.push(outGain);

  // Dual glottal source: harmonic-rich sawtooth + warm triangle reinforcement
  const oscMain = ctx.createOscillator();
  const oscWarm = ctx.createOscillator();
  const oscMixMain = ctx.createGain();
  const oscMixWarm = ctx.createGain();

  oscMain.type = "sawtooth";
  oscWarm.type = "triangle";
  oscMixMain.gain.value = 0.68;
  oscMixWarm.gain.value = 0.42;

  const freqCurve = new Float32Array(contour.frequenciesHz.map((f) => Math.max(60, Math.min(2600, f))));
  oscMain.frequency.setValueCurveAtTime(freqCurve, startAt, dur);
  oscWarm.frequency.setValueCurveAtTime(freqCurve, startAt, dur);

  oscMain.connect(oscMixMain);
  oscWarm.connect(oscMixWarm);

  // 3 parallel vowel formant resonators (F1, F2, F3)
  const formantWeights = [1.0, 0.72, 0.34] as const;
  const formantQs = [5.4, 7.2, 8.8] as const;
  const nyquist = ctx.sampleRate * 0.45;

  for (let i = 0; i < 3; i++) {
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.Q.value = formantQs[i];
    const fStart = Math.max(120, Math.min(nyquist, spec.start[i] * tractScale));
    const fEnd = Math.max(120, Math.min(nyquist, spec.end[i] * tractScale));
    bp.frequency.setValueAtTime(fStart, startAt);
    bp.frequency.exponentialRampToValueAtTime(fEnd, endAt);

    const fg = ctx.createGain();
    fg.gain.value = formantWeights[i];

    oscMixMain.connect(bp);
    oscMixWarm.connect(bp);
    bp.connect(fg);
    fg.connect(outGain);
  }

  oscMain.start(startAt);
  oscWarm.start(startAt);
  oscMain.stop(endAt + 0.015);
  oscWarm.stop(endAt + 0.015);
  nodes.push(oscMain, oscWarm);

  // Consonant transient noise burst (gives syllables realistic crisp articulation)
  if (spec.onset !== "vowel" && spec.onset !== "nasal") {
    const noise = ctx.createBufferSource();
    noise.buffer = getNoiseBuffer(ctx);
    const nFilter = ctx.createBiquadFilter();
    nFilter.type = "bandpass";
    nFilter.frequency.value = Math.min(nyquist, spec.burstHz * tractScale);
    nFilter.Q.value = spec.onset === "fricative" ? 2.4 : 1.4;

    const nGain = ctx.createGain();
    const burstDur = spec.onset === "plosive" ? 0.014 : spec.onset === "affricate" ? 0.022 : 0.028;
    const burstAmp = amp * (spec.onset === "fricative" ? 0.28 : 0.22);
    nGain.gain.setValueAtTime(0.0001, startAt);
    nGain.gain.exponentialRampToValueAtTime(Math.max(0.001, burstAmp), startAt + 0.003);
    nGain.gain.exponentialRampToValueAtTime(0.0001, startAt + burstDur);

    noise.connect(nFilter);
    nFilter.connect(nGain);
    nGain.connect(destination);

    noise.start(startAt, Math.random() * 0.25, burstDur + 0.01);
    nodes.push(noise);
  }

  return nodes;
}
