import type { ClockPauseEvent, DropReason, SpeechUnit, VoiceClockDecision } from "./types";

/** Silence inserted on the voice clock after punctuation marks (seconds). */
export const VOICE_GAP: Readonly<Record<string, number>> = {
  ",": 0.14,
  "，": 0.14,
  "、": 0.14,
  ";": 0.2,
  "；": 0.2,
  ":": 0.2,
  "：": 0.2,
  ".": 0.45,
  "。": 0.45,
  "!": 0.5,
  "！": 0.5,
  "?": 0.5,
  "？": 0.5,
  "…": 0.3,
  "~": 0.1,
  "～": 0.1,
  "っ": 0.055,
  "ッ": 0.055,
  "\n": 0.35,
};

/** Typing pause steps (1 step = 1/20s at 1.0x typing speed) after punctuation marks. */
export const TEXT_PAUSE: Readonly<Record<string, number>> = {
  ",": 5,
  "，": 5,
  "、": 5,
  ";": 6,
  "；": 6,
  ":": 6,
  "：": 6,
  ".": 9,
  "。": 9,
  "!": 9,
  "！": 9,
  "?": 9,
  "？": 9,
  "…": 4,
  "~": 3,
  "～": 3,
  "っ": 1.3,
  "ッ": 1.3,
  "\n": 7,
};

/**
 * Alphabetic characters advance faster than CJK logographs because ~3-4 letters
 * form one spoken syllable, keeping visual typing aligned with speech cadence.
 */
export const LETTER_STEP = 0.4;
export const KANA_STEP = 0.72;
const ALPHABET_RE = /[\p{Script=Latin}\p{Script=Cyrillic}\p{Script=Greek}\s'’-]/u;
const KANA_STEP_RE = /[\u3041-\u3096\u30A1-\u30FAー]/;

export function textRevealSteps(ch: string): number {
  const pause = TEXT_PAUSE[ch];
  if (pause !== undefined) return pause;
  if (ALPHABET_RE.test(ch)) return LETTER_STEP;
  if (KANA_STEP_RE.test(ch)) return KANA_STEP;
  return 1;
}

export interface PlaceOptions {
  weak: boolean;
  isPhraseFinal: boolean;
  unitsPerSecond: number;
  durationSec: number;
  maxLag: number;
  preGapSec?: number;
}

/**
 * Dual-clock scheduler decoupling text reveal from speech onset.
 *
 * The text cursor reveals characters at its own reading rhythm while `VoiceClock`
 * caps syllable onset density (`unitsPerSecond`), honors punctuation silence gaps
 * (`VOICE_GAP`), and sheds weak or non-final syllables if audio lags too far behind.
 */
export class VoiceClock {
  last = -Infinity;
  end = 0;
  free = 0;
  gap = 0;

  constructor(startTime = 0) {
    this.reset(startTime);
  }

  reset(now = 0): void {
    this.last = -Infinity;
    this.end = now;
    this.free = now;
    this.gap = 0;
  }

  /** Record a punctuation pause mark revealed at `now`. */
  mark(ch: string, charIndex: number, now: number): ClockPauseEvent | null {
    const add = VOICE_GAP[ch];
    if (add === undefined || add <= 0) return null;
    this.gap += add;
    this.free = Math.max(this.end, now) + this.gap;
    return {
      charIndex,
      ch,
      revealedAt: now,
      gapSec: add,
      freeAfterAt: this.free,
    };
  }

  /**
   * Compute the exact onset timestamp for a unit revealed at `now`, or drop it
   * when voice lag exceeds `maxLag` (weak) or `2 * maxLag` (non-final).
   */
  place(now: number, unit: Pick<SpeechUnit, "id" | "at" | "surface" | "key">, opts: PlaceOptions): VoiceClockDecision {
    if (opts.preGapSec && opts.preGapSec > 0) {
      this.gap += opts.preGapSec;
      this.free = Math.max(this.end, now) + this.gap;
    }
    const minInterval = 1 / Math.max(1, opts.unitsPerSecond);
    const earliestByRate = Number.isFinite(this.last) ? this.last + minInterval : now;
    const candidate = Math.max(now, earliestByRate, this.free);
    const lag = Math.max(0, candidate - now);
    const appliedGap = this.gap;

    let dropReason: DropReason | null = null;
    if (!opts.isPhraseFinal) {
      if (lag > 2 * opts.maxLag) {
        dropReason = "lag-hard";
      } else if (lag > opts.maxLag && opts.weak) {
        dropReason = "lag-weak";
      }
    }

    if (dropReason !== null) {
      return {
        unitId: unit.id,
        at: unit.at,
        surface: unit.surface,
        key: unit.key,
        revealedAt: now,
        scheduledAt: null,
        duration: opts.durationSec,
        lag,
        gapApplied: appliedGap,
        dropped: true,
        dropReason,
        isWeak: opts.weak,
        isPhraseFinal: opts.isPhraseFinal,
      };
    }

    this.last = candidate;
    this.end = Math.max(this.end, candidate + opts.durationSec);
    this.gap = 0;

    return {
      unitId: unit.id,
      at: unit.at,
      surface: unit.surface,
      key: unit.key,
      revealedAt: now,
      scheduledAt: candidate,
      duration: opts.durationSec,
      lag,
      gapApplied: appliedGap,
      dropped: false,
      dropReason: null,
      isWeak: opts.weak,
      isPhraseFinal: opts.isPhraseFinal,
    };
  }
}

export interface TimelineSimulation {
  totalTextSec: number;
  totalVoiceSec: number;
  charTimes: number[];
  pauses: ClockPauseEvent[];
  decisions: VoiceClockDecision[];
}

/**
 * Pure deterministic offline simulation of the dual-clock timeline and lag-shedding decisions.
 */
export function simulateVoiceClock(
  text: string,
  units: readonly SpeechUnit[],
  options: {
    typingSpeed: number;
    unitsPerSecond: number;
    maxLag: number;
    unitDurationFor: (u: SpeechUnit) => number;
    isMuted?: (u: SpeechUnit) => boolean;
    isWeakOverride?: (u: SpeechUnit) => boolean;
  }
): TimelineSimulation {
  const clock = new VoiceClock(0);
  const byAt = new Map<number, SpeechUnit[]>();
  for (const u of units) {
    const list = byAt.get(u.at);
    if (list) list.push(u);
    else byAt.set(u.at, [u]);
  }

  const rate = Math.max(0.15, options.typingSpeed) * 20;
  const charTimes: number[] = [];
  const pauses: ClockPauseEvent[] = [];
  const decisions: VoiceClockDecision[] = [];

  let t = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    charTimes.push(t);
    const pauseEv = clock.mark(ch, i, t);
    if (pauseEv) pauses.push(pauseEv);

    const unitsAtChar = byAt.get(i);
    if (unitsAtChar) {
      for (const u of unitsAtChar) {
        const dur = options.unitDurationFor(u);
        if (options.isMuted?.(u)) {
          decisions.push({
            unitId: u.id,
            at: u.at,
            surface: u.surface,
            key: u.key,
            revealedAt: t,
            scheduledAt: null,
            duration: dur,
            lag: 0,
            gapApplied: 0,
            dropped: true,
            dropReason: "muted",
            isWeak: options.isWeakOverride ? options.isWeakOverride(u) : u.weak,
            isPhraseFinal: u.isPhraseFinal,
          });
          continue;
        }
        const weak = options.isWeakOverride ? options.isWeakOverride(u) : u.weak;
        const effectiveUps = u.bank === "ja" ? options.unitsPerSecond * 1.22 : options.unitsPerSecond;
        decisions.push(
          clock.place(t, u, {
            weak,
            isPhraseFinal: u.isPhraseFinal,
            unitsPerSecond: effectiveUps,
            durationSec: dur,
            maxLag: options.maxLag,
            preGapSec: u.preGapSec,
          })
        );
      }
    }

    t += textRevealSteps(ch) / rate;
  }

  return {
    totalTextSec: t,
    totalVoiceSec: Math.max(t, clock.end),
    charTimes,
    pauses,
    decisions,
  };
}
