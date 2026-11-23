export type ToneNumber = 0 | 1 | 2 | 3 | 4 | 5;

export type UnitScript = "han" | "digit" | "latin" | "kana";

export type VoiceBankId = "zh" | "en" | "ja";

export type LanguageMode = "auto" | "zh" | "ja" | "en";

export type JaPitchRole = "low" | "rise" | "high" | "fall" | "drop";

export interface SpeechUnit {
  /** Monotonic unit index inside the tokenized utterance */
  id: number;
  /** Zero-based character index in the original text */
  at: number;
  /** Original character or chunk text */
  surface: string;
  /** Target voice bank ("zh" Mandarin, "en" English, "ja" Japanese) */
  bank: VoiceBankId;
  /** Syllable key matching the target bank manifest & formant engine */
  key: string;
  /** 1-4 Mandarin tones, 5 neutral tone, 0 non-tonal */
  tone: ToneNumber;
  /** Source script category */
  script: UnitScript;
  /** Whether this unit carries prosodic stress */
  stress: boolean;
  /** Weak function word eligible for lag shedding */
  weak: boolean;
  /** Optional Japanese bunsetsu pitch-accent role (Tokyo L-H-H-L arch) */
  jaAccent?: JaPitchRole;
  /** Optional micro-silence inserted right before this unit (e.g. Japanese sokuon っ inside Kanji) */
  preGapSec?: number;
  /** Position inside the current phrase */
  pos: number;
  /** Total number of units inside the current phrase */
  of: number;
  /** Zero-based phrase index */
  phraseIndex: number;
  /** Closing punctuation marks attached to this phrase */
  closingPunct: string;
  /** True when among the phrase's last two units (protected from lag dropping) */
  isPhraseFinal: boolean;
}

export interface SyllableOverride {
  /** User semitone offset (-12 .. +12) applied on top of prosody curve */
  pitchOffsetSt?: number;
  /** Override stress flag */
  stress?: boolean;
  /** Override weak flag */
  weak?: boolean;
  /** Skip playback for this unit */
  muted?: boolean;
}

export interface VoiceBankManifest {
  /** Fundamental frequency (Hz) to which every unit in the sprite was pitch-flattened */
  hz: number;
  /** Optional provenance metadata */
  source?: string;
  license?: string;
  voice?: string;
  /** Map from syllable key -> [offsetSeconds, durationSeconds] */
  units: Record<string, [number, number]>;
}

export interface DecodedVoiceBank {
  id: VoiceBankId;
  name: string;
  hz: number;
  units: Map<string, [number, number]>;
  buffer: AudioBuffer;
}

/** Raw JSON mapping `"syllableTone"` (e.g. `"ni3"`) -> concatenated Han characters */
export type RawHanziLexicon = Record<string, string>;

export type HanziLookupMap = Map<string, { key: string; tone: ToneNumber }>;

/** English word -> space-separated `en.json` syllable keys (trailing `*` marks primary stress) */
export type EnglishLexicon = Record<string, string>;

/** Japanese Kanji compound & single-Kanji readings -> Hiragana */
export interface JapaneseLexicon {
  words: Record<string, string>;
  chars: Record<string, string>;
}

export interface ProsodyConfig {
  /** Target speaking pitch in Hz (e.g. 220..620) */
  basePitchHz: number;
  /** Maximum syllable onset rate per second */
  unitsPerSecond: number;
  /** Multiplier applied to the 20 steps/sec typewriter reveal clock */
  typingSpeed: number;
  /** Scaling factor for Mandarin lexical tone contours (0..1.5) */
  toneAmount: number;
  /** Semitone lift for stressed syllables */
  stressSemitones: number;
  /** Seconds behind text cursor before dropping weak units (2x drops non-final units) */
  maxLag: number;
  /**
   * Continuous Voice Mix ratio:
   * 0.0 = 100% Sample Sprite Bank
   * 0.5 = Equal-power 50/50 blend
   * 1.0 = 100% Zero-Asset Web Audio Formant Synth
   */
  mixRatio: number;
  /** Master lowpass filter cutoff (Hz) for tonal warmth */
  warmthHz: number;
  /** Master output gain (0..1) */
  volume: number;
  /** Language routing mode for Kanji disambiguation ("auto" detects Kana per clause) */
  languageMode?: LanguageMode;
  /**
   * True 1:1 dry bypass mode: plays raw `.ogg` slices at exact `playbackRate = 1.0`
   * with zero prosody curves, zero tone bends, zero formant mix, and transparent filter.
   */
  rawDry?: boolean;
}

export interface PitchContour {
  /** 8-point semitone offsets relative to basePitchHz */
  semitones: number[];
  /** 8-point playbackRate multipliers relative to bankHz */
  playbackRates: number[];
  /** 8-point absolute target fundamental frequencies in Hz */
  frequenciesHz: number[];
  /** Starting semitone offset before tone curve */
  baseSemitone: number;
  /** End-of-syllable pitch bend in semitones (e.g. +3 on question final) */
  bendSemitones: number;
  /** Per-unit amplitude multiplier (e.g. 1.26 for exclamation) */
  gainScale: number;
  /** Effective playback duration in seconds */
  durationSec: number;
}

export type DropReason = "lag-weak" | "lag-hard" | "muted" | "missing-key";

export interface VoiceClockDecision {
  unitId: number;
  at: number;
  surface: string;
  key: string;
  revealedAt: number;
  scheduledAt: number | null;
  duration: number;
  lag: number;
  gapApplied: number;
  dropped: boolean;
  dropReason: DropReason | null;
  isWeak: boolean;
  isPhraseFinal: boolean;
}

export interface ClockPauseEvent {
  charIndex: number;
  ch: string;
  revealedAt: number;
  gapSec: number;
  freeAfterAt: number;
}

export interface VoicePreset {
  id: string;
  name: string;
  enName: string;
  description: string;
  config: Partial<ProsodyConfig>;
}

export interface PlaybackCallbacks {
  onRevealChar?: (charIndex: number, revealedCount: number, elapsedSec: number) => void;
  onPauseMark?: (event: ClockPauseEvent) => void;
  onUnitScheduled?: (unit: SpeechUnit, decision: VoiceClockDecision, contour: PitchContour) => void;
  onUnitPlay?: (unit: SpeechUnit, contour: PitchContour, elapsedSec: number) => void;
  onUnitDropped?: (unit: SpeechUnit, decision: VoiceClockDecision) => void;
  onComplete?: () => void;
}
