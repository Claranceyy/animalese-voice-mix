export {
  VOICE_GAP,
  TEXT_PAUSE,
  LETTER_STEP,
  textRevealSteps,
  VoiceClock,
  simulateVoiceClock,
  type PlaceOptions,
  type TimelineSimulation,
} from "./clock";

export {
  WEAK_PARTICLES,
  PHRASE_MARKS,
  parseHanziLexicon,
  mapEnglishChunk,
  mapLatinChunk,
  tokenizeSpeech,
  type TokenizerOptions,
} from "./tokenizer";

export {
  TONE_CURVES,
  CONTOUR_POINTS,
  deterministicJitter,
  solvePitchContour,
} from "./prosody";

export {
  resolveFormantSpec,
  scheduleFormantSyllable,
  type FormantSpec,
} from "./formant";

export {
  DEFAULT_CONFIG,
  VOICE_PRESETS,
} from "./presets";

export {
  AnimaleseEngine,
  type EngineStatus,
} from "./engine";

export {
  VoiceBankStore,
  type BankStoreSummary,
} from "./audio/bankStore";

export {
  WebAudioPipeline,
  computeEqualPowerGains,
} from "./audio/audioPipeline";

export type {
  ToneNumber,
  UnitScript,
  VoiceBankId,
  LanguageMode,
  JaPitchRole,
  SpeechUnit,
  SyllableOverride,
  VoiceBankManifest,
  DecodedVoiceBank,
  RawHanziLexicon,
  HanziLookupMap,
  EnglishLexicon,
  JapaneseLexicon,
  ProsodyConfig,
  PitchContour,
  DropReason,
  VoiceClockDecision,
  ClockPauseEvent,
  VoicePreset,
  PlaybackCallbacks,
} from "./types";
