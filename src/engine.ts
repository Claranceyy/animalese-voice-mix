import { WebAudioPipeline } from "./audio/audioPipeline";
import { VoiceBankStore } from "./audio/bankStore";
import { VoiceClock, textRevealSteps } from "./clock";
import { DEFAULT_CONFIG } from "./presets";
import { solvePitchContour } from "./prosody";
import { tokenizeSpeech } from "./tokenizer";
import type {
  EnglishLexicon,
  JapaneseLexicon,
  PitchContour,
  PlaybackCallbacks,
  ProsodyConfig,
  RawHanziLexicon,
  SpeechUnit,
  SyllableOverride,
  VoiceBankId,
  VoiceBankManifest,
  VoiceClockDecision,
} from "./types";

export interface EngineStatus {
  bankLoaded: boolean;
  loadedBanks: VoiceBankId[];
  bankName: string;
  bankUnitCount: number;
  lexiconSize: number;
  playing: boolean;
}

/**
 * Clean-room MIT Animal-Crossing-style prosody engine facade.
 * Composes `VoiceBankStore`, `WebAudioPipeline`, and `VoiceClock` with high cohesion
 * and strict separation of concerns.
 */
export class AnimaleseEngine {
  private readonly banks = new VoiceBankStore();
  private readonly audio = new WebAudioPipeline();
  private readonly clock = new VoiceClock(0);

  private config: ProsodyConfig = { ...DEFAULT_CONFIG };
  private timers = new Set<number>();
  private rafId: number | null = null;
  private playing = false;

  // Live typing stream state
  private liveText = "";
  private liveRevealed = 0;
  private liveUnits: SpeechUnit[] = [];
  private liveOverrides = new Map<number, SyllableOverride>();

  getConfig(): ProsodyConfig {
    return { ...this.config };
  }

  configure(patch: Partial<ProsodyConfig>): void {
    this.config = { ...this.config, ...patch };
    this.audio.applyConfig(this.config);
  }

  getStatus(): EngineStatus {
    return {
      ...this.banks.getSummary(),
      playing: this.playing,
    };
  }

  getBankHz(bankId: VoiceBankId = "zh"): number {
    return this.banks.getBankHz(bankId);
  }

  getRawUnitDuration(key: string, bankId: VoiceBankId = "zh"): number {
    return this.banks.getRawUnitDuration(key, bankId);
  }

  getValidKeysByBank(): Partial<Record<VoiceBankId, ReadonlySet<string>>> {
    return this.banks.getValidKeysByBank();
  }

  async loadBank(id: VoiceBankId, manifestUrl: string, audioUrl: string, bankName?: string): Promise<void> {
    await this.banks.loadBank(id, manifestUrl, audioUrl, bankName, this.audio.getContext());
  }

  async loadAssets(paths: {
    manifestUrl: string;
    audioUrl: string;
    lexiconUrl: string;
    bankName?: string;
    extraBanks?: Partial<Record<VoiceBankId, { manifestUrl: string; audioUrl: string; bankName?: string }>>;
  }): Promise<void> {
    await this.banks.loadAssets(paths, this.audio.getContext());
  }

  async loadTrilingualAssets(baseDir = "./speech"): Promise<void> {
    await this.banks.loadTrilingualAssets(baseDir, this.audio.getContext());
  }

  async loadCustomBank(
    manifest: VoiceBankManifest,
    audioBytes: ArrayBuffer,
    name: string,
    id: VoiceBankId = "zh"
  ): Promise<void> {
    const ctx = this.ensureAudioContext();
    await this.banks.loadCustomBank(manifest, audioBytes, name, id, ctx);
  }

  setLexicon(raw: RawHanziLexicon): void {
    this.banks.setHanziLexicon(raw);
  }

  setEnglishLexicon(lex: EnglishLexicon): void {
    this.banks.enLexicon = lex;
  }

  setJapaneseLexicon(lex: JapaneseLexicon): void {
    this.banks.jaLexicon = lex;
  }

  tokenize(text: string): SpeechUnit[] {
    return tokenizeSpeech(text, {
      lexicon: this.banks.lexicon,
      enLexicon: this.banks.enLexicon,
      jaLexicon: this.banks.jaLexicon,
      languageMode: this.config.languageMode ?? "auto",
      validKeysByBank: this.config.mixRatio >= 0.999 ? undefined : this.banks.getValidKeysByBank(),
    });
  }

  solveContour(unit: SpeechUnit, override?: SyllableOverride): PitchContour {
    return solvePitchContour(
      unit,
      this.getBankHz(unit.bank),
      this.getRawUnitDuration(unit.key, unit.bank),
      this.config,
      override
    );
  }

  ensureAudioContext(): AudioContext | null {
    return this.audio.ensureContext(this.config);
  }

  async unlock(): Promise<void> {
    const ctx = this.ensureAudioContext();
    if (ctx) {
      await this.banks.decodeAll(ctx);
    }
  }

  readWaveform(out: Float32Array<ArrayBuffer>): boolean {
    return this.audio.readWaveform(out);
  }

  readSpectrum(out: Uint8Array<ArrayBuffer>): boolean {
    return this.audio.readSpectrum(out);
  }

  getAudioTime(): number {
    return this.audio.getCurrentTime();
  }

  private scheduleUnitWithCallback(
    unit: SpeechUnit,
    contour: PitchContour,
    startAt: number,
    onPlay?: () => void
  ): void {
    this.audio.scheduleUnitAudio(unit, contour, startAt, this.config, this.banks.resolveSample(unit));
    const ctx = this.audio.getContext();
    if (onPlay && ctx) {
      const delayMs = Math.max(0, (startAt - ctx.currentTime) * 1000);
      const timerId = window.setTimeout(() => {
        this.timers.delete(timerId);
        onPlay();
      }, delayMs);
      this.timers.add(timerId);
    }
  }

  async auditionUnit(unit: SpeechUnit, override?: SyllableOverride, onPlay?: () => void): Promise<void> {
    if (override?.muted) return;
    await this.unlock();
    const ctx = this.audio.getContext();
    if (!ctx) return;
    const contour = this.solveContour(unit, override);
    this.scheduleUnitWithCallback(unit, contour, ctx.currentTime + 0.012, onPlay);
  }

  async playText(
    text: string,
    overrides: ReadonlyMap<number, SyllableOverride>,
    callbacks: PlaybackCallbacks = {}
  ): Promise<void> {
    this.stop();
    await this.unlock();
    const ctx = this.audio.getContext();
    if (!ctx || !text) {
      callbacks.onComplete?.();
      return;
    }

    const units = this.tokenize(text);
    const byAt = groupUnitsByCharIndex(units);
    const startAudioTime = ctx.currentTime + 0.03;
    this.clock.reset(startAudioTime);
    this.playing = true;

    let shown = 0;
    let acc = 0;
    let lastFrameMs = performance.now();

    const step = (nowMs: number) => {
      if (!this.playing) return;
      const dt = Math.min(0.08, Math.max(0, (nowMs - lastFrameMs) / 1000));
      lastFrameMs = nowMs;

      acc += dt * 20 * Math.max(0.15, this.config.typingSpeed);
      while (acc >= 1 && shown < text.length) {
        const charIdx = shown++;
        const ch = text[charIdx];
        acc -= textRevealSteps(ch);

        const audioNow = Math.max(startAudioTime, ctx.currentTime + 0.02);
        const elapsedSec = Math.max(0, audioNow - startAudioTime);
        callbacks.onRevealChar?.(charIdx, shown, elapsedSec);

        const pauseEv = this.clock.mark(ch, charIdx, audioNow);
        if (pauseEv) {
          callbacks.onPauseMark?.({
            ...pauseEv,
            revealedAt: pauseEv.revealedAt - startAudioTime,
            freeAfterAt: pauseEv.freeAfterAt - startAudioTime,
          });
        }

        const charUnits = byAt.get(charIdx);
        if (charUnits) {
          for (const u of charUnits) {
            this.placeAndScheduleUnit(u, overrides.get(u.id), audioNow, startAudioTime, elapsedSec, callbacks);
          }
        }
      }

      if (shown < text.length) {
        this.rafId = window.requestAnimationFrame(step);
      } else {
        const remainingMs = Math.max(80, (this.clock.end - ctx.currentTime) * 1000 + 80);
        const endTimer = window.setTimeout(() => {
          this.timers.delete(endTimer);
          if (this.playing) {
            this.playing = false;
            callbacks.onComplete?.();
          }
        }, remainingMs);
        this.timers.add(endTimer);
      }
    };

    this.rafId = window.requestAnimationFrame(step);
  }

  resetLiveStream(): void {
    this.stop();
    this.liveText = "";
    this.liveRevealed = 0;
    this.liveUnits = [];
    this.liveOverrides.clear();
  }

  async feedLiveText(
    nextText: string,
    callbacks: Omit<PlaybackCallbacks, "onComplete"> = {},
    streamOriginSec?: number
  ): Promise<{ units: SpeechUnit[]; revealed: number }> {
    await this.unlock();
    const ctx = this.audio.getContext();
    if (!ctx) return { units: this.liveUnits, revealed: this.liveRevealed };

    if (!nextText.startsWith(this.liveText)) {
      this.liveText = nextText;
      this.liveRevealed = nextText.length;
      this.liveUnits = this.tokenize(nextText);
      this.clock.reset(ctx.currentTime + 0.015);
      return { units: this.liveUnits, revealed: this.liveRevealed };
    }

    const startIdx = this.liveRevealed;
    this.liveText = nextText;
    this.liveUnits = this.tokenize(nextText);

    const byAt = groupUnitsByCharIndex(this.liveUnits, startIdx);
    const nowAudio = ctx.currentTime + 0.018;
    if (this.clock.end < nowAudio - 2.0) {
      this.clock.reset(nowAudio);
    }
    const origin = streamOriginSec ?? nowAudio;
    const elapsedSec = Math.max(0, nowAudio - origin);

    for (let i = startIdx; i < nextText.length; i++) {
      const ch = nextText[i];
      callbacks.onRevealChar?.(i, i + 1, elapsedSec);

      const pauseEv = this.clock.mark(ch, i, nowAudio);
      if (pauseEv) {
        callbacks.onPauseMark?.({
          ...pauseEv,
          revealedAt: Math.max(0, pauseEv.revealedAt - origin),
          freeAfterAt: Math.max(0, pauseEv.freeAfterAt - origin),
        });
      }

      const charUnits = byAt.get(i);
      if (charUnits) {
        for (const u of charUnits) {
          this.placeAndScheduleUnit(u, this.liveOverrides.get(u.id), nowAudio, origin, elapsedSec, callbacks);
        }
      }
    }

    this.liveRevealed = nextText.length;
    return { units: this.liveUnits, revealed: this.liveRevealed };
  }

  private placeAndScheduleUnit(
    u: SpeechUnit,
    ov: SyllableOverride | undefined,
    audioNow: number,
    originTime: number,
    elapsedSec: number,
    callbacks: Omit<PlaybackCallbacks, "onComplete">
  ): void {
    const contour = this.solveContour(u, ov);
    const isWeak = ov?.weak !== undefined ? ov.weak : u.weak;

    if (ov?.muted) {
      callbacks.onUnitDropped?.(u, {
        unitId: u.id,
        at: u.at,
        surface: u.surface,
        key: u.key,
        revealedAt: elapsedSec,
        scheduledAt: null,
        duration: contour.durationSec,
        lag: 0,
        gapApplied: 0,
        dropped: true,
        dropReason: "muted",
        isWeak,
        isPhraseFinal: u.isPhraseFinal,
      });
      return;
    }

    const decision = this.clock.place(audioNow, u, {
      weak: isWeak,
      isPhraseFinal: u.isPhraseFinal,
      unitsPerSecond: !this.config.rawDry && u.bank === "ja" ? this.config.unitsPerSecond * 1.22 : this.config.unitsPerSecond,
      durationSec: contour.durationSec,
      maxLag: this.config.maxLag,
      preGapSec: u.preGapSec,
    });

    const normalizedDecision: VoiceClockDecision = {
      ...decision,
      revealedAt: Math.max(0, decision.revealedAt - originTime),
      scheduledAt: decision.scheduledAt !== null ? Math.max(0, decision.scheduledAt - originTime) : null,
    };

    if (decision.dropped || decision.scheduledAt === null) {
      callbacks.onUnitDropped?.(u, normalizedDecision);
    } else {
      callbacks.onUnitScheduled?.(u, normalizedDecision, contour);
      const playAtRel = normalizedDecision.scheduledAt ?? elapsedSec;
      this.scheduleUnitWithCallback(u, contour, decision.scheduledAt, () => {
        callbacks.onUnitPlay?.(u, contour, playAtRel);
      });
    }
  }

  stop(): void {
    this.playing = false;
    if (this.rafId !== null) {
      window.cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    for (const id of this.timers) {
      window.clearTimeout(id);
    }
    this.timers.clear();
    this.audio.stopAllSources();
    this.clock.reset(this.audio.getCurrentTime());
  }
}

function groupUnitsByCharIndex(units: readonly SpeechUnit[], minCharIndex = 0): Map<number, SpeechUnit[]> {
  const byAt = new Map<number, SpeechUnit[]>();
  for (const u of units) {
    if (u.at < minCharIndex) continue;
    const list = byAt.get(u.at);
    if (list) list.push(u);
    else byAt.set(u.at, [u]);
  }
  return byAt;
}
