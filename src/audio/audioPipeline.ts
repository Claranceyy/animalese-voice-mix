import { scheduleFormantSyllable } from "../formant";
import type { DecodedVoiceBank, PitchContour, ProsodyConfig, SpeechUnit } from "../types";

/**
 * Encapsulates the Web Audio node graph (VoiceBus -> Lowpass Warmth -> MasterGain
 * -> DynamicsCompressor -> Analyser -> Destination) and schedules individual
 * Sample / Formant syllables onto the audio timeline.
 */
export class WebAudioPipeline {
  private ctx: AudioContext | null = null;
  private voiceBus: GainNode | null = null;
  private warmthFilter: BiquadFilterNode | null = null;
  private masterGain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private activeSources = new Set<AudioNode>();

  getContext(): AudioContext | null {
    return this.ctx;
  }

  getCurrentTime(): number {
    return this.ctx?.currentTime ?? 0;
  }

  ensureContext(config: ProsodyConfig): AudioContext | null {
    if (typeof window === "undefined") return null;
    if (!this.ctx) {
      const AudioCtor =
        window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtor) return null;

      const ctx = new AudioCtor();
      const voiceBus = ctx.createGain();
      const warmth = ctx.createBiquadFilter();
      warmth.type = "lowpass";
      warmth.frequency.value = config.rawDry ? 20000 : config.warmthHz;
      warmth.Q.value = 0.65;

      const master = ctx.createGain();
      master.gain.value = config.volume * 0.9;

      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -12;
      comp.knee.value = 18;
      comp.ratio.value = 4;

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.78;

      voiceBus.connect(warmth);
      warmth.connect(master);
      master.connect(comp);
      comp.connect(analyser);
      analyser.connect(ctx.destination);

      this.ctx = ctx;
      this.voiceBus = voiceBus;
      this.warmthFilter = warmth;
      this.masterGain = master;
      this.analyser = analyser;
    }
    if (this.ctx.state === "suspended") {
      void this.ctx.resume();
    }
    return this.ctx;
  }

  applyConfig(config: ProsodyConfig): void {
    if (!this.ctx) return;
    if (this.warmthFilter) {
      const targetHz = config.rawDry ? 20000 : config.warmthHz;
      const clamped = Math.max(800, Math.min(20000, targetHz));
      this.warmthFilter.frequency.setTargetAtTime(clamped, this.ctx.currentTime, 0.02);
    }
    if (this.masterGain) {
      const vol = Math.max(0, Math.min(1, config.volume));
      this.masterGain.gain.setTargetAtTime(vol * 0.9, this.ctx.currentTime, 0.02);
    }
  }

  readWaveform(out: Float32Array<ArrayBuffer>): boolean {
    if (!this.analyser) return false;
    this.analyser.getFloatTimeDomainData(out);
    return true;
  }

  readSpectrum(out: Uint8Array<ArrayBuffer>): boolean {
    if (!this.analyser) return false;
    this.analyser.getByteFrequencyData(out);
    return true;
  }

  /** Schedule one syllable onto the Web Audio bus with continuous equal-power Sample ↔ Formant mixing. */
  scheduleUnitAudio(
    unit: SpeechUnit,
    contour: PitchContour,
    startAt: number,
    config: ProsodyConfig,
    resolved: { bank: DecodedVoiceBank; slice: [number, number] } | null
  ): void {
    const ctx = this.ctx;
    const bus = this.voiceBus;
    if (!ctx || !bus) return;

    const { sampleGain, formantGain } = config.rawDry
      ? { sampleGain: resolved ? 1 : 0, formantGain: 0 }
      : computeEqualPowerGains(config.mixRatio, resolved !== null);

    if (sampleGain > 0.001 && resolved) {
      const [offset, length] = resolved.slice;
      const src = ctx.createBufferSource();
      src.buffer = resolved.bank.buffer;
      if (config.rawDry) {
        src.playbackRate.setValueAtTime(1.0, startAt);
      } else {
        src.playbackRate.setValueCurveAtTime(new Float32Array(contour.playbackRates), startAt, contour.durationSec);
      }

      const g = ctx.createGain();
      g.gain.value = 0.56 * sampleGain * contour.gainScale;

      src.connect(g);
      g.connect(bus);
      src.start(startAt, offset, length);
      this.activeSources.add(src);
      src.onended = () => this.activeSources.delete(src);
    }

    if (formantGain > 0.001) {
      const fNodes = scheduleFormantSyllable(ctx, bus, unit.key, contour, startAt, formantGain, config.basePitchHz);
      for (const node of fNodes) {
        this.activeSources.add(node);
        if ("onended" in node) {
          (node as AudioScheduledSourceNode).onended = () => this.activeSources.delete(node);
        }
      }
    }
  }

  stopAllSources(): void {
    for (const node of this.activeSources) {
      try {
        if ("stop" in node && typeof (node as AudioScheduledSourceNode).stop === "function") {
          (node as AudioScheduledSourceNode).stop();
        }
      } catch {
        // already stopped
      }
      try {
        node.disconnect();
      } catch {
        // ignore
      }
    }
    this.activeSources.clear();
  }
}

export function computeEqualPowerGains(mixRatio: number, hasSample: boolean): { sampleGain: number; formantGain: number } {
  if (!hasSample) return { sampleGain: 0, formantGain: 1 };
  const mix = Math.max(0, Math.min(1, mixRatio));
  if (mix <= 0.001) return { sampleGain: 1, formantGain: 0 };
  if (mix >= 0.999) return { sampleGain: 0, formantGain: 1 };
  const angle = mix * (Math.PI / 2);
  return {
    sampleGain: Math.cos(angle),
    formantGain: Math.sin(angle),
  };
}
