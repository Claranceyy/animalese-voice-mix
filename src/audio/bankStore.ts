import { parseHanziLexicon } from "../tokenizer";
import type {
  DecodedVoiceBank,
  EnglishLexicon,
  HanziLookupMap,
  JapaneseLexicon,
  RawHanziLexicon,
  SpeechUnit,
  VoiceBankId,
  VoiceBankManifest,
} from "../types";

export const DEFAULT_BANK_HZ = 364;
export const DEFAULT_UNIT_DUR = 0.115;

interface PendingBankData {
  id: VoiceBankId;
  name: string;
  manifest: VoiceBankManifest;
  rawAudioBytes: ArrayBuffer;
}

export interface BankStoreSummary {
  bankLoaded: boolean;
  loadedBanks: VoiceBankId[];
  bankName: string;
  bankUnitCount: number;
  lexiconSize: number;
}

/**
 * Isolated store for trilingual audio sprite banks (`zh`, `en`, `ja`) and
 * pronunciation lexicons. Handles HTTP fetching, custom bank injection, and
 * lazy `AudioContext.decodeAudioData` caching.
 */
export class VoiceBankStore {
  lexicon: HanziLookupMap = new Map();
  enLexicon: EnglishLexicon | null = null;
  jaLexicon: JapaneseLexicon | null = null;

  private pendingBanks = new Map<VoiceBankId, PendingBankData>();
  private decodedBanks = new Map<VoiceBankId, DecodedVoiceBank>();
  private decodingPromises = new Map<VoiceBankId, Promise<void>>();

  getSummary(): BankStoreSummary {
    const ids = new Set<VoiceBankId>([...this.pendingBanks.keys(), ...this.decodedBanks.keys()]);
    const loadedBanks = (["zh", "en", "ja"] as VoiceBankId[]).filter((id) => ids.has(id));
    let totalUnits = 0;
    const names: string[] = [];

    for (const id of loadedBanks) {
      const dec = this.decodedBanks.get(id);
      const pend = this.pendingBanks.get(id);
      if (dec) {
        totalUnits += dec.units.size;
        names.push(dec.name);
      } else if (pend) {
        totalUnits += Object.keys(pend.manifest.units).length;
        names.push(pend.name);
      }
    }

    return {
      bankLoaded: loadedBanks.length > 0,
      loadedBanks,
      bankName: names.join(" + ") || "Formant Synth",
      bankUnitCount: totalUnits,
      lexiconSize: this.lexicon.size,
    };
  }

  getBankHz(bankId: VoiceBankId = "zh"): number {
    return (
      this.decodedBanks.get(bankId)?.hz ??
      this.pendingBanks.get(bankId)?.manifest.hz ??
      this.decodedBanks.get("zh")?.hz ??
      this.pendingBanks.get("zh")?.manifest.hz ??
      DEFAULT_BANK_HZ
    );
  }

  getRawUnitDuration(key: string, bankId: VoiceBankId = "zh"): number {
    const dec = this.decodedBanks.get(bankId)?.units.get(key);
    if (dec) return dec[1];
    const pend = this.pendingBanks.get(bankId)?.manifest.units[key];
    if (pend) return pend[1];
    return DEFAULT_UNIT_DUR;
  }

  getValidKeysByBank(): Partial<Record<VoiceBankId, ReadonlySet<string>>> {
    const out: Partial<Record<VoiceBankId, ReadonlySet<string>>> = {};
    for (const id of ["zh", "en", "ja"] as VoiceBankId[]) {
      const dec = this.decodedBanks.get(id);
      if (dec) {
        out[id] = new Set(dec.units.keys());
      } else {
        const pend = this.pendingBanks.get(id);
        if (pend) out[id] = new Set(Object.keys(pend.manifest.units));
      }
    }
    return out;
  }

  setHanziLexicon(raw: RawHanziLexicon): void {
    this.lexicon = parseHanziLexicon(raw);
  }

  async loadBank(
    id: VoiceBankId,
    manifestUrl: string,
    audioUrl: string,
    bankName?: string,
    ctx?: AudioContext | null
  ): Promise<void> {
    const [manifestRes, audioRes] = await Promise.all([fetch(manifestUrl), fetch(audioUrl)]);
    if (!manifestRes.ok) throw new Error(`${id} manifest HTTP ${manifestRes.status}`);
    if (!audioRes.ok) throw new Error(`${id} audio HTTP ${audioRes.status}`);
    const [manifest, rawAudioBytes] = (await Promise.all([
      manifestRes.json(),
      audioRes.arrayBuffer(),
    ])) as [VoiceBankManifest, ArrayBuffer];

    this.setPendingBank(id, bankName ?? `${id}.ogg`, manifest, rawAudioBytes);
    if (ctx) {
      await this.ensureDecoded(id, ctx);
    }
  }

  async loadAssets(
    paths: {
      manifestUrl: string;
      audioUrl: string;
      lexiconUrl: string;
      bankName?: string;
      extraBanks?: Partial<Record<VoiceBankId, { manifestUrl: string; audioUrl: string; bankName?: string }>>;
    },
    ctx?: AudioContext | null
  ): Promise<void> {
    const jobs: Promise<unknown>[] = [
      this.loadBank("zh", paths.manifestUrl, paths.audioUrl, paths.bankName ?? "zh.ogg", ctx),
      fetch(paths.lexiconUrl).then(async (r) => {
        if (!r.ok) throw new Error(`Lexicon HTTP ${r.status}`);
        this.lexicon = parseHanziLexicon((await r.json()) as RawHanziLexicon);
      }),
    ];

    if (paths.extraBanks) {
      for (const [id, info] of Object.entries(paths.extraBanks) as [
        VoiceBankId,
        { manifestUrl: string; audioUrl: string; bankName?: string },
      ][]) {
        if (!info) continue;
        jobs.push(
          this.loadBank(id, info.manifestUrl, info.audioUrl, info.bankName ?? `${id}.ogg`, ctx).catch((err) => {
            console.warn(`Optional ${id} bank not loaded:`, err);
          })
        );
      }
    }

    await Promise.all(jobs);
  }

  async loadTrilingualAssets(baseDir = "./speech", ctx?: AudioContext | null): Promise<void> {
    const base = baseDir.replace(/\/+$/, "");
    await Promise.all([
      this.loadAssets(
        {
          manifestUrl: `${base}/zh.json`,
          audioUrl: `${base}/zh.ogg`,
          lexiconUrl: `${base}/zh-lexicon.json`,
          bankName: "zh.ogg",
          extraBanks: {
            en: { manifestUrl: `${base}/en.json`, audioUrl: `${base}/en.ogg`, bankName: "en.ogg" },
            ja: { manifestUrl: `${base}/ja.json`, audioUrl: `${base}/ja.ogg`, bankName: "ja.ogg" },
          },
        },
        ctx
      ),
      fetch(`${base}/en-lexicon.json`)
        .then(async (r) => {
          if (r.ok) this.enLexicon = (await r.json()) as EnglishLexicon;
        })
        .catch(() => {}),
      fetch(`${base}/ja-lexicon.json`)
        .then(async (r) => {
          if (r.ok) this.jaLexicon = (await r.json()) as JapaneseLexicon;
        })
        .catch(() => {}),
    ]);
  }

  async loadCustomBank(
    manifest: VoiceBankManifest,
    audioBytes: ArrayBuffer,
    name: string,
    id: VoiceBankId = "zh",
    ctx?: AudioContext | null
  ): Promise<void> {
    if (!manifest || typeof manifest.hz !== "number" || !manifest.units) {
      throw new Error("Invalid voice bank manifest: expected { hz: number, units: Record<string, [offset, len]> }");
    }
    this.setPendingBank(id, name, manifest, audioBytes.slice(0));
    if (ctx) {
      await this.ensureDecoded(id, ctx);
    }
  }

  async decodeAll(ctx: AudioContext): Promise<void> {
    const ids = Array.from(this.pendingBanks.keys());
    await Promise.all(ids.map((id) => this.ensureDecoded(id, ctx)));
  }

  resolveSample(unit: SpeechUnit): { bank: DecodedVoiceBank; slice: [number, number] } | null {
    const primary = this.decodedBanks.get(unit.bank);
    const primarySlice = primary?.units.get(unit.key);
    if (primary && primarySlice) return { bank: primary, slice: primarySlice };

    const zh = this.decodedBanks.get("zh");
    const zhSlice = zh?.units.get(unit.key);
    if (zh && zhSlice) return { bank: zh, slice: zhSlice };
    return null;
  }

  private setPendingBank(id: VoiceBankId, name: string, manifest: VoiceBankManifest, rawAudioBytes: ArrayBuffer): void {
    this.pendingBanks.set(id, { id, name, manifest, rawAudioBytes });
    this.decodedBanks.delete(id);
    this.decodingPromises.delete(id);
  }

  private async ensureDecoded(id: VoiceBankId, ctx: AudioContext): Promise<void> {
    if (this.decodedBanks.has(id)) return;
    const pending = this.pendingBanks.get(id);
    if (!pending) return;

    let job = this.decodingPromises.get(id);
    if (!job) {
      const bytes = pending.rawAudioBytes.slice(0);
      job = ctx
        .decodeAudioData(bytes)
        .then((buffer) => {
          this.decodedBanks.set(id, {
            id,
            name: pending.name,
            hz: pending.manifest.hz,
            units: new Map(Object.entries(pending.manifest.units)),
            buffer,
          });
        })
        .catch((err) => {
          console.error(`Failed to decode ${id} voice bank:`, err);
        });
      this.decodingPromises.set(id, job);
    }
    await job;
  }
}
