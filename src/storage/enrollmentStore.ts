/**
 * Encrypted local gallery of enrolled identities.
 *
 * Backed by react-native-mmkv with an encryption key (templates are biometric
 * data and must be encrypted at rest). Embeddings are stored base64-encoded.
 * The store is fully offline; nothing leaves the device here — sync of
 * *attendance events* (not templates) is handled separately by syncQueue.
 *
 * The MMKV instance is injected so this module is testable with an in-memory
 * backend and has no hard native import.
 */

import type { EnrolledIdentity, Embedding } from '../types';

/** Minimal structural type matching react-native-mmkv's instance. */
export interface KeyValueStore {
  set(key: string, value: string): void;
  getString(key: string): string | undefined;
  delete(key: string): void;
  getAllKeys(): string[];
}

const KEY_PREFIX = 'faceauth:identity:';

function encodeEmbedding(e: Embedding): string {
  const bytes = new Uint8Array(e.buffer, e.byteOffset, e.byteLength);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  // base64 — global btoa is polyfilled in RN via the bundled base-64; fall back.
  return typeof btoa === 'function' ? btoa(bin) : Buffer.from(bytes).toString('base64');
}

function decodeEmbedding(s: string): Embedding {
  const bin =
    typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('binary');
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export class EnrollmentStore {
  constructor(private readonly kv: KeyValueStore) {}

  save(identity: EnrolledIdentity): void {
    const payload = JSON.stringify({
      personId: identity.personId,
      template: encodeEmbedding(identity.template),
      templates: identity.templates?.map(encodeEmbedding),
      enrolledAt: identity.enrolledAt,
      sampleCount: identity.sampleCount,
    });
    this.kv.set(KEY_PREFIX + identity.personId, payload);
  }

  get(personId: string): EnrolledIdentity | null {
    const key = KEY_PREFIX + personId;
    const raw = this.kv.getString(key);
    if (!raw) return null;
    return this.tryDeserialize(key, raw);
  }

  list(): EnrolledIdentity[] {
    return this.kv
      .getAllKeys()
      .filter((k) => k.startsWith(KEY_PREFIX))
      .map((k) => {
        const v = this.kv.getString(k);
        return v ? this.tryDeserialize(k, v) : null;
      })
      .filter((x): x is EnrolledIdentity => x !== null);
  }

  /**
   * Deserialize one record, tolerating corruption: a malformed/legacy value must
   * not crash the gallery (and with it the whole app). We drop the bad key so the
   * store self-heals on next read.
   */
  private tryDeserialize(key: string, raw: string): EnrolledIdentity | null {
    try {
      return this.deserialize(raw);
    } catch {
      this.kv.delete(key);
      return null;
    }
  }

  remove(personId: string): void {
    this.kv.delete(KEY_PREFIX + personId);
  }

  private deserialize(raw: string): EnrolledIdentity {
    const o = JSON.parse(raw) as {
      personId: string;
      template: string;
      templates?: string[];
      enrolledAt: string;
      sampleCount: number;
    };
    return {
      personId: o.personId,
      template: decodeEmbedding(o.template),
      templates: o.templates?.map(decodeEmbedding),
      enrolledAt: o.enrolledAt,
      sampleCount: o.sampleCount,
    };
  }
}
