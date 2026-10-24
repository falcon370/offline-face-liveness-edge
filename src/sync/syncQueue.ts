/**
 * Offline-first attendance sync queue with purge-after-upload.
 *
 * Field devices operate in zero-network zones, so auth events are persisted
 * locally (encrypted KV) and uploaded opportunistically when connectivity
 * returns. Uploads are idempotent (keyed by record id) so retries can't create
 * duplicates, and *successfully uploaded records are purged from the device* to
 * satisfy the data-minimization requirement.
 *
 * Connectivity is injected (wire to @react-native-community/netinfo in the host)
 * and the uploader is injected (mock now, real AWS later), keeping this testable.
 */

import type { AttendanceRecord } from '../types';
import type { KeyValueStore } from '../storage/enrollmentStore';

/** Uploads a batch, returns ids the server durably accepted. Throws on failure. */
export type RemoteUploader = (batch: AttendanceRecord[]) => Promise<string[]>;

const KEY_PREFIX = 'faceauth:pending:';

export interface SyncQueueOptions {
  /** Max records per upload batch. */
  batchSize?: number;
}

export class SyncQueue {
  private readonly batchSize: number;

  constructor(
    private readonly kv: KeyValueStore,
    private readonly upload: RemoteUploader,
    opts: SyncQueueOptions = {}
  ) {
    this.batchSize = opts.batchSize ?? 50;
  }

  /** Persist an event locally for later sync. */
  enqueue(record: AttendanceRecord): void {
    this.kv.set(KEY_PREFIX + record.id, JSON.stringify(record));
  }

  pendingCount(): number {
    return this.pendingKeys().length;
  }

  /**
   * Attempt to flush all pending records. Returns counts. Purges only the ids
   * the server confirmed. Safe to call repeatedly; a thrown uploader error
   * leaves the unconfirmed records queued for the next attempt.
   */
  async flush(): Promise<{ uploaded: number; remaining: number }> {
    let uploaded = 0;
    const keys = this.pendingKeys();
    for (let i = 0; i < keys.length; i += this.batchSize) {
      const slice = keys.slice(i, i + this.batchSize);
      const batch = slice
        .map((k) => this.kv.getString(k))
        .filter((v): v is string => !!v)
        .map((v) => JSON.parse(v) as AttendanceRecord);
      if (batch.length === 0) continue;

      const acceptedIds = await this.upload(batch); // throws -> stop, keep queued
      const accepted = new Set(acceptedIds);
      for (const rec of batch) {
        if (accepted.has(rec.id)) {
          this.kv.delete(KEY_PREFIX + rec.id); // purge confirmed
          uploaded++;
        }
      }
    }
    return { uploaded, remaining: this.pendingCount() };
  }

  private pendingKeys(): string[] {
    return this.kv.getAllKeys().filter((k) => k.startsWith(KEY_PREFIX));
  }
}
