/**
 * Mock AWS endpoint for the prototype.
 *
 * Simulates an API Gateway / Lambda that accepts a batch of attendance records
 * and returns the set of accepted (idempotency) ids. Swapping in the real AWS
 * endpoint is a one-line change: replace this uploader with a `fetch()` to the
 * deployed URL (see RemoteUploader contract in syncQueue.ts).
 */

import type { AttendanceRecord } from '../types';
import type { RemoteUploader } from './syncQueue';

export interface MockEndpointOptions {
  /** Simulated round-trip latency in ms. */
  latencyMs?: number;
  /** 0..1 probability a batch fails (to exercise retry). */
  failureRate?: number;
}

/** Records the server "received" — inspectable in the demo/tests. */
export class MockServer {
  readonly received = new Map<string, AttendanceRecord>();

  uploader(opts: MockEndpointOptions = {}): RemoteUploader {
    const latency = opts.latencyMs ?? 150;
    const failureRate = opts.failureRate ?? 0;
    return async (batch: AttendanceRecord[]): Promise<string[]> => {
      await new Promise((r) => setTimeout(r, latency));
      if (Math.random() < failureRate) {
        throw new Error('mock-endpoint: simulated upload failure');
      }
      for (const rec of batch) this.received.set(rec.id, rec); // idempotent by id
      return batch.map((r) => r.id);
    };
  }
}
