import { SyncQueue } from '../sync/syncQueue';
import { MockServer } from '../sync/mockEndpoint';
import type { KeyValueStore } from '../storage/enrollmentStore';
import type { AttendanceRecord } from '../types';

/** In-memory KeyValueStore for tests. */
class MemKV implements KeyValueStore {
  private m = new Map<string, string>();
  set(k: string, v: string) { this.m.set(k, v); }
  getString(k: string) { return this.m.get(k); }
  delete(k: string) { this.m.delete(k); }
  getAllKeys() { return [...this.m.keys()]; }
}

function rec(id: string): AttendanceRecord {
  return {
    id,
    personId: 'p1',
    timestamp: new Date().toISOString(),
    verified: true,
    similarity: 0.8,
    liveness: { outcome: 'live', passiveScore: 0.9, completed: ['blink'] },
  };
}

describe('SyncQueue', () => {
  it('uploads then purges confirmed records', async () => {
    const kv = new MemKV();
    const server = new MockServer();
    const q = new SyncQueue(kv, server.uploader({ latencyMs: 0 }));
    q.enqueue(rec('a'));
    q.enqueue(rec('b'));
    expect(q.pendingCount()).toBe(2);

    const r = await q.flush();
    expect(r.uploaded).toBe(2);
    expect(r.remaining).toBe(0);
    expect(q.pendingCount()).toBe(0);
    expect(server.received.size).toBe(2);
  });

  it('keeps records queued when the upload fails', async () => {
    const kv = new MemKV();
    const failing = async () => { throw new Error('offline'); };
    const q = new SyncQueue(kv, failing);
    q.enqueue(rec('a'));
    await expect(q.flush()).rejects.toThrow('offline');
    expect(q.pendingCount()).toBe(1); // not purged
  });

  it('is idempotent across retries (no duplicates server-side)', async () => {
    const kv = new MemKV();
    const server = new MockServer();
    const q = new SyncQueue(kv, server.uploader({ latencyMs: 0 }));
    q.enqueue(rec('dup'));
    await q.flush();
    // Re-enqueue same id (e.g. a double-submit) and flush again.
    q.enqueue(rec('dup'));
    await q.flush();
    expect(server.received.size).toBe(1);
  });
});
