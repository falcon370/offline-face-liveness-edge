import { EnrollmentStore, type KeyValueStore } from '../storage/enrollmentStore';
import { l2Normalize } from '../recognition/matcher';

class MemKV implements KeyValueStore {
  private m = new Map<string, string>();
  set(k: string, v: string) { this.m.set(k, v); }
  getString(k: string) { return this.m.get(k); }
  delete(k: string) { this.m.delete(k); }
  getAllKeys() { return [...this.m.keys()]; }
}

describe('EnrollmentStore', () => {
  it('round-trips an embedding exactly', () => {
    const store = new EnrollmentStore(new MemKV());
    const template = l2Normalize(Float32Array.from([0.1, -0.2, 0.3, 0.9]));
    store.save({ personId: 'p1', template, enrolledAt: '2026-05-29', sampleCount: 5 });

    const got = store.get('p1');
    expect(got).not.toBeNull();
    expect(got!.personId).toBe('p1');
    expect(got!.sampleCount).toBe(5);
    expect(Array.from(got!.template)).toEqual(Array.from(template));
  });

  it('lists and removes identities', () => {
    const store = new EnrollmentStore(new MemKV());
    const t = l2Normalize(Float32Array.from([1, 0]));
    store.save({ personId: 'a', template: t, enrolledAt: '', sampleCount: 1 });
    store.save({ personId: 'b', template: t, enrolledAt: '', sampleCount: 1 });
    expect(store.list().map((i) => i.personId).sort()).toEqual(['a', 'b']);
    store.remove('a');
    expect(store.list().map((i) => i.personId)).toEqual(['b']);
  });

  it('skips and self-heals a corrupt record instead of throwing', () => {
    const kv = new MemKV();
    const store = new EnrollmentStore(kv);
    const t = l2Normalize(Float32Array.from([1, 0]));
    store.save({ personId: 'good', template: t, enrolledAt: '', sampleCount: 1 });
    // A malformed (non-JSON) value under an identity key — e.g. a partial write.
    kv.set('faceauth:identity:bad', 'jZ9-not-json');

    // list() must not throw, returns only the good record...
    const ids = store.list();
    expect(ids.map((i) => i.personId)).toEqual(['good']);
    // ...and the corrupt key is dropped so it can't keep breaking reads.
    expect(kv.getString('faceauth:identity:bad')).toBeUndefined();
    // get() on a corrupt key returns null rather than throwing.
    kv.set('faceauth:identity:bad2', 'jXX');
    expect(store.get('bad2')).toBeNull();
  });
});
