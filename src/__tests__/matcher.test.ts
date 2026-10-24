import {
  l2Normalize,
  cosineSimilarity,
  averageTemplate,
  identify,
  verifyAgainst,
} from '../recognition/matcher';
import type { EnrolledIdentity } from '../types';

const v = (...xs: number[]) => Float32Array.from(xs);

describe('matcher', () => {
  it('l2Normalize produces a unit vector', () => {
    const n = l2Normalize(v(3, 4));
    expect(cosineSimilarity(n, n)).toBeCloseTo(1, 6);
    expect(Math.hypot(n[0], n[1])).toBeCloseTo(1, 6);
  });

  it('cosineSimilarity: identical=1, orthogonal=0, opposite=-1', () => {
    const a = l2Normalize(v(1, 0));
    const b = l2Normalize(v(0, 1));
    const c = l2Normalize(v(-1, 0));
    expect(cosineSimilarity(a, a)).toBeCloseTo(1, 6);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0, 6);
    expect(cosineSimilarity(a, c)).toBeCloseTo(-1, 6);
  });

  it('cosineSimilarity throws on dimension mismatch', () => {
    expect(() => cosineSimilarity(v(1, 2), v(1, 2, 3))).toThrow();
  });

  it('averageTemplate averages then normalizes', () => {
    const t = averageTemplate([l2Normalize(v(1, 0)), l2Normalize(v(0, 1))]);
    expect(Math.hypot(t[0], t[1])).toBeCloseTo(1, 6);
    expect(t[0]).toBeCloseTo(t[1], 6);
  });

  it('identify returns best match above threshold', () => {
    const gallery: EnrolledIdentity[] = [
      { personId: 'alice', template: l2Normalize(v(1, 0, 0)), enrolledAt: '', sampleCount: 1 },
      { personId: 'bob', template: l2Normalize(v(0, 1, 0)), enrolledAt: '', sampleCount: 1 },
    ];
    const probe = l2Normalize(v(0.95, 0.05, 0));
    const r = identify(probe, gallery, 0.6);
    expect(r.matched).toBe(true);
    expect(r.personId).toBe('alice');
  });

  it('identify rejects when below threshold', () => {
    const gallery: EnrolledIdentity[] = [
      { personId: 'alice', template: l2Normalize(v(1, 0, 0)), enrolledAt: '', sampleCount: 1 },
    ];
    const probe = l2Normalize(v(0, 1, 0));
    const r = identify(probe, gallery, 0.6);
    expect(r.matched).toBe(false);
    expect(r.personId).toBeNull();
  });

  it('verifyAgainst respects the threshold', () => {
    const id: EnrolledIdentity = {
      personId: 'x', template: l2Normalize(v(1, 0)), enrolledAt: '', sampleCount: 1,
    };
    expect(verifyAgainst(l2Normalize(v(0.99, 0.01)), id, 0.6).matched).toBe(true);
    expect(verifyAgainst(l2Normalize(v(0.4, 0.6)), id, 0.9).matched).toBe(false);
  });
});
