import { softmax, fusePassiveScores, LIVE_CLASS_INDEX } from '../liveness/passive';

describe('passive anti-spoof scoring', () => {
  it('softmax sums to 1 and is monotonic', () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[2]).toBeGreaterThan(p[1]);
    expect(p[1]).toBeGreaterThan(p[0]);
  });

  it('softmax is numerically stable for large logits', () => {
    const p = softmax([1000, 1001, 1002]);
    expect(p.every((x) => Number.isFinite(x))).toBe(true);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it('fusePassiveScores averages live probability across models', () => {
    // Two models, both confident the sample is live (index 1).
    const score = fusePassiveScores(
      [
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1],
      ],
      { alreadySoftmax: true, liveIndex: LIVE_CLASS_INDEX }
    );
    expect(score).toBeCloseTo(0.75, 6);
  });

  it('fusePassiveScores applies softmax to raw logits', () => {
    const score = fusePassiveScores([[0, 5, 0]]); // strongly live
    expect(score).toBeGreaterThan(0.9);
  });
});
