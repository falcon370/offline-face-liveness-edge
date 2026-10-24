import {
  extractBlendshapes,
  headPoseFromMatrix,
  boxFromLandmarks,
  toFaceFrame,
} from '../frame/mediapipeAdapter';

describe('mediapipeAdapter', () => {
  it('extractBlendshapes maps category names', () => {
    const bs = extractBlendshapes([
      { categoryName: 'eyeBlinkLeft', score: 0.7 },
      { categoryName: 'mouthSmileRight', score: 0.4 },
    ]);
    expect(bs.eyeBlinkLeft).toBe(0.7);
    expect(bs.mouthSmileRight).toBe(0.4);
    expect(bs.jawOpen).toBe(0); // missing -> 0
  });

  it('headPoseFromMatrix returns ~0 for identity', () => {
    const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const p = headPoseFromMatrix(I);
    expect(p.yaw).toBeCloseTo(0, 4);
    expect(p.pitch).toBeCloseTo(0, 4);
    expect(p.roll).toBeCloseTo(0, 4);
  });

  it('headPoseFromMatrix detects a yaw rotation', () => {
    // 90° yaw about Y (row-major 3x3): yaw = atan2(-r20, sqrt(r00^2+r10^2))
    const deg = 30 * (Math.PI / 180);
    const cy = Math.cos(deg);
    const sy = Math.sin(deg);
    const m = [cy, 0, sy, 0, -sy, 0, cy]; // 3x3 rotation about Y, partial
    // Build a proper 3x3 Ry
    const Ry = [cy, 0, sy, 0, 1, 0, -sy, 0, cy];
    expect(Ry.length).toBe(9);
    const p = headPoseFromMatrix(Ry);
    expect(Math.abs(p.yaw)).toBeGreaterThan(20);
    void m;
  });

  it('boxFromLandmarks computes the bounding box', () => {
    const box = boxFromLandmarks([
      { x: 0.2, y: 0.3 },
      { x: 0.6, y: 0.9 },
      { x: 0.4, y: 0.5 },
    ]);
    expect(box.x).toBeCloseTo(0.2);
    expect(box.y).toBeCloseTo(0.3);
    expect(box.width).toBeCloseTo(0.4);
    expect(box.height).toBeCloseTo(0.6);
  });

  it('toFaceFrame marks facePresent false for empty results', () => {
    const f = toFaceFrame(null, 123);
    expect(f.facePresent).toBe(false);
    expect(f.timestampMs).toBe(123);
  });
});
