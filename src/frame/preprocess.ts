/**
 * Geometry + intensity preprocessing shared by detection alignment, MiniFASNet,
 * and MobileFaceNet input prep.
 *
 * Pure helpers here (no native deps) so they are unit-testable. The actual pixel
 * resampling is done natively by vision-camera-resize-plugin inside the frame
 * processor worklet; these helpers compute the crop rectangles and the affine
 * alignment from landmarks, and provide an illumination-normalization curve used
 * to improve robustness in harsh sunlight / low light / shadows.
 */

import type { BoundingBox, Point2D } from '../types';

/** Expand a bbox by `margin` fraction on each side, clamped to [0,1]. */
export function expandBox(box: BoundingBox, margin: number): BoundingBox {
  const dx = box.width * margin;
  const dy = box.height * margin;
  const x = Math.max(0, box.x - dx);
  const y = Math.max(0, box.y - dy);
  const width = Math.min(1 - x, box.width + 2 * dx);
  const height = Math.min(1 - y, box.height + 2 * dy);
  return { x, y, width, height };
}

/**
 * In-plane rotation (radians) needed to level the eyes, from the two eye-center
 * landmarks. MobileFaceNet expects roughly upright, eye-aligned 112x112 crops;
 * de-rotating by this angle materially improves matching accuracy.
 */
export function eyeAlignmentAngle(leftEye: Point2D, rightEye: Point2D): number {
  return Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
}

/** Whether a detected face is large/centered enough to be usable. */
export function isFaceUsable(
  box: BoundingBox,
  opts: { minWidth?: number; centerTolerance?: number } = {}
): boolean {
  const minWidth = opts.minWidth ?? 0.25; // face must fill >=25% of frame width
  const centerTolerance = opts.centerTolerance ?? 0.3;
  if (box.width < minWidth) return false;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  return (
    Math.abs(cx - 0.5) <= centerTolerance && Math.abs(cy - 0.5) <= centerTolerance
  );
}

/**
 * Build a contrast-limited lookup curve (poor man's CLAHE) for a [0,255] channel
 * given a histogram. Used to normalize illumination before embedding so that
 * harsh-sunlight / shadow frames map closer to training distribution.
 * Returns a 256-entry LUT.
 */
export function buildClaheLut(
  histogram: ArrayLike<number>,
  clipLimit = 0.01,
  totalPixels?: number
): Uint8Array {
  const bins = 256;
  const total =
    totalPixels ?? (Array.from(histogram).reduce((a, b) => a + b, 0) || 1);
  const clip = clipLimit * total;

  // Clip and redistribute excess uniformly.
  const clipped = new Float64Array(bins);
  let excess = 0;
  for (let i = 0; i < bins; i++) {
    const v = histogram[i] ?? 0;
    if (v > clip) {
      excess += v - clip;
      clipped[i] = clip;
    } else {
      clipped[i] = v;
    }
  }
  const redistribute = excess / bins;
  for (let i = 0; i < bins; i++) clipped[i] += redistribute;

  // CDF -> LUT.
  const lut = new Uint8Array(bins);
  let cdf = 0;
  for (let i = 0; i < bins; i++) {
    cdf += clipped[i];
    lut[i] = Math.max(0, Math.min(255, Math.round((cdf / total) * 255)));
  }
  return lut;
}

/** Standard MobileFaceNet normalization: (pixel/255 - 0.5) / 0.5 -> [-1, 1]. */
export function normalizePixel(value0to255: number): number {
  return value0to255 / 127.5 - 1;
}
