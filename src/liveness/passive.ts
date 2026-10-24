/**
 * Passive (silent) anti-spoofing via MiniFASNet (Silent-Face-Anti-Spoofing).
 *
 * MiniFASNet is a single-image classifier: given an 80x80 face crop it returns a
 * 3-class softmax [fake/print, live, replay/screen] (class order depends on the
 * exact checkpoint — confirm against the bundled model, see models/README.md).
 * The original system fuses two scale variants (1_80x80 and 2.7_80x80); we expose
 * a helper to fuse multiple model outputs by averaging probabilities.
 *
 * This stage runs with NO user action and rejects printed-photo / phone-screen
 * presentation attacks that could otherwise pass the active challenges.
 */

/** Index of the "live/real" class in the MiniFASNet softmax output. */
export const LIVE_CLASS_INDEX = 1;

/** Numerically-stable softmax. */
export function softmax(logits: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  let sum = 0;
  const exps = new Array<number>(logits.length);
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum;
  return exps;
}

/**
 * Fuse one or more MiniFASNet outputs into a single live-probability (0..1).
 * Each input may be raw logits or probabilities; pass `alreadySoftmax` when the
 * model output is already normalized.
 */
export function fusePassiveScores(
  outputs: ArrayLike<number>[],
  opts: { liveIndex?: number; alreadySoftmax?: boolean } = {}
): number {
  const liveIndex = opts.liveIndex ?? LIVE_CLASS_INDEX;
  if (outputs.length === 0) return 0;
  let acc = 0;
  for (const out of outputs) {
    const probs = opts.alreadySoftmax ? Array.from(out) : softmax(out);
    acc += probs[liveIndex] ?? 0;
  }
  return acc / outputs.length;
}
