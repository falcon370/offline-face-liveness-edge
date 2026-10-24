/**
 * Face matching: L2 normalization, cosine similarity, template averaging, and
 * 1:N gallery search. MobileFaceNet embeddings are compared by cosine distance;
 * a threshold around 0.6 separates same-identity from different-identity pairs
 * (tune per deployment against a labeled validation set — see docs/TECHNICAL.md).
 */

import type { Embedding, EnrolledIdentity, VerificationResult } from '../types';

/** Default cosine-similarity decision threshold for a positive match. */
export const DEFAULT_MATCH_THRESHOLD = 0.6;

/** Return a new L2-normalized copy of an embedding (unit length). */
export function l2Normalize(v: Embedding): Embedding {
  let sumSq = 0;
  for (let i = 0; i < v.length; i++) sumSq += v[i] * v[i];
  const norm = Math.sqrt(sumSq) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

/**
 * Cosine similarity in [-1, 1]. Assumes inputs are already L2-normalized
 * (the embedder normalizes on output), so this is a plain dot product.
 */
export function cosineSimilarity(a: Embedding, b: Embedding): number {
  if (a.length !== b.length) {
    throw new Error(
      `Embedding length mismatch: ${a.length} vs ${b.length}`
    );
  }
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/**
 * Average a set of enrollment embeddings into a single normalized template.
 * Averaging multiple captures (varied pose/lighting) yields a more robust
 * template than any single frame.
 */
export function averageTemplate(samples: Embedding[]): Embedding {
  if (samples.length === 0) throw new Error('Cannot build template from 0 samples');
  const dim = samples[0].length;
  const acc = new Float32Array(dim);
  for (const s of samples) {
    if (s.length !== dim) throw new Error('Inconsistent embedding dimensions');
    for (let i = 0; i < dim; i++) acc[i] += s[i];
  }
  for (let i = 0; i < dim; i++) acc[i] /= samples.length;
  return l2Normalize(acc);
}

/**
 * Best cosine similarity of a probe against an identity. If the identity has
 * per-pose templates (multi-angle enrollment), we take the MAX over them so an
 * off-frontal probe matches whichever enrolled pose is closest; otherwise we fall
 * back to the single averaged template.
 */
export function identitySimilarity(probe: Embedding, identity: EnrolledIdentity): number {
  const set =
    identity.templates && identity.templates.length > 0
      ? identity.templates
      : [identity.template];
  let best = -Infinity;
  for (const t of set) {
    const sim = cosineSimilarity(probe, t);
    if (sim > best) best = sim;
  }
  return best;
}

/**
 * 1:1 verification against a known identity's template(s).
 */
export function verifyAgainst(
  probe: Embedding,
  identity: EnrolledIdentity,
  threshold = DEFAULT_MATCH_THRESHOLD
): VerificationResult {
  const similarity = identitySimilarity(probe, identity);
  return {
    matched: similarity >= threshold,
    personId: similarity >= threshold ? identity.personId : null,
    similarity,
    threshold,
  };
}

/**
 * 1:N identification against the local gallery. Returns the best match above
 * threshold, or a non-match with the closest similarity for diagnostics.
 */
export function identify(
  probe: Embedding,
  gallery: EnrolledIdentity[],
  threshold = DEFAULT_MATCH_THRESHOLD
): VerificationResult {
  let best: { id: string; sim: number } | null = null;
  for (const id of gallery) {
    const sim = identitySimilarity(probe, id);
    if (best === null || sim > best.sim) best = { id: id.personId, sim };
  }
  if (best === null) {
    return { matched: false, personId: null, similarity: -1, threshold };
  }
  return {
    matched: best.sim >= threshold,
    personId: best.sim >= threshold ? best.id : null,
    similarity: best.sim,
    threshold,
  };
}
