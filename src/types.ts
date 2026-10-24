/**
 * Shared types for the offline face-auth pipeline.
 *
 * The pipeline has three on-device stages — Detect → Liveness → Recognize —
 * plus enrollment/matching and an offline-first sync queue. These types are the
 * contract between the pure-logic modules (challenge engine, matcher, sync) and
 * the native glue (MediaPipe landmarks, TFLite inference).
 */

/** A 512-d (or 128-d) L2-normalized face embedding produced by MobileFaceNet. */
export type Embedding = Float32Array;

/** Normalized image-space point in [0,1] range (origin top-left). */
export interface Point2D {
  x: number;
  y: number;
}

/** Axis-aligned face bounding box in normalized [0,1] image coordinates. */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * The subset of MediaPipe Face Landmarker blendshape coefficients (0..1) we use
 * to drive active-liveness challenges. Names match MediaPipe's category names.
 */
export interface Blendshapes {
  eyeBlinkLeft: number;
  eyeBlinkRight: number;
  mouthSmileLeft: number;
  mouthSmileRight: number;
  jawOpen: number;
}

/** Head pose in degrees, derived from the MediaPipe face transform matrix. */
export interface HeadPose {
  /** Left/right rotation. Positive = subject turns to their right. */
  yaw: number;
  /** Up/down rotation. */
  pitch: number;
  /** Tilt. */
  roll: number;
}

/** One sample of facial state from a single camera frame. */
export interface FaceFrame {
  timestampMs: number;
  box: BoundingBox;
  /** Key landmarks used for alignment (eyes, nose, mouth corners). */
  landmarks: Point2D[];
  blendshapes: Blendshapes;
  headPose: HeadPose;
  /** True when a single, sufficiently large, well-centered face is present. */
  facePresent: boolean;
}

/** The randomized challenge types for active liveness. */
export type ChallengeType = 'blink' | 'smile' | 'turnLeft' | 'turnRight';

export type ChallengeStatus = 'pending' | 'active' | 'passed' | 'failed';

export interface ChallengeState {
  type: ChallengeType;
  status: ChallengeStatus;
  /** 0..1 progress toward completing the current challenge. */
  progress: number;
}

export type LivenessOutcome = 'inProgress' | 'live' | 'spoof' | 'timeout';

export interface LivenessResult {
  outcome: LivenessOutcome;
  /** Passive anti-spoof score (0..1, higher = more likely live). */
  passiveScore: number;
  /** Which active challenges were completed. */
  completed: ChallengeType[];
}

/** A person enrolled into the local gallery. */
export interface EnrolledIdentity {
  personId: string;
  /** Mean L2-normalized embedding over all enrollment captures (1:1 / fallback). */
  template: Embedding;
  /**
   * Per-pose L2-normalized templates captured during the multi-angle enrollment
   * sweep (e.g. center/left/right/up/down). Matching takes the BEST (max cosine)
   * of these, so an off-frontal probe — chin lifted, head turned — still matches.
   * Optional for backward compatibility with older single-template records.
   */
  templates?: Embedding[];
  /** ISO timestamp of enrollment. */
  enrolledAt: string;
  /** Number of frames/poses captured for this identity. */
  sampleCount: number;
}

export interface VerificationResult {
  matched: boolean;
  personId: string | null;
  /** Cosine similarity of the best match (−1..1). */
  similarity: number;
  /** Threshold used for the decision. */
  threshold: number;
}

/** An attendance/auth event queued for sync, then purged after upload. */
export interface AttendanceRecord {
  /** Idempotency key — stable across retries to dedupe server-side. */
  id: string;
  personId: string;
  timestamp: string;
  verified: boolean;
  similarity: number;
  liveness: LivenessResult;
  /** Optional: device id / location captured by the host app. */
  meta?: Record<string, string | number | boolean>;
}

export type SyncStatus = 'queued' | 'uploading' | 'uploaded' | 'failed';
