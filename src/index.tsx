/**
 * react-native-face-auth — public API.
 *
 * Offline, on-device facial recognition + liveness for React Native. Composes
 * open-source models (MobileFaceNet + MiniFASNet + MediaPipe Face Landmarker)
 * via react-native-vision-camera, react-native-fast-tflite and MMKV. Designed to
 * drop into the Datalake 3.0 app as a module.
 *
 * Two ways to use it:
 *  1) <LivenessCamera mode="verify" .../>  — turnkey UI component (recommended).
 *  2) FaceAuthSession + the building blocks — for custom UIs / advanced flows.
 */

// ── Turnkey UI ────────────────────────────────────────────────────────────────
export { LivenessCamera } from './components/LivenessCamera';
export { useFaceAuth } from './hooks/useFaceAuth';

// ── Orchestration ──────────────────────────────────────────────────────────────
export { FaceAuthSession, VerifyRunner, EnrollRunner } from './session';
export type { FaceAuthDeps, EmbedNow, PassiveScoreNow } from './session';

// ── Building blocks ──────────────────────────────────────────────────────────
export { ChallengeEngine, DEFAULT_CHALLENGE_CONFIG } from './liveness/challengeEngine';
export type { ChallengeEngineConfig } from './liveness/challengeEngine';
export { softmax, fusePassiveScores, LIVE_CLASS_INDEX } from './liveness/passive';

export {
  cosineSimilarity,
  l2Normalize,
  averageTemplate,
  verifyAgainst,
  identify,
  DEFAULT_MATCH_THRESHOLD,
} from './recognition/matcher';
export { embed, MOBILEFACENET_INPUT } from './recognition/embedder';
export type { TFLiteModel } from './recognition/embedder';

export { EnrollmentStore } from './storage/enrollmentStore';
export type { KeyValueStore } from './storage/enrollmentStore';

export { SyncQueue } from './sync/syncQueue';
export type { RemoteUploader, SyncQueueOptions } from './sync/syncQueue';
export { MockServer } from './sync/mockEndpoint';

export {
  toFaceFrame,
  extractBlendshapes,
  headPoseFromMatrix,
  boxFromLandmarks,
} from './frame/mediapipeAdapter';
export type { MpCategory, MpFaceResult } from './frame/mediapipeAdapter';

export {
  expandBox,
  eyeAlignmentAngle,
  isFaceUsable,
  buildClaheLut,
  normalizePixel,
} from './frame/preprocess';

export { MODELS } from './models/registry';
export type { ModelSpec } from './models/registry';

// ── Types ───────────────────────────────────────────────────────────────────
export type {
  Embedding,
  Point2D,
  BoundingBox,
  Blendshapes,
  HeadPose,
  FaceFrame,
  ChallengeType,
  ChallengeStatus,
  ChallengeState,
  LivenessOutcome,
  LivenessResult,
  EnrolledIdentity,
  VerificationResult,
  AttendanceRecord,
  SyncStatus,
} from './types';
