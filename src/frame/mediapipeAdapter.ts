/**
 * Adapts MediaPipe Face Landmarker output into our FaceFrame contract.
 *
 * react-native-mediapipe's useFaceLandmarkDetection (LIVE_STREAM) yields, per
 * frame: face landmarks, blendshape categories (name + score), and an optional
 * 4x4 facial transformation matrix. We map the blendshape names we care about
 * and derive head yaw/pitch/roll from the transform matrix's rotation part.
 *
 * Pure functions — no native imports — so the mapping is unit-testable.
 */

import type {
  Blendshapes,
  BoundingBox,
  FaceFrame,
  HeadPose,
  Point2D,
} from '../types';
import { isFaceUsable } from './preprocess';

export interface MpCategory {
  categoryName: string;
  score: number;
}

const RAD2DEG = 180 / Math.PI;

/** Pull the blendshapes we use out of MediaPipe's category list. */
export function extractBlendshapes(categories: MpCategory[]): Blendshapes {
  const get = (name: string): number =>
    categories.find((c) => c.categoryName === name)?.score ?? 0;
  return {
    eyeBlinkLeft: get('eyeBlinkLeft'),
    eyeBlinkRight: get('eyeBlinkRight'),
    mouthSmileLeft: get('mouthSmileLeft'),
    mouthSmileRight: get('mouthSmileRight'),
    jawOpen: get('jawOpen'),
  };
}

/**
 * Euler angles (deg) from a 4x4 (or 3x3) row-major transform matrix.
 * Uses the standard ZYX decomposition; yaw is rotation about the vertical axis.
 */
export function headPoseFromMatrix(m: number[] | undefined): HeadPose {
  if (!m || m.length < 9) return { yaw: 0, pitch: 0, roll: 0 };
  // Support both 16-length (4x4) and 9-length (3x3) inputs.
  const stride = m.length >= 16 ? 4 : 3;
  const r = (row: number, col: number): number => m[row * stride + col];

  const sy = Math.sqrt(r(0, 0) * r(0, 0) + r(1, 0) * r(1, 0));
  const singular = sy < 1e-6;

  let pitch: number;
  let yaw: number;
  let roll: number;
  if (!singular) {
    pitch = Math.atan2(r(2, 1), r(2, 2));
    yaw = Math.atan2(-r(2, 0), sy);
    roll = Math.atan2(r(1, 0), r(0, 0));
  } else {
    pitch = Math.atan2(-r(1, 2), r(1, 1));
    yaw = Math.atan2(-r(2, 0), sy);
    roll = 0;
  }
  return { yaw: yaw * RAD2DEG, pitch: pitch * RAD2DEG, roll: roll * RAD2DEG };
}

/** Bounding box (normalized) from a set of normalized landmarks. */
export function boxFromLandmarks(landmarks: Point2D[]): BoundingBox {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const p of landmarks) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

export interface MpFaceResult {
  landmarks: Point2D[];
  blendshapes: MpCategory[];
  transformMatrix?: number[];
  timestampMs: number;
}

/** Build a FaceFrame from a single-face MediaPipe result. */
export function toFaceFrame(result: MpFaceResult | null, now: number): FaceFrame {
  if (!result || result.landmarks.length === 0) {
    return {
      timestampMs: now,
      box: { x: 0, y: 0, width: 0, height: 0 },
      landmarks: [],
      blendshapes: {
        eyeBlinkLeft: 0,
        eyeBlinkRight: 0,
        mouthSmileLeft: 0,
        mouthSmileRight: 0,
        jawOpen: 0,
      },
      headPose: { yaw: 0, pitch: 0, roll: 0 },
      facePresent: false,
    };
  }
  const box = boxFromLandmarks(result.landmarks);
  return {
    timestampMs: result.timestampMs || now,
    box,
    landmarks: result.landmarks,
    blendshapes: extractBlendshapes(result.blendshapes),
    headPose: headPoseFromMatrix(result.transformMatrix),
    facePresent: isFaceUsable(box),
  };
}
