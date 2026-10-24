/**
 * MobileFaceNet embedding via react-native-fast-tflite.
 *
 * fast-tflite loads a `.tflite` asset and exposes `runSync(inputs)` which is
 * callable from a VisionCamera worklet (the camera thread). The model takes a
 * 112x112x3 float tensor (NHWC, normalized to [-1,1]) and returns a 128- or
 * 512-d embedding which we L2-normalize for cosine matching.
 *
 * We keep the heavy native object out of this pure module; callers inject the
 * loaded model (from `useTensorflowModel`) so this file stays testable and the
 * native dependency stays optional at import time.
 */

import { l2Normalize } from './matcher';
import type { Embedding } from '../types';

/** Minimal structural type matching fast-tflite's loaded model. */
export interface TFLiteModel {
  runSync(inputs: (Float32Array | Uint8Array | Int32Array)[]):
    (Float32Array | Uint8Array | Int32Array)[];
}

export const MOBILEFACENET_INPUT = { width: 112, height: 112, channels: 3 } as const;

/**
 * Run MobileFaceNet on an already-prepared, normalized input tensor and return
 * the L2-normalized embedding. `input` must be width*height*channels float32 in
 * [-1, 1], laid out NHWC (the layout vision-camera-resize-plugin produces).
 */
export function embed(model: TFLiteModel, input: Float32Array): Embedding {
  const expected =
    MOBILEFACENET_INPUT.width *
    MOBILEFACENET_INPUT.height *
    MOBILEFACENET_INPUT.channels;
  if (input.length !== expected) {
    throw new Error(
      `MobileFaceNet expects ${expected} floats, got ${input.length}`
    );
  }
  const out = model.runSync([input]);
  const raw = out[0];
  const vec = raw instanceof Float32Array ? raw : Float32Array.from(raw as ArrayLike<number>);
  return l2Normalize(vec);
}
