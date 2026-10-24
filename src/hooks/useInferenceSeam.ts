/**
 * useInferenceSeam — native inference glue for MobileFaceNet + MiniFASNet.
 *
 * Runs both TFLite models inside a Vision Camera frame processor worklet
 * (camera thread, no JS bridge overhead). Results are written into shared
 * values so the JS-thread callbacks `embedNow` / `passiveScoreNow` can read
 * the latest inference result at any moment.
 *
 * Usage in the host app:
 *   const mfn = useTensorflowModel(require('../models/mobilefacenet.tflite'));
 *   const fas = useTensorflowModel(require('../models/minifasnet_v2.tflite'));
 *   const seam = useInferenceSeam(mfn, fas);
 *   const faceAuth = useFaceAuth({ ...seam, encryptionKey, uploader });
 *   <LivenessCamera ... inferenceFrameProcessor={seam.frameProcessor} />
 */

import { useCallback } from 'react';
import { useSharedValue } from 'react-native-worklets-core';
import { useFrameProcessor } from 'react-native-vision-camera';
import type { FrameProcessor } from 'react-native-vision-camera';
import { resize } from 'vision-camera-resize-plugin';
import type { TensorflowModelState } from 'react-native-fast-tflite';
import type { EmbedNow, PassiveScoreNow } from '../session';
import type { Embedding } from '../types';
import { l2Normalize } from '../recognition/matcher';

export interface InferenceSeam {
  embedNow: EmbedNow;
  passiveScoreNow: PassiveScoreNow;
  /** Compose this with LivenessCamera's inferenceFrameProcessor prop. */
  frameProcessor: FrameProcessor;
}

/**
 * Wire the two TFLite models to the camera thread.
 *
 * @param recognition  State from useTensorflowModel(mobilefacenet.tflite)
 * @param antiSpoof    State from useTensorflowModel(minifasnet_v2.tflite)
 */
export function useInferenceSeam(
  recognition: TensorflowModelState,
  antiSpoof: TensorflowModelState
): InferenceSeam {
  // number[] rather than Float32Array: simpler cross-thread serialization.
  const rawEmb = useSharedValue<number[]>([]);
  const liveScore = useSharedValue<number>(0.95);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      const mfn = recognition.model;
      if (!mfn) return;

      // ── MobileFaceNet — 112×112 RGB, normalized to [-1, 1] ────────────────
      const rgb112 = resize(frame, {
        scale: { width: 112, height: 112 },
        pixelFormat: 'rgb',
        dataType: 'float32',
      }) as Float32Array;
      for (let i = 0; i < rgb112.length; i++) {
        rgb112[i] = rgb112[i] / 127.5 - 1;
      }
      const [embOut] = mfn.runSync([rgb112]);
      const embF32 =
        embOut instanceof Float32Array
          ? embOut
          : Float32Array.from(embOut as ArrayLike<number>);
      rawEmb.value = Array.from(embF32);

      // ── MiniFASNet — 80×80 RGB, normalized to [0, 1] ──────────────────────
      const fas = antiSpoof.model;
      if (!fas) return;
      const rgb80 = resize(frame, {
        scale: { width: 80, height: 80 },
        pixelFormat: 'rgb',
        dataType: 'float32',
      }) as Float32Array;
      for (let i = 0; i < rgb80.length; i++) {
        rgb80[i] = rgb80[i] / 255;
      }
      const [fasOut] = fas.runSync([rgb80]);

      // Numerically-stable softmax → P(live) at class index 1
      const logits = fasOut as Float32Array;
      let max = -Infinity;
      for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
      let sum = 0;
      const exps: number[] = [];
      for (let i = 0; i < logits.length; i++) {
        const e = Math.exp(logits[i] - max);
        exps.push(e);
        sum += e;
      }
      liveScore.value = (exps[1] ?? 0) / (sum || 1);
    },
    [recognition, antiSpoof]
  );

  const embedNow: EmbedNow = useCallback((): Embedding | null => {
    const raw = rawEmb.value;
    if (!raw || raw.length === 0) return null;
    return l2Normalize(new Float32Array(raw));
  }, [rawEmb]);

  const passiveScoreNow: PassiveScoreNow = useCallback(
    () => liveScore.value,
    [liveScore]
  );

  return { embedNow, passiveScoreNow, frameProcessor };
}
