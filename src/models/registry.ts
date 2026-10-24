/**
 * Bundled model assets + their input contracts.
 *
 * Asset files live in `models/` and are loaded with `useTensorflowModel` from
 * react-native-fast-tflite (TFLite) and MediaPipe's own loader for the .task
 * graph. Keeping the require()s and shapes in one place documents the footprint
 * and makes it easy to swap quantized variants.
 *
 * Footprint budget (target <20 MB total):
 *   mobilefacenet.tflite     ~4-5 MB  (int8: ~1.5 MB)
 *   minifasnet_v2.tflite     ~1.9 MB
 *   face_landmarker.task     ~3-4 MB  (BlazeFace + mesh + blendshapes)
 *   ----------------------------------
 *   total                    ~9-11 MB
 */

export interface ModelSpec {
  /** Relative asset path (resolved by the host bundler). */
  asset: string;
  inputWidth: number;
  inputHeight: number;
  inputChannels: number;
  /** Per-channel normalization applied before inference. */
  normalize: 'pm1' | 'unit' | 'none'; // [-1,1] | [0,1] | raw
  description: string;
}

export const MODELS: Record<'recognition' | 'antiSpoof' | 'landmarker', ModelSpec> = {
  recognition: {
    asset: '../../models/mobilefacenet.tflite',
    inputWidth: 112,
    inputHeight: 112,
    inputChannels: 3,
    normalize: 'pm1',
    description: 'MobileFaceNet (ArcFace) face embedding',
  },
  antiSpoof: {
    asset: '../../models/minifasnet_v2.tflite',
    inputWidth: 80,
    inputHeight: 80,
    inputChannels: 3,
    normalize: 'unit',
    description: 'MiniFASNet v2 silent anti-spoofing',
  },
  landmarker: {
    asset: '../../models/face_landmarker.task',
    inputWidth: 0,
    inputHeight: 0,
    inputChannels: 3,
    normalize: 'none',
    description: 'MediaPipe Face Landmarker (BlazeFace + mesh + blendshapes)',
  },
};
