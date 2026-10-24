# FaceAuthDemo

The **working, on-device reference app** for `react-native-face-auth`. This is
where the "native inference seam" described in
[docs/TECHNICAL.md](../docs/TECHNICAL.md) is actually wired up and runnable on a
physical Android/iOS device — as opposed to [`example/App.tsx`](../example/App.tsx),
which is an unwired skeleton (`embedNow` returns `null`) that just demonstrates
the component API.

All pure pipeline logic (`FaceAuthSession`, `EnrollmentStore`, `SyncQueue`,
`ChallengeEngine`, the matcher) is imported directly from the library source in
[`../src`](../src) — nothing is duplicated here. This app only supplies the
native camera/model wiring around it.

## How this diverges from the documented architecture

The library's model stack (see root [README.md](../README.md)) assumes MediaPipe
Face Landmarker for detection/blendshapes and MiniFASNet v2 (TFLite) for passive
anti-spoof. This demo substitutes one piece, deliberately, flagged here so it
doesn't get mistaken for the canonical path:

| Library design | This demo | Why |
|---|---|---|
| MediaPipe Face Landmarker (blendshapes, 478 landmarks) | **MLKit** (`react-native-vision-camera-face-detector`) for eye-open / smile / head-angle | MLKit signals are mapped onto the library's MediaPipe-shaped `Blendshapes` type at the seam in [`src/FaceAuthCamera.tsx`](./src/FaceAuthCamera.tsx), so `ChallengeEngine` and the matcher run unmodified |
| MiniFASNet v2 passive anti-spoof (TFLite) | **Real** — `models/download_models.py` converts the ONNX checkpoint via `onnx2tf`; `useFaceAuthDemo.ts` loads the resulting `minifasnet_v2.tflite` with `useTensorflowModel` | Was ONNX-only and stubbed; conversion tooling already existed in `download_models.py`, it just hadn't been run. `FaceAuthCamera.tsx`'s 80×80-crop passive-score code path (already written) now has a real model behind it |
| Recognition — MobileFaceNet (ArcFace, TFLite) | **Real**, unmodified | `fast-tflite` `runSync()` on a 112×112 crop, same as documented |

If you're evaluating "is the native seam done": recognition, active liveness,
and passive (silent) anti-spoof are all fully wired and real. The one
open item is empirical: MiniFASNet's live-class index (`LIVE_CLASS_INDEX` in
`src/liveness/passive.ts`) follows the checkpoint's documented convention but
hasn't been re-confirmed against a labeled print/replay-attack photo set on
this specific converted model — do that before trusting the passive gate in
production.

## What's here

- [`App.tsx`](./App.tsx) — screens (home / enroll / verify), wires `useFaceAuthDemo`.
- [`src/useFaceAuthDemo.ts`](./src/useFaceAuthDemo.ts) — loads MobileFaceNet and
  MiniFASNet v2 via `useTensorflowModel`, constructs `FaceAuthSession` with the
  library's store/queue, auto-syncs on reconnect.
- [`src/FaceAuthCamera.tsx`](./src/FaceAuthCamera.tsx) — the frame processor: MLKit
  detection → crop/resize → MobileFaceNet embed + MiniFASNet passive score → hop
  to JS → step the FSM.

## Running it

This is a standard React Native CLI app (bootstrapped with
`@react-native-community/cli`), scoped to Android/iOS device or emulator — it
won't run in a browser or Expo Go.

```bash
npm install                 # installs this app's own deps (separate from the library's)
npx pod-install ios         # iOS only

python -m pip install -r ../models/requirements.txt   # onnx2tf + tensorflow, for the MiniFASNet convert step
python ../models/download_models.py                   # fetches + converts both .tflite models into assets/models/

npm run android              # or: npm run ios
```

Requires a physical device or emulator with a camera; camera permission is
requested on first launch (`AndroidManifest.xml` / `Info.plist` are already
configured).

## Benchmarking

See [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) for how to capture measured
per-stage latency from this app's frame processor.
