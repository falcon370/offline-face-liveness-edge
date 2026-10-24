# On-Device Latency Benchmarks

Measured end-to-end on mid-range Android and iOS devices.
Record times with `performance.now()` around each stage in `FaceAuthCamera.tsx`.

**Status: the three model/camera stages below are still unmeasured.** Capturing
them requires a physical Android or iOS device connected via `adb`/Xcode running
`FaceAuthDemo`; no device was available in the environment these docs were last
updated from (checked: `adb devices` listed none, no Xcode toolchain on this
Windows machine). The matching stage doesn't depend on a phone (it's pure JS) and
has been measured on a dev machine as a stand-in — see below.

## Target vs. measured

| Stage | Target | Android (mid-range) | iOS (mid-range) |
|---|---|---|---|
| MLKit detect + pose (per frame @ 12 fps) | 8–20 ms | — *(needs device)* | — *(needs device)* |
| MiniFASNet passive (per face crop) | ~20 ms | — *(needs device)* | — *(needs device)* |
| MobileFaceNet embed (1× after liveness) | 15–40 ms | — *(needs device)* | — *(needs device)* |
| 1:N cosine match (N ≈ 1 000) | < 5 ms | see dev-machine figure below | see dev-machine figure below |
| **Recognition pass total** | **< 200 ms** | — *(needs device)* | — *(needs device)* |

## Matching stage — measured (dev machine, not mobile)

`identify()` (`src/recognition/matcher.ts`) is pure JS/TS with no native
dependency, so it can be timed off-device as a rough proxy — a real phone CPU
will be slower, but this at least confirms the algorithm's own cost is not the
bottleneck. Measured with `performance.now()`, 512-d embeddings, N=1000 gallery
entries (3 pose templates each), 200 trials after 20 warm-up iterations, on an
Intel Core i5-13420H (Node v24.11.0):

| Metric | Value |
|---|---|
| Mean | 1.75 ms |
| p50 | 1.71 ms |
| p95 | 2.10 ms |
| Min / Max | 1.46 ms / 2.98 ms |

Comfortably under the < 5 ms target even with JS-array overhead; a phone's JS
thread should land in the same range since this is O(N·D) scalar math, not
something that benefits much from desktop-class SIMD.

## Test devices

| Device | SoC | OS | Notes |
|---|---|---|---|
| — | — | — | fill in once run on a real Android/iOS device |

## How to measure

Add `performance.now()` around each stage in the frame processor worklet and
`onFrame` callback in `FaceAuthDemo/src/FaceAuthCamera.tsx`, then read the
timings from the Metro console while running on a real device.

```ts
// Example: time the MobileFaceNet runSync call
const t0 = performance.now();
const out = model.runSync([input]);
console.log('embed ms', performance.now() - t0);
```
