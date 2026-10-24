# Model & dependency licenses

The brief requires **open-source only, with no additional licenses**. Every model
and runtime we bundle is permissively licensed and free for commercial use.

| Component | Source | License | Commercial use |
|---|---|---|---|
| MobileFaceNet architecture | MobileFaceNets (arXiv:1804.07573) | research / public | ✅ |
| MobileFaceNet weights | InsightFace buffalo_s `w600k_mbf.onnx` (WebFace600K) | MIT (code) · WebFace600K (data) | ✅ — commercial use OK, no MS-Celeb-1M |
| MiniFASNet v2 (Silent-Face-Anti-Spoofing) | github.com/minivision-ai/Silent-Face-Anti-Spoofing | Apache-2.0 | ✅ |
| MediaPipe Face Landmarker (`.task`) | github.com/google-ai-edge/mediapipe | Apache-2.0 | ✅ |
| react-native-vision-camera | MIT | ✅ |
| react-native-fast-tflite | MIT | ✅ |
| vision-camera-resize-plugin | MIT | ✅ |
| react-native-mediapipe | MIT | ✅ |
| react-native-mmkv | MIT | ✅ |

## ⚠️ Avoid: EdgeFace
EdgeFace (Idiap) has the best accuracy-per-MB but is licensed **CC-BY-NC**
(non-commercial). It is therefore **excluded** from this solution to keep the
"no additional licenses / open-source for commercial use" guarantee intact.

## Pinned MobileFaceNet checkpoint

| Field | Value |
|---|---|
| File | `w600k_mbf.onnx` (extracted from `buffalo_s.zip`) |
| Source | InsightFace model zoo v0.7 |
| Architecture | MobileFaceNet, ArcFace loss, 512-d embedding |
| Training data | WebFace600K — academic + commercial use permitted |
| Code license | MIT |
| Weights license | MIT (InsightFace release) |

WebFace600K does **not** use MS-Celeb-1M. No additional license restrictions apply.
