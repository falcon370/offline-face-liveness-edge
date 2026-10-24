# Models

This directory holds the three on-device model assets. They are **not committed**
(see repo `.gitignore`) — run `python download_models.py` to fetch and convert
them. All are open-source; see [LICENSES.md](./LICENSES.md).

## Footprint budget (target < 20 MB total)

| File | Model | Stage | Size (fp32) | Size (int8) |
|---|---|---|---|---|
| `mobilefacenet.tflite` | MobileFaceNet (ArcFace) | recognition | ~4.9 MB | ~1.5 MB |
| `minifasnet_v2.tflite` | MiniFASNet v2 | passive anti-spoof | ~1.9 MB | ~0.6 MB |
| `face_landmarker.task` | MediaPipe Face Landmarker (BlazeFace + mesh + blendshapes) | detect + active liveness | ~3.7 MB | — |
| **Total** | | | **~10.5 MB** | **~6.2 MB** |

Comfortably under the 20 MB cap. Quantizing the two TFLite models to int8 (script
flag `--quantize`) roughly halves the footprint with negligible accuracy loss for
this task; benchmark before/after on your validation set.

## Input contracts (see `src/models/registry.ts`)

- **MobileFaceNet** — `112x112x3`, RGB, normalized `(p/127.5 - 1)` → `[-1, 1]`,
  outputs a 128-/512-d embedding (L2-normalized in `embed()`).
- **MiniFASNet v2** — `80x80x3`, RGB, `[0, 1]`, outputs a 3-class softmax
  `[fake, live, replay]`. Confirm class order against the checkpoint you download
  and adjust `LIVE_CLASS_INDEX` in `src/liveness/passive.ts` if needed.
- **Face Landmarker** — handled by MediaPipe; produces 478 landmarks, 52
  blendshapes, and a 4×4 facial transform matrix per face.

## Obtaining the models
Run the script (requires Python 3.9+, `pip install -r requirements.txt`):

```bash
python download_models.py            # fp32
python download_models.py --quantize # int8 (smaller)
```

The script downloads from the official open-source sources and converts to the
formats above. If a source URL changes, the script prints the canonical project
page so you can grab the current artifact manually.
