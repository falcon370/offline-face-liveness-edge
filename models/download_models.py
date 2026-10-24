#!/usr/bin/env python3
"""Fetch and convert the three on-device model assets.

Outputs into this directory:
  - mobilefacenet.tflite   (face recognition embedding)
  - minifasnet_v2.tflite   (passive anti-spoofing)
  - face_landmarker.task   (MediaPipe detect + blendshapes)

All sources are open-source and free for commercial use (see LICENSES.md).
EdgeFace is intentionally NOT used (non-commercial license).

Usage:
    pip install -r requirements.txt
    python download_models.py [--quantize]

If a download URL has changed, the script prints the canonical project page so
you can grab the current artifact manually and drop it in this folder.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# Canonical open-source sources. URLs occasionally move; project pages are the
# stable fallback the script prints on failure.
SOURCES = {
    "face_landmarker.task": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "page": "https://developers.google.com/mediapipe/solutions/vision/face_landmarker",
        "license": "Apache-2.0",
    },
    "minifasnet_v2.onnx": {
        # MiniFASNet v2 (Silent-Face-Anti-Spoofing). Convert ONNX -> TFLite below.
        "url": "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx",
        "page": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing",
        "license": "Apache-2.0",
    },
    # InsightFace buffalo_s — w600k_mbf.onnx, MobileFaceNet trained on WebFace600K.
    # WebFace600K permits academic + commercial use; no MS-Celeb-1M weights.
    # The zip contains buffalo_s/w600k_mbf.onnx; zip_member extracts it directly.
    "mobilefacenet.onnx": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        "page": "https://github.com/deepinsight/insightface/tree/master/python-package",
        "license": "MIT (code) · WebFace600K (data, commercial OK)",
        "zip_member": "buffalo_s/w600k_mbf.onnx",
    },
}


def download(name: str, url: str, page: str, zip_member: str | None = None) -> str | None:
    dest = os.path.join(HERE, name)
    if os.path.exists(dest):
        print(f"  ✓ {name} already present, skipping")
        return dest
    if not url:
        print(f"  ! {name}: no direct URL configured. Get it from:\n      {page}")
        return None
    try:
        if zip_member:
            import io
            import zipfile
            print(f"  ↓ {name} (extracting {zip_member} from zip) ...")
            data = urllib.request.urlopen(url).read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                with zf.open(zip_member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
        else:
            print(f"  ↓ {name} ...")
            urllib.request.urlretrieve(url, dest)
        print(f"  ✓ {name}")
        return dest
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {name} failed: {e}\n      Download manually from: {page}")
        return None


def onnx_to_tflite(onnx_path: str, tflite_path: str, quantize: bool) -> None:
    """Convert ONNX -> TF SavedModel -> TFLite using onnx2tf (or onnx-tf)."""
    try:
        import onnx2tf  # type: ignore
    except ImportError:
        print(
            "  ! onnx2tf not installed. `pip install onnx2tf tensorflow onnx onnxsim`\n"
            "    then re-run, or convert manually."
        )
        return
    out_dir = tflite_path + ".saved"
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        output_integer_quantized_tflite=quantize,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    print(f"  ✓ converted -> {out_dir} (locate the .tflite and copy to {tflite_path})")


def main() -> int:
    # Windows consoles default to a legacy codepage that can't encode the ✓/↓/✗
    # progress glyphs below, crashing mid-download. Force UTF-8 stdout so this
    # runs the same on Windows as on macOS/Linux.
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", action="store_true", help="produce int8 TFLite")
    args = ap.parse_args()

    demo_assets = os.path.join(HERE, "..", "FaceAuthDemo", "assets", "models")

    print("Fetching models (open-source, commercial-OK; EdgeFace excluded)\n")
    for name, meta in SOURCES.items():
        path = download(name, meta["url"], meta["page"], meta.get("zip_member"))
        if path and name.endswith(".onnx"):
            tflite = path.replace(".onnx", ".tflite")
            onnx_to_tflite(path, tflite, args.quantize)
            tflite_name = os.path.basename(tflite)
            dst = os.path.join(demo_assets, tflite_name)
            if os.path.exists(tflite) and not os.path.exists(dst):
                import shutil
                os.makedirs(demo_assets, exist_ok=True)
                shutil.copy2(tflite, dst)
                print(f"  → {tflite_name} copied to FaceAuthDemo/assets/models/")

    print("\nDone. Verify the three assets exist and run the example app.")
    print("See README.md for input contracts and the footprint budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
