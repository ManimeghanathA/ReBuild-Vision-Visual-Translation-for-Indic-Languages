"""
local_web_server.py
---------------------
Tiny local HTTP server that:
  1) serves the frontend (index.html / styles.css / app.js)
  2) exposes POST /api/translate for image upload

This server REUSES your existing modular pipeline:
  - scripts/vtt/* (OCR + Sarvam translation + TELEA inpainting)
  - CRAFT-pytorch test.py (text region detection)

No changes are made to your existing main app code.
"""

from __future__ import annotations

import base64
import cgi
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRAFT_DIR = os.path.join(BASE_DIR, "CRAFT-pytorch")
WEIGHTS = os.path.join(CRAFT_DIR, "craft_mlt_25k.pth")
RESULT_DIR = os.path.join(CRAFT_DIR, "result")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

# Make `scripts/vtt` importable as `vtt`
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

_craft_ready = False
_craft_ready_lock = threading.Lock()

_ocr_reader = None
_ocr_reader_lock = threading.Lock()


def _has_gpu() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def setup_craft() -> None:
    """
    One-time CRAFT setup (clone weights + patch torchvision compatibility).
    Safe to call multiple times.
    """
    global _craft_ready
    with _craft_ready_lock:
        if _craft_ready:
            return

        # 1) Ensure CRAFT repo exists
        if not os.path.exists(CRAFT_DIR):
            print("[setup] Cloning CRAFT-pytorch...")
            subprocess.run(
                ["git", "clone", "https://github.com/clovaai/CRAFT-pytorch.git", CRAFT_DIR],
                check=True,
                capture_output=True,
                text=True,
            )

        # 2) Ensure weights exist
        os.makedirs(RESULT_DIR, exist_ok=True)
        if not os.path.exists(WEIGHTS):
            print("[setup] Downloading CRAFT weights (~170MB)...")
            try:
                import gdown  # type: ignore

                gdown.download(
                    "https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ",
                    WEIGHTS,
                    quiet=False,
                )
            except Exception as e:
                raise RuntimeError(f"CRAFT weights download failed: {e}")

        # 3) Patch modern torchvision compatibility (idempotent)
        vgg_path = os.path.join(CRAFT_DIR, "basenet", "vgg16_bn.py")
        craft_py = os.path.join(CRAFT_DIR, "craft.py")

        if os.path.exists(vgg_path):
            with open(vgg_path, "r", encoding="utf-8", errors="ignore") as f:
                vgg_src = f.read()

            # Comment out model_urls import if present
            vgg_src2 = vgg_src.replace(
                "from torchvision.models.vgg import model_urls",
                "# from torchvision.models.vgg import model_urls",
            )

            # Also ensure line ~24 is commented (same logic as app.py)
            lines = vgg_src2.split("\n")
            if len(lines) > 24 and not lines[24].startswith("#"):
                lines[24] = "# " + lines[24]
            vgg_src3 = "\n".join(lines)

            if vgg_src3 != vgg_src:
                with open(vgg_path, "w", encoding="utf-8") as f:
                    f.write(vgg_src3)

        if os.path.exists(craft_py):
            with open(craft_py, "r", encoding="utf-8", errors="ignore") as f:
                craft_src = f.read()

            craft_src2 = craft_src.replace(
                "vgg16_bn(pretrained=True, freeze=True)",
                "vgg16_bn(pretrained=False, freeze=True)",
            )
            if craft_src2 != craft_src:
                with open(craft_py, "w", encoding="utf-8") as f:
                    f.write(craft_src2)

        _craft_ready = True


def run_craft(img_path: str) -> str:
    """Run CRAFT test.py on a single image. Returns path to res_{stem}.txt."""
    setup_craft()

    stem = os.path.splitext(os.path.basename(img_path))[0]
    res_path = os.path.join(RESULT_DIR, f"res_{stem}.txt")

    # Run CRAFT on only this image to avoid extra work
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_img = os.path.join(tmp_dir, os.path.basename(img_path))
        shutil.copy2(img_path, tmp_img)

        cmd = [
            sys.executable,
            "test.py",
            f"--trained_model={WEIGHTS}",
            f"--test_folder={tmp_dir}",
            f"--cuda={'True' if _has_gpu() else 'False'}",
        ]
        result = subprocess.run(
            cmd,
            cwd=CRAFT_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CRAFT failed:\n{result.stderr[-2000:]}")

    if not os.path.exists(res_path):
        raise FileNotFoundError(
            f"CRAFT result not found: {res_path}\n"
            f"CRAFT stdout snippet: (see server logs)"
        )
    return res_path


def load_ocr_reader():
    """Load EasyOCR reader once and reuse."""
    global _ocr_reader
    with _ocr_reader_lock:
        if _ocr_reader is not None:
            return _ocr_reader

        import easyocr  # type: ignore

        _ocr_reader = easyocr.Reader(["te", "en"], gpu=_has_gpu())
        return _ocr_reader


def _strip_think(text: str) -> str:
    import re

    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def run_pipeline(img_path: str, api_key: str, skip_translate: bool) -> dict:
    """
    Returns:
      - inpainted_rgb: np.ndarray (RGB)
      - translation_data: list[dict]
      - image_type: str
      - n_areas: int
    """
    from vtt import (  # pylint: disable=import-outside-toplevel
        area_bbox,
        deduplicate_craft_boxes,
        deduplicate_ocr_across_areas,
        detect_image_type,
        generate_area_mask,
        inpaint_all_areas,
        inpaint_noise_boxes,
        is_telugu_area,
        merge_overlapping_areas,
        normalize_telugu_ocr,
        ocr_area,
        purify_areas,
        reconstruct_area_sentence,
        split_telugu_and_other,
        translate_areas,
        load_craft_boxes,
    )

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # CRAFT
    res_path = run_craft(img_path)
    raw_boxes = load_craft_boxes(res_path)
    boxes = deduplicate_craft_boxes(raw_boxes)

    # Detection -> areas
    from vtt import build_text_areas  # keep import local

    areas_raw = build_text_areas(boxes)
    areas_merged = merge_overlapping_areas(areas_raw)
    valid_areas, _ = purify_areas(areas_merged, img.shape)

    # OCR -> processed_areas
    processed_raw = []
    ocr_reader = load_ocr_reader()

    for idx, area in enumerate(valid_areas):
        ocr_results = ocr_area(img, area, ocr_reader)
        if not ocr_results:
            continue
        if not is_telugu_area(ocr_results, min_telugu_chars=2):
            continue

        sentence = reconstruct_area_sentence(ocr_results)
        if not sentence:
            continue

        mask = generate_area_mask(img.shape, area)
        telugu_words, other_words = split_telugu_and_other(ocr_results)
        processed_raw.append(
            {
                "area_idx": idx,
                "area_bbox": area_bbox(area),
                "area_quads": [b["quad"] for b in area],
                "sentence": sentence,
                "full_text": " ".join(w["text"] for w in sentence),
                "telugu_words": telugu_words,
                "other_words": other_words,
                "raw_ocr": ocr_results,
                "mask": mask,
            }
        )

    processed_areas = deduplicate_ocr_across_areas(processed_raw)

    if not processed_areas:
        return {
            "inpainted_rgb": img,
            "translation_data": [],
            "image_type": "unknown",
            "n_areas": 0,
        }

    # Translation (optional)
    image_type = "unknown"
    if not skip_translate and api_key:
        image_type = detect_image_type(processed_areas, api_key)

        for area in processed_areas:
            raw = area.get("full_text", "").strip()
            area["corrected_telugu"] = normalize_telugu_ocr(raw, api_key) if raw else ""
            time.sleep(0.25)

        corrected = [a.get("corrected_telugu", "") for a in processed_areas]
        tamil_results = translate_areas(corrected, image_type, api_key)
        for i, area in enumerate(processed_areas):
            area["tamil_translation"] = tamil_results[i]
    else:
        # Keep keys consistent with frontend.
        for area in processed_areas:
            area["corrected_telugu"] = area.get("corrected_telugu", "")
            area["tamil_translation"] = area.get("tamil_translation", "")

    # Inpainting
    inpainted = inpaint_all_areas(img, processed_areas)
    inpainted = inpaint_noise_boxes(inpainted, raw_boxes, processed_areas)

    translation_data = [
        {
            "area": i + 1,
            "raw_ocr": area.get("full_text", ""),
            "corrected_telugu": _strip_think(area.get("corrected_telugu", "")),
            "tamil_translation": area.get("tamil_translation", ""),
            "bbox": list(area["area_bbox"]),
        }
        for i, area in enumerate(processed_areas)
    ]

    return {
        "inpainted_rgb": inpainted,
        "translation_data": translation_data,
        "image_type": image_type,
        "n_areas": len(processed_areas),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalVTTWebServer/1.0"

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, rel_path: str) -> None:
        path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(path) or os.path.isdir(path):
            self.send_error(404, "Not found")
            return

        ctype, _ = mimetypes.guess_type(path)
        if not ctype:
            ctype = "application/octet-stream"

        with open(path, "rb") as f:
            body = f.read()

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed_path = self.path.split("?", 1)[0]

        if parsed_path in ("/", "/index.html"):
            return self._serve_static("frontend/index.html")

        if parsed_path in ("/styles.css", "/app.js"):
            return self._serve_static(f"frontend/{parsed_path.lstrip('/')}")

        if parsed_path.startswith("/static/"):
            # Optional: if you later add a /static folder.
            return self._serve_static(parsed_path.lstrip("/"))

        if parsed_path == "/api/health":
            return self._send_json(200, {"ok": True})

        self.send_error(404, "Not found")

    def do_POST(self):  # noqa: N802
        parsed_path = self.path.split("?", 1)[0]

        if parsed_path != "/api/translate":
            return self.send_error(404, "Not found")

        # Parse multipart/form-data
        try:
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                return self._send_json(400, {"error": "Expected multipart/form-data"})

            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return self._send_json(400, {"error": "Empty request"})

            env = {"REQUEST_METHOD": "POST"}
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ=env,
                keep_blank_values=True,
            )

            if "image" not in form:
                return self._send_json(400, {"error": "Missing form field: image"})

            image_item = form["image"]
            filename = getattr(image_item, "filename", None) or "upload"
            api_key = str(form.getvalue("api_key") or "").strip()
            skip_translate = str(form.getvalue("skip_translate") or "").strip() in ("1", "true", "True")

            if not api_key:
                # Allow inpainting-only mode, but frontend should usually pass the key.
                skip_translate = True

            # Save upload to a temp file
            suffix = os.path.splitext(filename)[1].lower() or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_item.file.read())
                tmp_path = tmp.name

            try:
                t0 = time.time()
                result = run_pipeline(tmp_path, api_key=api_key, skip_translate=skip_translate)

                inpainted_rgb = result["inpainted_rgb"]
                inpainted_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
                ok, img_bytes = cv2.imencode(".jpg", inpainted_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError("Failed to encode inpainted image")

                b64 = base64.b64encode(img_bytes.tobytes()).decode("ascii")

                out = {
                    "image_type": result["image_type"],
                    "n_areas": result["n_areas"],
                    "inpainted_image_jpeg_base64": b64,
                    "translation_data": result["translation_data"],
                    "elapsed_sec": round(time.time() - t0, 2),
                    "skip_translate": skip_translate,
                }
                return self._send_json(200, out)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            return self._send_json(500, {"error": f"Server error: {e}"})


def main() -> None:
    port = 8000
    host = "127.0.0.1"
    print(f"[server] Starting on http://{host}:{port}")
    print("[server] Open the browser and upload an image.")

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

