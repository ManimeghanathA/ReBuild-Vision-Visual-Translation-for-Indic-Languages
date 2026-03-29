"""
app.py — ReBuild Vision Web Demo
──────────────────────────────────
Streamlit application for Telugu → Tamil visual text translation.

Designed to run on:
  • Local machine:           streamlit run app.py
  • Hugging Face Spaces:     push repo, select Streamlit SDK, GPU T4 (free)

On first run, downloads CRAFT weights (~170 MB) from Google Drive and
patches CRAFT-pytorch for modern torchvision. This takes ~2 minutes once,
then the weights are cached for all subsequent runs.

The user provides:
  1. An image (jpg / png)
  2. Their Sarvam AI API key

The app returns:
  1. Inpainted image (Telugu text removed) — downloadable
  2. Translation table (area-by-area JSON) — downloadable
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import textwrap

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReBuild Vision — Telugu → Tamil Visual Translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CRAFT_DIR  = os.path.join(BASE_DIR, "CRAFT-pytorch")
WEIGHTS    = os.path.join(CRAFT_DIR, "craft_mlt_25k.pth")
RESULT_DIR = os.path.join(CRAFT_DIR, "result")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

# Add scripts/ to path so `from vtt import ...` works
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


# ── CRAFT setup (runs once, cached) ──────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def setup_craft():
    """
    One-time setup:
      1. Clone CRAFT-pytorch if not present
      2. Download model weights if not present
      3. Patch for modern torchvision
      4. Return True when ready
    """
    # 1. Clone
    if not os.path.exists(CRAFT_DIR):
        st.info("Cloning CRAFT-pytorch… (first run only)")
        subprocess.run(
            ["git", "clone", "https://github.com/clovaai/CRAFT-pytorch.git",
             CRAFT_DIR],
            check=True, capture_output=True,
        )

    # 2. Download weights
    if not os.path.exists(WEIGHTS):
        st.info("Downloading CRAFT weights (~170 MB)… (first run only)")
        try:
            import gdown
            gdown.download(
                "https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ",
                WEIGHTS, quiet=False,
            )
        except Exception as e:
            st.error(f"Failed to download weights: {e}")
            return False

    # 3. Patch vgg16_bn.py (idempotent)
    vgg_path = os.path.join(CRAFT_DIR, "basenet", "vgg16_bn.py")
    craft_py  = os.path.join(CRAFT_DIR, "craft.py")
    with open(vgg_path, "r") as f:
        vgg_src = f.read()
    vgg_src = vgg_src.replace(
        "from torchvision.models.vgg import model_urls",
        "# from torchvision.models.vgg import model_urls",
    )
    lines = vgg_src.split("\n")
    if len(lines) > 24 and not lines[24].startswith("#"):
        lines[24] = "# " + lines[24]
    vgg_src = "\n".join(lines)
    with open(vgg_path, "w") as f:
        f.write(vgg_src)

    with open(craft_py, "r") as f:
        craft_src = f.read()
    craft_src = craft_src.replace(
        "vgg16_bn(pretrained=True, freeze=True)",
        "vgg16_bn(pretrained=False, freeze=True)",
    )
    with open(craft_py, "w") as f:
        f.write(craft_src)

    os.makedirs(RESULT_DIR, exist_ok=True)
    return True


@st.cache_resource(show_spinner=False)
def load_ocr_reader():
    """Load EasyOCR reader once and cache it."""
    import easyocr
    return easyocr.Reader(["te", "en"], gpu=_has_gpu())


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── CRAFT inference ───────────────────────────────────────────────────────────

def run_craft(img_path: str) -> str:
    """
    Run CRAFT test.py on a single image.
    Returns path to the generated result .txt file.
    """
    stem      = os.path.splitext(os.path.basename(img_path))[0]
    res_path  = os.path.join(RESULT_DIR, f"res_{stem}.txt")

    # Build a temp folder with just this image so CRAFT doesn't process others
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Symlink or copy the image
        tmp_img = os.path.join(tmp_dir, os.path.basename(img_path))
        import shutil
        shutil.copy2(img_path, tmp_img)

        cmd = [
            sys.executable, "test.py",
            f"--trained_model={WEIGHTS}",
            f"--test_folder={tmp_dir}",
            f"--cuda={'True' if _has_gpu() else 'False'}",
        ]
        result = subprocess.run(
            cmd, cwd=CRAFT_DIR,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CRAFT failed:\n{result.stderr[-2000:]}"
            )

    # CRAFT writes result to CRAFT-pytorch/result/res_{stem}.txt
    if not os.path.exists(res_path):
        raise FileNotFoundError(
            f"CRAFT result not found: {res_path}\n"
            f"CRAFT stdout: {result.stdout[-1000:]}"
        )
    return res_path


# ── Full pipeline (cached per image+key combo) ────────────────────────────────

def run_pipeline(img_path: str, api_key: str, ocr_reader) -> dict:
    """
    Runs the full VTT pipeline on one image.
    Returns dict with keys: inpainted_rgb, translation_data.
    """
    from vtt import (
        load_craft_boxes, deduplicate_craft_boxes,
        build_text_areas, merge_overlapping_areas,
        purify_areas, area_bbox, generate_area_mask,
        ocr_area, reconstruct_area_sentence,
        is_telugu_area, split_telugu_and_other,
        deduplicate_ocr_across_areas,
        detect_image_type, normalize_telugu_ocr, translate_areas,
        IMAGE_TYPE_DESCRIPTIONS,
        inpaint_all_areas, inpaint_noise_boxes,
    )
    import re

    def _strip_think(t):
        return re.sub(r"<think>.*?</think>", "", t or "",
                      flags=re.DOTALL).strip()

    # Load image
    img_bgr = cv2.imread(img_path)
    img     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # CRAFT
    res_path = run_craft(img_path)

    # Detection
    raw_boxes = load_craft_boxes(res_path)
    boxes     = deduplicate_craft_boxes(raw_boxes)
    areas_raw    = build_text_areas(boxes)
    areas_merged = merge_overlapping_areas(areas_raw)
    valid_areas, _ = purify_areas(areas_merged, img.shape)

    # OCR
    processed_raw = []
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
        processed_raw.append({
            "area_idx":     idx,
            "area_bbox":    area_bbox(area),
            "area_quads":   [b["quad"] for b in area],
            "sentence":     sentence,
            "full_text":    " ".join(w["text"] for w in sentence),
            "telugu_words": telugu_words,
            "other_words":  other_words,
            "raw_ocr":      ocr_results,
            "mask":         mask,
        })
    processed_areas = deduplicate_ocr_across_areas(processed_raw)

    if not processed_areas:
        return {"inpainted_rgb": img, "translation_data": [],
                "image_type": "unknown", "n_areas": 0}

    # Translation
    image_type = detect_image_type(processed_areas, api_key)
    for area in processed_areas:
        raw = area.get("full_text", "").strip()
        area["corrected_telugu"] = (
            normalize_telugu_ocr(raw, api_key) if raw else ""
        )
        time.sleep(0.25)
    corrected     = [a.get("corrected_telugu", "") for a in processed_areas]
    tamil_results = translate_areas(corrected, image_type, api_key)
    for i, area in enumerate(processed_areas):
        area["tamil_translation"] = tamil_results[i]

    # Inpainting
    inpainted = inpaint_all_areas(img, processed_areas)
    inpainted = inpaint_noise_boxes(inpainted, raw_boxes, processed_areas)

    # Build output data
    translation_data = [{
        "area":              i + 1,
        "raw_ocr":           area.get("full_text", ""),
        "corrected_telugu":  _strip_think(area.get("corrected_telugu", "")),
        "tamil_translation": area.get("tamil_translation", ""),
        "bbox":              list(area["area_bbox"]),
    } for i, area in enumerate(processed_areas)]

    return {
        "inpainted_rgb":   inpainted,
        "translation_data": translation_data,
        "image_type":       image_type,
        "n_areas":          len(processed_areas),
    }


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/translation.png", width=64)
        st.title("ReBuild Vision")
        st.caption("Telugu Scene Text → Tamil")
        st.divider()

        st.subheader("🔑 API Key")
        api_key = st.text_input(
            "Sarvam AI API key",
            type="password",
            placeholder="sk-…",
            help="Get your key at sarvam.ai",
        )
        if not api_key:
            st.warning("Enter your Sarvam AI API key to enable translation.")

        st.divider()
        st.subheader("ℹ️ About")
        st.markdown(textwrap.dedent("""
            This demo runs the full **VTT pipeline**:

            1. **CRAFT** detects text regions
            2. **EasyOCR** reads Telugu
            3. **sarvam-m** normalises + translates
            4. **TELEA** inpaints (erases original text)

            [GitHub →](https://github.com/ManimeghanathA/ReBuild-Vision-Visual-Translation-for-Indic-Languages)
        """))

        gpu_status = "✅ GPU" if _has_gpu() else "⚠️ CPU (slower)"
        st.caption(f"Runtime: {gpu_status}")

    # ── Main area ─────────────────────────────────────────────────────────────
    st.title("🌐 Telugu → Tamil Visual Translation")
    st.markdown(
        "Upload a photo of a Telugu sign, poster, or document. "
        "The pipeline will erase the Telugu text and return the Tamil translation."
    )

    # Setup CRAFT on first load
    with st.spinner("Setting up CRAFT (first run downloads ~170 MB)…"):
        ready = setup_craft()
    if not ready:
        st.error("CRAFT setup failed. Check the logs.")
        return

    # Upload
    st.divider()
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
        help="Signboards, posters, road signs, documents — any Telugu text in a photo.",
    )

    if uploaded is None:
        st.info("👆 Upload an image to get started.")
        _show_sample_results()
        return

    # Show uploaded image
    col_orig, col_result = st.columns(2)
    with col_orig:
        st.subheader("Original")
        st.image(uploaded, use_container_width=True)

    # Run button
    if not api_key:
        st.warning("Enter your Sarvam AI API key in the sidebar to run.")
        return

    if st.button("▶ Run Translation Pipeline", type="primary",
                 use_container_width=True):
        _run_and_display(uploaded, api_key, col_result)


def _run_and_display(uploaded, api_key: str, col_result):
    """Save upload, run pipeline, display results."""
    ocr_reader = load_ocr_reader()

    # Save uploaded file to a temp location
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    try:
        status_box = st.empty()
        progress   = st.progress(0)

        status_box.info("🔍 Running CRAFT text detection…")
        progress.progress(10)

        with st.spinner("Running pipeline… this takes 30–90 seconds."):
            t0     = time.time()
            result = run_pipeline(tmp_path, api_key, ocr_reader)
            elapsed = time.time() - t0

        progress.progress(100)
        status_box.success(
            f"✅ Done in {elapsed:.0f}s  |  "
            f"{result['n_areas']} Telugu areas found  |  "
            f"Image type: {result['image_type']}"
        )

        inpainted_rgb      = result["inpainted_rgb"]
        translation_data   = result["translation_data"]

        # Show result image
        with col_result:
            st.subheader("Telugu Removed")
            st.image(inpainted_rgb, use_container_width=True)

        st.divider()

        # Download buttons
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            # Convert to JPEG bytes
            inpainted_bgr  = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
            _, img_bytes   = cv2.imencode(".jpg", inpainted_bgr,
                                          [cv2.IMWRITE_JPEG_QUALITY, 95])
            stem = os.path.splitext(uploaded.name)[0]
            st.download_button(
                "⬇ Download inpainted image",
                data=img_bytes.tobytes(),
                file_name=f"{stem}_inpainted.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
        with dl_col2:
            json_bytes = json.dumps(translation_data,
                                    ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "⬇ Download translations (JSON)",
                data=json_bytes,
                file_name=f"{stem}_translations.json",
                mime="application/json",
                use_container_width=True,
            )

        # Translation table
        if translation_data:
            st.subheader("Translation Results")
            for row in translation_data:
                with st.expander(
                    f"Area {row['area']}  —  {row['raw_ocr'][:60]}…"
                    if len(row['raw_ocr']) > 60 else
                    f"Area {row['area']}  —  {row['raw_ocr']}"
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.markdown("**Raw OCR (Telugu)**")
                    c1.write(row["raw_ocr"])
                    c2.markdown("**Corrected Telugu**")
                    c2.write(row["corrected_telugu"])
                    c3.markdown("**Tamil Translation**")
                    c3.write(row["tamil_translation"])
        else:
            st.info("No Telugu text was detected in this image.")

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _show_sample_results():
    """Show example before/after when no image is uploaded."""
    st.divider()
    st.subheader("Example output")
    st.markdown(textwrap.dedent("""
    | Input | Output |
    |---|---|
    | Telugu signboard photo | Inpainted image (text removed) + Tamil translation JSON |

    **Three translation rules applied per word:**
    - Native Telugu words → translated to Tamil
    - English loanwords in Telugu script → restored to English
    - Proper nouns and place names → transliterated to Tamil script
    """))


if __name__ == "__main__":
    main()
