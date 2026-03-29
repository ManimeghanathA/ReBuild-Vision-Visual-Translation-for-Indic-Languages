# ReBuild Vision — Visual Translation for Indic Languages
### Telugu Scene Text → Tamil  |  Detection · OCR · Translation · Inpainting

<p align="center">
  <img src="docs/banner.png" alt="Before and After" width="800"/>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-system-architecture">Architecture</a> ·
  <a href="#-deep-technical-details">Technical Deep-Dive</a> ·
  <a href="#-installation">Installation</a> ·
  <a href="#-current-status--future-direction">Roadmap</a>
</p>

---

## What is this project, in plain English?

Imagine you're a Tamil speaker travelling through Andhra Pradesh.  
Every signboard, warning notice, poster, and public announcement is written in Telugu — a script you cannot read.

**ReBuild Vision** takes a photograph of any scene containing Telugu text and automatically:

1. **Finds** every Telugu word in the image (even on curved, tilted, or colourful signboards).
2. **Reads** the Telugu text using AI.
3. **Translates** it to Tamil — correctly handling place names, English loanwords, and native Telugu phrases differently.
4. **Erases** the original Telugu text from the photograph, leaving the background clean.
5. Returns both the **Tamil translation** and the **cleaned image**, ready for the translated Tamil text to be rendered on top.

The result is a photo that looks as if it was always written in Tamil.

---

## Problem Statement & Motivation

India is a land of 22 officially recognised languages spread across geographically contiguous states.  
A person crossing from Tamil Nadu into Andhra Pradesh encounters an immediate and practical barrier: **every piece of public text is in a script they cannot read**, even though the two languages share centuries of cultural and lexical overlap.

Existing machine translation services work on typed text. They do nothing for a person standing in front of a physical signboard.  
Existing image translation apps (Google Lens, etc.) overlay translated text but cannot adapt to the visual style of the original sign — font weight, colour, position — and critically, they do not **remove** the original text first, leading to cluttered, unreadable overlays.

This project was built to demonstrate a **clean, pipeline-based approach** to scene-text visual translation — one that:

- Is language-pair-agnostic (the same pipeline can be adapted to any Indic language pair).
- Respects the visual context of the sign (inpainting reconstructs the background rather than painting over it).
- Is open-source and reproducible without proprietary datasets.
- Produces output that is immediately useful for downstream Tamil text rendering (Phase 3, in progress).

---

## What the System Does

```
Input image (Telugu signboard / poster / document)
        │
        ▼
┌──────────────────┐
│  CRAFT Detection │  ← finds every text region as a quadrilateral polygon
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  EasyOCR         │  ← reads the Telugu characters from each region
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  Sarvam AI (sarvam-m)    │  ← normalises OCR errors, then translates Te→Ta
└────────┬─────────────────┘
         │
         ▼
┌──────────────────┐
│  Stroke Masking  │  ← builds a pixel-precise mask of the ink strokes
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  TELEA Inpainting│  ← reconstructs the background where text was
└────────┬─────────┘
         │
         ▼
Inpainted image + JSON translation results
```

---

## Key Capabilities

| Capability | Detail |
|---|---|
| **Script detection** | Detects Telugu Unicode range (U+0C00–U+0C7F) at word level |
| **Mixed-script safety** | Preserves Devanagari (Hindi) and clean English words — never erases them |
| **Three-rule translation** | Native Telugu → translate · English loanwords → restore to English · Proper nouns → transliterate to Tamil script |
| **OCR error correction** | sarvam-m fixes character-level recognition errors before translation |
| **Context-aware translation** | All text areas sent in one API call so the LLM has full document context |
| **Tight stroke masking** | Per-quad Otsu thresholding isolates only ink pixels; background texture preserved |
| **Batch processing** | Processes all images in a folder, or selectively by name |
| **Image-type auto-detection** | Classifies image as signboard / poster / road sign / newspaper / document and adjusts translation prompt accordingly |

---

## Challenges Faced During Development

### 1. CRAFT detection gap for large decorative text
CRAFT was trained primarily on document-scale text. For large (80px+), 3D, or heavily stylised billboard fonts, the detected quadrilaterals cover only ~60% of the actual glyph height. The top serifs and descenders fall outside all detected quads and are therefore never included in the stroke mask. Vertical quad expansion was evaluated (expanding each quad outward from its centroid by a ratio of its height) but did not reliably improve results — in some cases it expanded into adjacent non-text regions. This is an accepted limitation documented in the codebase.

### 2. Otsu threshold failing on coloured ink
The stroke mask generation uses Otsu thresholding on the grayscale rectified patch. For red ink on a white background (common on Telugu warning signs), the red channel collapses to a mid-gray in grayscale and is indistinguishable from the white background. The Otsu confidence metric (`between_class_variance / total_variance`) correctly detects this failure and falls back to a full quad polygon mask, but that polygon is only as good as the CRAFT detection (see challenge 1).

### 3. Ghost word bleeding from adjacent sign regions
Before fix B13, the perspective rectification of a CRAFT quad box sometimes captured pixels from a physically adjacent sign (e.g. a box at the boundary of two different boards). This produced "ghost words" — the OCR read text from the wrong sign. Fixed by zeroing all image pixels outside the area's bounding box before rectification.

### 4. Hindi text erasure in mixed-script areas
Early versions of the Telugu quad filter used a simple area-level classification: if the area contained mostly Telugu, all unmatched quads were erased. This caused Devanagari text (e.g. "सिंडिकेटबैंक" on a SyndicateBank poster) to be erased when it shared a bounding box with Telugu text. Fixed in v2 by `is_protected_non_telugu()` which hard-protects both Devanagari characters and clean English words (>4 chars, ≥70% alphabetic), combined with a quad-height-scaled proximity threshold for the no-match case.

### 5. TELEA inpainting artefacts on solid-colour backgrounds
TELEA reconstructs missing pixels by propagating values from the mask boundary inward. For red ink on a white background, this works well. For colored text on a matching-colored background (e.g. the Eenadu logo — blue 3D text on a red background), the boundary pixels are a mix of the two colors, and TELEA produces a visible blotchy smear instead of a clean fill. LaMa (a deep learning inpainter) was evaluated but produced visible blurring on signboard-type images. TELEA is retained as it produces the best results on the majority of test images.

### 6. Cross-area OCR duplication
When two text areas have overlapping bounding boxes (e.g. a heading that extends slightly into the next area's region), EasyOCR reads the same physical word twice. Fixed by cross-area IoU deduplication that keeps the higher-confidence copy.

### 7. EasyOCR confidence calibration
EasyOCR's raw confidence scores for Telugu are not well-calibrated — it produces low-confidence scores for correct detections on bold fonts and high-confidence scores for some hallucinations. A minimum threshold of 0.15 was chosen after empirical evaluation across the test set. Lowering it increases recall but introduces noise; raising it misses valid detections.

---

## System Architecture

```
Visual Text Translation/
├── scripts/
│   ├── run_pipeline.py          ← CLI entry point (batch + selective)
│   └── vtt/                     ← Python package
│       ├── __init__.py          ← public API surface
│       ├── detection.py         ← CRAFT box loading, dedup, grouping, merging
│       ├── ocr.py               ← EasyOCR wrapper, line reconstruction
│       ├── translation.py       ← Sarvam AI integration
│       ├── inpainting.py        ← stroke mask + TELEA inpainting (v2)
│       └── visualisation.py     ← debug visualisations
├── CRAFT-pytorch/               ← submodule (clovaai/CRAFT-pytorch)
│   ├── craft_mlt_25k.pth        ← model weights (Git LFS)
│   └── result/                  ← CRAFT output .txt files (gitignored)
├── data/
│   └── images/                  ← input images (Git LFS)
├── output/                      ← inpainted images + JSON (gitignored)
├── docs/                        ← assets for README
├── app.py                       ← Streamlit web demo
├── requirements.txt
├── .gitignore
├── .gitattributes               ← Git LFS rules
└── Visual_Translation.ipynb    ← interactive notebook (main pipeline)
```

---

## Deep Technical Details

> **[→ Jump to full technical breakdown](#detection--craft)**

### Detection — CRAFT

CRAFT (Character Region Awareness for Text Detection) is a character-level text detector trained to produce region score maps and affinity score maps. The pipeline uses the pre-trained `craft_mlt_25k.pth` weights (trained on a multilingual dataset of 25,000 images).

**What CRAFT outputs:** For each image, CRAFT produces a `.txt` file containing one quadrilateral per detected text region. Each quad is defined by four `(x, y)` corner points in pixel coordinates.

**Box deduplication (B2):** CRAFT is run at multiple scales internally and can produce nested duplicate boxes. The pipeline deduplicates by containment — if box A is >70% contained within box B (measured by intersection area / A area), box A is suppressed. Boxes are sorted largest-first so that the containing box is always evaluated before the contained one.

**Area grouping (B1):** Individual character-level quads are grouped into text-line areas using a running-median center-y with a stride cap of 1.2× median box height. This prevents area drift on multi-line text blocks.

**Area merging (B3, B7):** Adjacent areas are merged iteratively if they have >40% vertical overlap AND >15% horizontal overlap. The horizontal threshold prevents merging text from different columns of a board.

**Area purification:** Areas smaller than 0.04% of image area, or single-box areas smaller than 0.2% of image area, are classified as noise and excluded from OCR.

### OCR — EasyOCR

EasyOCR is run in `['te', 'en']` mode (Telugu + English). Each CRAFT quad is processed individually rather than running OCR on the whole image, for two reasons: (1) perspective rectification corrects tilt, improving recognition accuracy on tilted signboards; (2) it allows per-quad confidence filtering and spatial containment testing.

**Quad rectification (B9):** Each quad's four corners are ordered (TL, TR, BR, BL) and a perspective transform matrix M is computed mapping the quad to a flat rectangle at 2× upscale. EasyOCR reads the rectified patch. Detected word bounding boxes are then inverse-mapped back to image coordinates using M⁻¹.

**Ghost word prevention (B13):** Before rectification, all pixels outside the area's bounding box are zeroed. This prevents adjacent signs' text from bleeding into a quad's rectified patch when the quad's physical edge overlaps a neighbouring region.

**CLAHE enhancement (B12):** The rectified patch is contrast-enhanced using CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel of LAB color space, improving OCR quality on faded or unevenly lit text.

**Line reconstruction (B11):** OCR words are clustered into lines using height-adaptive vertical tolerance. Words whose height ratio is <0.40 relative to the line median are assigned to a new line. This correctly separates headlines from subtitles.

**Cross-area deduplication (B4):** After all areas are OCR'd, words from different areas that overlap by >50% (by area) are deduplicated, keeping the higher-confidence copy.

### Translation — Sarvam AI

All translation is performed via the Sarvam AI API using the `sarvam-m` model, a multilingual LLM with strong Indic language support.

**Image type detection:** Before translation, the concatenated OCR text is sent to sarvam-m with a classification prompt. The model classifies the image as one of: `signboard`, `newspaper`, `road_sign`, `poster`, `document`. This label is used in the translation prompt to provide context.

**OCR normalisation:** A dedicated sarvam-m call fixes character-level OCR errors (digits substituted for visually similar Telugu letters, broken vowel signs, split conjunct consonants). The prompt explicitly forbids paraphrasing or rewriting.

**Three-rule translation:** All areas are translated in a single API call (full document context). The prompt enforces three rules per word:
- Native Telugu → translate to Tamil
- English words written in Telugu script (e.g. పోలీస్) → restore to English
- Proper nouns and place names → transliterate to Tamil script

**Think-tag stripping:** sarvam-m sometimes prepends `<think>...</think>` chain-of-thought blocks to its output. These are stripped with a regex before using the result.

### Stroke Mask Generation

For each CRAFT quad classified as Telugu, a pixel-precise mask of the ink strokes is generated:

1. The quad patch is rectified at 2× upscale (same as OCR, for accuracy).
2. Otsu thresholding is applied to the grayscale patch. The Otsu confidence metric (between-class variance / total variance) is computed. If confidence < 0.15 (low-contrast patch, common for colored ink on matching-colored backgrounds), the full quad polygon is used as a fallback mask.
3. A polarity check ensures ink pixels are white (text-on-background, not background-on-text).
4. The binary mask is dilated by 2 pixels to catch anti-aliased stroke edges.
5. The mask pixels are inverse-mapped back to image space using M⁻¹.

**Telugu quad filter (v2):** Before building the stroke mask, `is_telugu_quad()` decides whether each CRAFT quad should be erased:
- If an OCR word centre falls inside the quad: erase if Telugu, skip if Devanagari or clean English (hard-protected).
- If no OCR word matches (EasyOCR missed the quad): in a pure-Telugu area, erase unconditionally (catches large/decorative fonts). In a mixed-script area, erase only if a Telugu word centre is within 3.5× the quad's own height — this scales the threshold to the actual size of the unmatched quad rather than anchoring it to small body-text word heights.

### Inpainting — TELEA

The stroke mask is dilated further (2–3 iterations depending on text height) and used as input to OpenCV's TELEA inpainting algorithm. TELEA (Fast Marching Method-based) reconstructs missing pixels by propagating color values from the mask boundary inward along a fast marching front. The inpaint radius is scaled to text height: 5 px for small text (<40 px tall), 8 px for medium, 12 px for large.

A post-processing noise cleanup pass erases any CRAFT quads that were classified as noise during area grouping (e.g. isolated dash separators) if they are ≤800 px² and fall within a confirmed Telugu area's bounding box.

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (CPU mode supported with `--no-gpu`)
- Git LFS installed (`git lfs install`)
- A Sarvam AI API key ([get one at sarvam.ai](https://sarvam.ai))

### 1. Clone the repository

```bash
git lfs install          # run once, installs LFS hooks
git clone https://github.com/ManimeghanathA/ReBuild-Vision-Visual-Translation-for-Indic-Languages.git
cd ReBuild-Vision-Visual-Translation-for-Indic-Languages
```

### 2. Set up the environment

```bash
python -m venv vision
# Windows
vision\Scripts\activate
# Linux / macOS
source vision/bin/activate

pip install -r requirements.txt
```

### 3. Set up CRAFT

CRAFT-pytorch is included as a Git submodule. The model weights are tracked via Git LFS and will download automatically on clone.

```bash
# If the weights didn't download (LFS not installed before clone):
git lfs pull
```

**Patch CRAFT for modern torchvision (Windows PowerShell):**
```powershell
(Get-Content CRAFT-pytorch/basenet/vgg16_bn.py) `
    -replace '^from torchvision.models.vgg import model_urls', '# from torchvision.models.vgg import model_urls' `
    | Set-Content CRAFT-pytorch/basenet/vgg16_bn.py

$file = Get-Content CRAFT-pytorch/basenet/vgg16_bn.py
$file[24] = "#" + $file[24]
$file | Set-Content CRAFT-pytorch/basenet/vgg16_bn.py

(Get-Content CRAFT-pytorch/craft.py) `
    -replace 'vgg16_bn\(pretrained=True, freeze=True\)', 'vgg16_bn(pretrained=False, freeze=True)' `
    | Set-Content CRAFT-pytorch/craft.py
```

**Linux / macOS:**
```bash
sed -i 's/^from torchvision.models.vgg import model_urls/#&/' CRAFT-pytorch/basenet/vgg16_bn.py
sed -i '25s/^/#/' CRAFT-pytorch/basenet/vgg16_bn.py
sed -i 's/vgg16_bn(pretrained=True, freeze=True)/vgg16_bn(pretrained=False, freeze=True)/g' CRAFT-pytorch/craft.py
```

### 4. Run CRAFT detection

```bash
cd CRAFT-pytorch
python test.py \
    --trained_model=craft_mlt_25k.pth \
    --test_folder=../data/images \
    --cuda=True          # use --cuda=False for CPU
cd ..
```

### 5. Run the pipeline

**All images:**
```powershell
python scripts/run_pipeline.py `
    --image-dir  data/images `
    --result-dir CRAFT-pytorch/result `
    --api-key    YOUR_SARVAM_API_KEY
```

**Specific images only:**
```powershell
python scripts/run_pipeline.py `
    --image-dir  data/images `
    --result-dir CRAFT-pytorch/result `
    --api-key    YOUR_SARVAM_API_KEY `
    --select     img1,img3,img7
```

**Single image (legacy mode):**
```powershell
python scripts/run_pipeline.py `
    --image  data/images/img7.jpeg `
    --result CRAFT-pytorch/result/res_img7.txt `
    --api-key YOUR_SARVAM_API_KEY `
    --show
```

Outputs are written to `output/`:
- `{stem}_inpainted.jpg` — image with Telugu text removed
- `{stem}_translation_results.json` — per-area OCR, corrected Telugu, and Tamil translation

### 6. Web demo (Streamlit)

```bash
streamlit run app.py
```

See [Deployment](#deployment) for hosting on Hugging Face Spaces.

---

## Project Structure Reference

```
scripts/vtt/
├── __init__.py        Public API — imports from all modules
├── detection.py       load_craft_boxes, deduplicate, build_text_areas,
│                      merge_overlapping_areas, purify_areas
├── ocr.py             enhance_for_ocr, rectify_quad, ocr_single_quad,
│                      ocr_area, reconstruct_area_sentence,
│                      deduplicate_ocr_across_areas, Telugu helpers
├── translation.py     detect_image_type, normalize_telugu_ocr,
│                      translate_areas, IMAGE_TYPE_DESCRIPTIONS
├── inpainting.py      build_stroke_mask_for_quad, build_stroke_mask_for_area,
│                      is_telugu_quad (v2), inpaint_area, inpaint_all_areas,
│                      inpaint_noise_boxes
└── visualisation.py   show_craft_results, visualize_areas,
                       visualize_final_areas, visualise_stroke_masks,
                       visualize_inpainted
```

---

## Current Status & Future Direction

### What works well (Phase 1 & 2 complete)

- ✅ Full detection → OCR → translation → inpainting pipeline
- ✅ Mixed-script safety (Devanagari, English preserved)
- ✅ Three-rule translation with full document context
- ✅ Batch and selective image processing
- ✅ Streamlit web demo

### Known limitations (documented, accepted for now)

- CRAFT underestimates quad coverage for large (80px+) 3D/stylised text — glyph top/bottom strokes outside detected quads are not erased
- Colored ink on matching-colored background (e.g. blue 3D text on red) produces imperfect Otsu masking
- EasyOCR accuracy drops on very bold decorative headline fonts

### Phase 3 — Tamil text rendering (in progress)

- [ ] Font style estimation (weight, size, color) from original text region
- [ ] Tamil text rendering back onto the inpainted image, matching original visual style
- [ ] Word-level bounding box alignment for precise placement

### Phase 4 — Generalisation

- [ ] Extend to additional Indic language pairs (Kannada→Tamil, Hindi→Tamil)
- [ ] Explore CRAFT fine-tuning on Indic scripts for better quad coverage
- [ ] Replace TELEA with a learned inpainting model for colored backgrounds

---

## Deployment

### Hugging Face Spaces (recommended — free GPU)

```bash
# Create a new Space at huggingface.co/spaces
# Runtime: GPU (T4 small, free tier)
# SDK: Streamlit

# Push this repo to the Space:
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/vtt-demo
git push space main
```

The `app.py` Streamlit app downloads CRAFT weights at first startup (one-time, ~170 MB) and caches them.

---

## Citation

If you use this work in research, please cite:

```bibtex
@misc{vtt2025,
  title  = {ReBuild Vision: Visual Translation for Indic Languages},
  author = {Manimeghanath A and Shriram},
  year   = {2025},
  url    = {https://github.com/ManimeghanathA/ReBuild-Vision-Visual-Translation-for-Indic-Languages}
}
```

---

## Acknowledgements

- [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) — Clova AI Research
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — Jaided AI
- [Sarvam AI](https://sarvam.ai) — for the sarvam-m multilingual LLM

---

*Phase 3 (Tamil text rendering) — documentation will be added here once complete.*
