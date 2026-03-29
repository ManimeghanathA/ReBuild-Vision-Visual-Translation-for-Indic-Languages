# ReBuild Vision — Visual Translation for Indic Languages
### Telugu Scene Text → Tamil  |  Detection · OCR · Translation · Inpainting

<p align="center">
  <img src="docs/banner.png" alt="Before and After — Telugu signboard with text removed" width="820"/>
</p>

<p align="center">
  <a href="#-what-is-this-project">About</a> ·
  <a href="#-try-the-live-demo">Live Demo</a> ·
  <a href="#%EF%B8%8F-installation--local-setup">Local Setup</a> ·
  <a href="#%EF%B8%8F-system-architecture">Architecture</a> ·
  <a href="#-deep-technical-details">Technical Deep-Dive</a> ·
  <a href="#-current-status--future-direction">Roadmap</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/EasyOCR-Telugu%20%2B%20English-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Sarvam%20AI-sarvam--m-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square"/>
</p>

---

## 🌐 What is this project?

Imagine you're a Tamil speaker travelling through Andhra Pradesh. Every signboard, warning notice, poster, and public announcement is written in Telugu — a script you cannot read.

**ReBuild Vision** takes a photograph of any scene containing Telugu text and automatically:

1. **Finds** every Telugu word in the image — even on curved, tilted, or colourful signboards.
2. **Reads** the Telugu text using AI-powered OCR.
3. **Translates** it to Tamil, correctly handling place names, English loanwords, and native Telugu phrases with three different rules.
4. **Erases** the original Telugu text from the photograph using stroke-level inpainting, leaving the background clean.
5. Returns both the **Tamil translation** and the **cleaned image**, ready for translated Tamil text to be rendered on top.

---

## 🚀 Try the Live Demo

> **Web app:**
> <!-- FILL IN AFTER DEPLOYMENT -->
> `https://huggingface.co/spaces/YOUR_USERNAME/vtt-demo`

Upload any Telugu image, paste your [Sarvam AI API key](https://sarvam.ai), and get:
- The inpainted image (Telugu text removed) — downloadable as JPG
- Translation results — downloadable as JSON

---

## 💡 Problem Statement & Motivation

India is a land of 22 officially recognised languages spread across geographically contiguous states. A person crossing from Tamil Nadu into Andhra Pradesh encounters an immediate and practical barrier: **every piece of public text is in a script they cannot read**, even though the two languages share centuries of cultural and lexical overlap.

Existing machine translation services work on typed text — they do nothing for a person standing in front of a physical signboard. Existing image translation apps (Google Lens, etc.) overlay translated text but do not **remove** the original text first, leading to cluttered overlays.

This project demonstrates a **clean, modular pipeline** for scene-text visual translation that:
- Is language-pair-agnostic (adaptable to any Indic language pair).
- Reconstructs the background rather than painting over it.
- Is fully open-source and reproducible without proprietary datasets.
- Produces output immediately useful for downstream Tamil text rendering (Phase 3, in progress).

---

## ✅ Key Capabilities

| Capability | Detail |
|---|---|
| **Script detection** | Detects Telugu Unicode range U+0C00–U+0C7F at word level |
| **Mixed-script safety** | Preserves Devanagari (Hindi) and clean English words — never erases them |
| **Three-rule translation** | Native Telugu → translate · English loanwords → restore to English · Proper nouns → transliterate to Tamil script |
| **OCR error correction** | sarvam-m fixes character-level recognition errors before translation |
| **Context-aware translation** | All text areas sent in one API call so the LLM sees full document context |
| **Tight stroke masking** | Per-quad Otsu thresholding isolates only ink pixels; background texture preserved |
| **Batch + selective processing** | All images in a folder, or specific images by name via `--select` |
| **Image-type detection** | Auto-classifies image as signboard / poster / road sign / newspaper / document |

---

## ⚙️ System Architecture

```
Your image (Telugu signboard / poster / document)
           │
           ▼
┌──────────────────────┐
│  CRAFT Detection     │  ← Finds every text region as a quadrilateral polygon
└──────────┬───────────┘   (cloned + run separately — see Step 4 below)
           │
           ▼
┌──────────────────────┐
│  EasyOCR             │  ← Reads Telugu characters (2× upscale + CLAHE)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────┐
│  Sarvam AI  (sarvam-m)       │  ← Normalises OCR errors → Translates Te→Ta
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────┐
│  Stroke Mask         │  ← Pixel-precise mask of just the ink strokes
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────────────────┐
│  TELEA Inpainting    │     │  Phase 3 — Tamil rendering  │
│  (text erased)       │────▶│  <!-- IN PROGRESS — TBD --> │
└──────────────────────┘     └─────────────────────────────┘
```

### Repository layout

```
ReBuild-Vision-Visual-Translation-for-Indic-Languages/
│
├── scripts/
│   ├── run_pipeline.py        ← CLI entry point (batch + selective)
│   └── vtt/                   ← our Python package
│       ├── __init__.py
│       ├── detection.py
│       ├── ocr.py
│       ├── translation.py
│       ├── inpainting.py      ← v2 with is_protected_non_telugu()
│       └── visualisation.py
│
├── data/images/               ← your input images (Git LFS)
├── output/                    ← generated outputs (gitignored)
├── docs/                      ← README assets
├── app.py                     ← Streamlit web demo
├── Visual_Translation.ipynb  ← interactive notebook
├── requirements.txt
├── .gitignore
└── .gitattributes             ← Git LFS rules

CRAFT-pytorch/                 ← NOT in this repo — cloned separately (Step 4)
    craft_mlt_25k.pth          ← downloaded separately (~170 MB)
    result/                    ← CRAFT output .txt files (generated, gitignored)
```

---

## 🛠️ Installation & Local Setup

> **Time:** ~10 min first time (EasyOCR model download)
> **Needs:** Python 3.10+, Git — GPU recommended but not required

### Step 1 — Clone this repository

```bash
git clone https://github.com/ManimeghanathA/ReBuild-Vision-Visual-Translation-for-Indic-Languages.git
cd ReBuild-Vision-Visual-Translation-for-Indic-Languages
```

### Step 2 — Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv vision
vision\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv vision
source vision/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

EasyOCR will download Telugu + English models (~300 MB) the first time it runs.

### Step 4 — Clone CRAFT and download weights

CRAFT is **not inside this repository** — you clone it separately, at the same level as this project folder:

```bash
# From inside the project root:
git clone https://github.com/clovaai/CRAFT-pytorch.git

# Download the model weights into the CRAFT folder (~170 MB):
cd CRAFT-pytorch
pip install gdown
gdown https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ
cd ..
```

After this your folder layout should look like:
```
Visual Text Translation/       ← this repo
CRAFT-pytorch/                 ← just cloned
    craft_mlt_25k.pth          ← just downloaded
```

### Step 5 — Patch CRAFT for modern torchvision

Run these once. They fix a compatibility break in newer versions of torchvision.

**Windows (PowerShell):**
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

### Step 6 — Add your images

Place your Telugu images (`.jpg`, `.jpeg`, `.png`) inside `data/images/`.

### Step 7 — Run CRAFT detection

```bash
cd CRAFT-pytorch
python test.py \
    --trained_model=craft_mlt_25k.pth \
    --test_folder=../data/images \
    --cuda=True
# use --cuda=False if you don't have a GPU
cd ..
```

Results are written to `CRAFT-pytorch/result/res_<imagename>.txt`.

### Step 8 — Run the pipeline

**All images:**
```powershell
# Windows
python scripts/run_pipeline.py `
    --image-dir  data/images `
    --result-dir CRAFT-pytorch/result `
    --api-key    YOUR_SARVAM_API_KEY
```
```bash
# Linux / macOS
python scripts/run_pipeline.py \
    --image-dir  data/images \
    --result-dir CRAFT-pytorch/result \
    --api-key    YOUR_SARVAM_API_KEY
```

**Specific images only** (comma-separated stems, no file extension):
```powershell
python scripts/run_pipeline.py `
    --image-dir  data/images `
    --result-dir CRAFT-pytorch/result `
    --api-key    YOUR_SARVAM_API_KEY `
    --select     img1,img3,img7
```

**Single image (legacy):**
```powershell
python scripts/run_pipeline.py `
    --image  data/images/img7.jpeg `
    --result CRAFT-pytorch/result/res_img7.txt `
    --api-key YOUR_SARVAM_API_KEY `
    --show
```

**All flags:**

| Flag | Description | Default |
|---|---|---|
| `--image-dir` | Folder of input images | — |
| `--result-dir` | Folder of CRAFT `.txt` result files | — |
| `--select` | Comma-separated stems to process | all images |
| `--image` | Single image path (legacy mode) | — |
| `--result` | Single CRAFT result path (legacy mode) | — |
| `--api-key` | Sarvam AI API key | required |
| `--output` | Output directory | `output/` |
| `--no-gpu` | Force CPU for EasyOCR | GPU if available |
| `--show` | Show before/after plot | off |

### Step 9 — View outputs

```
output/
├── img1_inpainted.jpg              ← Telugu text erased
├── img1_translation_results.json  ← OCR + corrected Telugu + Tamil
└── ...
```

Sample JSON output:
```json
[
  {
    "area_index": 0,
    "area_bbox": [45, 120, 680, 195],
    "raw_ocr": "2వ ప్రపంచ తెలుగు రచయితల మహాసభలు",
    "corrected_telugu": "2వ ప్రపంచ తెలుగు రచయితల మహాసభలు",
    "tamil_translation": "2வது உலக தெலுங்கு எழுத்தாளர்கள் மகாசபை"
  }
]
```

### Step 10 — Run the web demo locally (optional)

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Upload an image, enter your Sarvam API key, click Run.

---

## 📖 Deep Technical Details

### Detection — CRAFT

CRAFT (Character Region Awareness for Text Detection) produces region score maps and affinity score maps. We use `craft_mlt_25k.pth` — pre-trained on 25,000 multilingual images.

**What CRAFT outputs:** A `.txt` file per image with one quadrilateral (four `x,y` corner points) per detected text region.

**Box deduplication (B2):** CRAFT can produce nested duplicates at multiple scales. We suppress any box that is >70% contained inside a larger box (largest-first pass).

**Area grouping (B1):** Quads are grouped into text-line areas using running-median center-y with a stride cap of 1.2× median box height. Prevents lines from drifting together on multi-line blocks.

**Area merging (B3, B7):** Adjacent areas are merged when they have >40% vertical overlap AND >15% horizontal overlap. The horizontal threshold prevents merging columns.

**Area purification:** Areas smaller than 0.04% of image area, or single-box areas smaller than 0.2%, are noise and excluded from OCR.

### OCR — EasyOCR

Runs in `['te', 'en']` mode. Each CRAFT quad is processed individually for tilt correction and per-quad confidence filtering.

**Quad rectification (B9):** Corners ordered TL→TR→BR→BL, perspective transform M computed, patch warped to a flat rectangle at 2× upscale. Words inverse-mapped back with M⁻¹.

**Ghost word prevention (B13):** Pixels outside the area bbox are zeroed before rectification, preventing adjacent signs from bleeding in.

**CLAHE enhancement (B12):** Applied to the L channel of LAB color space to improve recognition on faded or uneven text.

**Line reconstruction (B11):** Words clustered into lines using height-adaptive vertical tolerance. Height ratio < 0.40 relative to line median → new line (separates headlines from subtitles).

**Cross-area deduplication (B4):** Words from different areas overlapping by >50% are deduplicated, keeping the higher-confidence copy.

### Translation — Sarvam AI

Uses `sarvam-m` via the Sarvam AI API.

**Image type detection:** OCR text → sarvam-m classifies as `signboard`, `newspaper`, `road_sign`, `poster`, or `document` → used in translation prompt for context.

**OCR normalisation:** Dedicated call fixes character-level errors. Prompt explicitly forbids paraphrasing.

**Three-rule translation (single call, full document context):**
- Native Telugu → **translate** to Tamil
- English in Telugu script (పోలీస్) → **restore** to English (Police)
- Proper nouns / places → **transliterate** to Tamil script

### Stroke Mask Generation

For each Telugu-classified quad:
1. Rectify at 2× upscale.
2. Otsu threshold on grayscale.
3. Confidence check (between-class variance / total variance): if < 0.15, fall back to full polygon mask.
4. Polarity check: ink = white.
5. Dilate 2 px for anti-aliased edges.
6. Inverse-map back to image space.

**`is_telugu_quad()` v2:** Matched OCR word → erase if Telugu, skip if Devanagari or clean English. No match in pure-Telugu area → erase unconditionally (catches large fonts OCR misses). No match in mixed area → erase only if within 3.5× this quad's own height from a Telugu word centre. The quad-height-relative scale was the key fix for Hindi logo erasure.

### Inpainting — TELEA

Stroke mask dilated 2–3 iterations, passed to OpenCV TELEA. Inpaint radius: 5 px (text < 40 px), 8 px (< 80 px), 12 px (≥ 80 px). Post-processing pass cleans isolated CRAFT noise quads ≤ 800 px².

---

## 🔍 Challenges During Development

| # | Challenge | Status |
|---|---|---|
| 1 | CRAFT quad undercoverage on large 3D/stylised fonts | Accepted limitation |
| 2 | Otsu fails on coloured ink (red on white) | Accepted limitation |
| 3 | Ghost words from adjacent sign bleeding | Fixed — B13 pixel clipping |
| 4 | Hindi text erased in mixed-script areas | Fixed — v2 `is_protected_non_telugu()` |
| 5 | TELEA smear on solid-colour backgrounds | Accepted — LaMa evaluated, rejected (blurring) |
| 6 | Cross-area OCR duplication | Fixed — IoU dedup (B4) |
| 7 | EasyOCR confidence miscalibration on bold fonts | Accepted — threshold 0.15 |

---

## 📊 Current Status & Future Direction

### Phase 1 & 2 — Complete ✅

- ✅ Detection → OCR → translation → inpainting pipeline
- ✅ Mixed-script safety (Devanagari, English preserved)
- ✅ Three-rule translation with full document context
- ✅ Batch + selective CLI processing
- ✅ Streamlit web demo on Hugging Face Spaces

### Phase 3 — Tamil Text Rendering *(in progress)*

<!-- IN PROGRESS — results and details will be added here once Phase 3 is complete -->

- [ ] Font style estimation (weight, size, colour) from original text region
- [ ] Tamil text rendering back onto the inpainted image matching original visual style
- [ ] Word-level bounding box alignment

### Phase 4 — Generalisation *(future)*

- [ ] Additional Indic language pairs (Kannada→Tamil, Hindi→Tamil)
- [ ] CRAFT fine-tuning on Indic scripts for large text coverage
- [ ] Learned inpainting model for solid-colour backgrounds

---

## 🌍 Live Demo & Deployment

> **Web app URL:**
> <!-- FILL IN AFTER DEPLOYMENT: https://huggingface.co/spaces/YOUR_USERNAME/vtt-demo -->

The app runs on **Hugging Face Spaces** (free GPU tier). CRAFT is downloaded automatically at first startup — it is not bundled in this repository.

For full hosting instructions see `DEPLOYMENT_GUIDE.md`.

---

## 📋 Module Reference

```
scripts/vtt/
├── detection.py     load_craft_boxes · deduplicate_craft_boxes
│                    build_text_areas · merge_overlapping_areas
│                    purify_areas · area_bbox · generate_area_mask
├── ocr.py           enhance_for_ocr · rectify_quad · ocr_single_quad
│                    ocr_area · reconstruct_area_sentence
│                    deduplicate_ocr_across_areas · Telugu helpers
├── translation.py   detect_image_type · normalize_telugu_ocr
│                    translate_areas · IMAGE_TYPE_DESCRIPTIONS
├── inpainting.py    build_stroke_mask_for_quad · build_stroke_mask_for_area
│                    is_telugu_quad (v2) · is_protected_non_telugu
│                    inpaint_area · inpaint_all_areas · inpaint_noise_boxes
└── visualisation.py show_craft_results · visualize_areas
                     visualize_final_areas · visualise_stroke_masks
                     visualize_inpainted
```

---

## 📎 Citation

```bibtex
@misc{vtt2025,
  title  = {ReBuild Vision: Visual Translation for Indic Languages},
  author = {Manimeghanath A and Shriram},
  year   = {2025},
  url    = {https://github.com/ManimeghanathA/ReBuild-Vision-Visual-Translation-for-Indic-Languages}
}
```

---

## 🙏 Acknowledgements

- [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) — Clova AI Research
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — Jaided AI
- [Sarvam AI](https://sarvam.ai) — `sarvam-m` multilingual LLM

---

*Phase 3 (Tamil text rendering) — documentation will be added here once complete.*
