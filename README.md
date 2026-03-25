# 🌐 Visual Text Translation
### Telugu Scene Text → Tamil | Detection · OCR · Translation · Inpainting

>  — Project by  
> Manimeghanath A [23BAI1278] · Shriram Narayanan [23BAI1095] · Koppesh P [23BAI1113]  
> Department of Computer Science and Engineering — VIT Chennai

---

## What This Does

Most video/image localization systems translate audio but leave **scene text** — signboards, posters, banners — in the source language. This project solves that.

Given an image with Telugu text embedded in the scene, the pipeline:

1. **Detects** text regions using CRAFT
2. **Reads** the text using EasyOCR (quad-rectified, perspective-corrected)
3. **Normalizes** OCR errors using Sarvam AI (character-level only, no rewrites)
4. **Translates** Telugu → Tamil using Sarvam AI (3 word-type rules)
5. **Removes** the original Telugu text via tight stroke-mask inpainting
6. *(Phase 3 — coming)* Re-renders Tamil text matching the original style

The result is an image that looks as if it was originally produced in Tamil.

---

## Pipeline Architecture

```
Image
  │
  ▼
CRAFT Text Detection
  │  ├─ load_craft_boxes()
  │  ├─ deduplicate_craft_boxes()   [Fix B2]
  │  ├─ build_text_areas()          [Fix B1]
  │  ├─ merge_overlapping_areas()   [Fix B3, B7]
  │  └─ purify_areas()
  │
  ▼
Quad-Rectified EasyOCR
  │  ├─ rectify_quad()              [Fix B9]
  │  ├─ enhance_for_ocr()           [Fix B12 — CLAHE]
  │  ├─ ocr_single_quad()           [Fix B10, B13]
  │  ├─ cluster_into_lines()        [Fix B11]
  │  ├─ reconstruct_area_sentence() [Fix B5, B6, B8]
  │  └─ deduplicate_ocr_across_areas() [Fix B4]
  │
  ▼
Sarvam AI Translation
  │  ├─ detect_image_type()         [auto — no manual setting needed]
  │  ├─ normalize_telugu_ocr()      [spell-check only, temperature=0]
  │  └─ translate_areas()           [3 word-type rules, single API call]
  │
  ▼
Stroke-Mask Inpainting
  │  ├─ build_stroke_mask_for_quad()  [Otsu + confidence fallback]
  │  ├─ is_telugu_quad()              [Telugu-only filter]
  │  ├─ inpaint_all_areas()           [TELEA inpainting]
  │  └─ inpaint_noise_boxes()         [cleanup pass for filtered punctuation]
  │
  ▼
Inpainted Image  +  Translation JSON
```

---

## Translation Rules

Three rules are enforced for every word in the translation prompt:

| Rule | Category | Example |
|------|----------|---------|
| **Rule 1** | Native Telugu → Translate to Tamil | కొంతమందికి → சிலருக்கு |
| **Rule 2** | English loanwords in Telugu script → Restore to English | పోలీస్ → Police |
| **Rule 3** | Proper nouns / place names → Transliterate to Tamil | రజనీకాంత్ → ரஜினிகாந்த் |

---

## Bugs Fixed

| ID | Description |
|----|-------------|
| B1 | Running-median cy + stride cap prevents area drift |
| B2 | CRAFT box deduplication (IoU containment > 70%) |
| B3 | Iterative merge with v_thresh=0.40, h_thresh=0.15 |
| B4 | Cross-area OCR deduplication |
| B5 | Consistent `sentence` key across all areas |
| B6 | All positions from `bbox_abs` (truly absolute coordinates) |
| B7 | v/h thresholds prevent cross-board merging |
| B8 | cx/cy derived from bbox_abs, not rectified space |
| B9 | Quad-rectified OCR per CRAFT box (warpPerspective) |
| B10 | conf_threshold=0.15 drops junk detections |
| B11 | Height-adaptive line clustering (h_compat > 0.40) |
| B12 | CLAHE on LAB L-channel for OCR quality |
| B13 | Pixel clip to area bbox before rectifying (ghost word prevention) |
| T1 | `<think>` strip with `flags=re.DOTALL` in normalize AND translate |
| T2 | `clean()` helper strips `<think>` from INPUT before translation |
| T3 | Normalize prompt: temperature=0, strict spell-checker only rules |
| I1 | Otsu confidence fallback: if `confidence < 0.15`, use full quad polygon |
| I2 | Auto inpaint radius by text height: h<40→5, h<80→8, h≥80→12 |
| I3 | Adaptive dilation: 3 iterations for h≥60px, 2 otherwise |
| I4 | Telugu-only inpainting: `is_telugu_quad()` skips English/non-Telugu quads |
| I5 | Context-aware no-match fallback: pure-Telugu area → erase, mixed area → row check |
| I6 | Noise box cleanup pass: 800px² threshold catches `-` separator, protects Hindi words |

---

## Known Limitations

1. **Large bold decorative fonts on coloured backgrounds** — EasyOCR often fails completely. This is a ceiling of the OCR engine, not the pipeline. Documented, not a bug.

2. **CRAFT misses some words** — Words not detected by CRAFT cannot be inpainted (e.g. a word over a face in a dense scene). CRAFT gap, not an inpainting bug.

3. **Red text on complex backgrounds** — Otsu threshold works on brightness. Red-on-green has ambiguous grayscale contrast. Partial survival possible. Colour-channel masking deferred to polish phase.

4. **Hindi / Devanagari text** — EasyOCR in Telugu mode does not return Devanagari. The pipeline uses a context-aware fallback (`area_is_pure_telugu`) and a strict 800px² size threshold in the noise cleanup pass to avoid erasing Hindi text. Works well in practice but may fail on very small Hindi text in purely Telugu areas.

5. **Scope constraints** — Offline processing only. Static or slow-moving text only. Short video clips (5–15 sec). Telugu → Tamil only. No real-time inference.

---

## Project Structure

```
visual_text_translation/
│
├── vtt/                          # Main package
│   ├── __init__.py               # Public API
│   ├── detection.py              # CRAFT loading, dedup, grouping, merging
│   ├── ocr.py                    # EasyOCR, line reconstruction, Telugu helpers
│   ├── translation.py            # Sarvam AI: type detection, normalize, translate
│   ├── inpainting.py             # Stroke masks, Telugu filter, TELEA inpainting
│   └── visualisation.py          # Matplotlib/CV2 visualisation helpers
│
├── scripts/
│   └── run_pipeline.py           # CLI: full end-to-end pipeline
│
├── data/
│   └── images/                   # Put your input images here
│
├── output/                       # Inpainted images + translation JSONs
├── result/                       # CRAFT .txt result files
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ManimeghanathA/ReBuild-Vision-Visual-Translation-for-Indic-Languages
cd ReBuild-Vision-Visual-Translation-for-Indic-Languages
```

### 2. Create virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Clone CRAFT and download weights

```bash
git clone https://github.com/clovaai/CRAFT-pytorch.git
cd CRAFT-pytorch

# Download model weights (requires gdown)
gdown https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ

# Patch for newer torchvision
sed -i 's/^from torchvision.models.vgg import model_urls/#&/' basenet/vgg16_bn.py
sed -i '25s/^/#/' basenet/vgg16_bn.py
sed -i 's/vgg16_bn(pretrained=True, freeze=True)/vgg16_bn(pretrained=False, freeze=True)/g' craft.py

cd ..
```

---

## Usage

### Step 1 — Run CRAFT detection

```bash
cd CRAFT-pytorch
python test.py \
    --trained_model=craft_mlt_25k.pth \
    --test_folder=../data/images \
    --cuda=True
cd ..
```

This generates `result/res_<image_name>.txt` files.

### Step 2 — Run the full pipeline

```bash
python scripts/run_pipeline.py \
    --image  data/images/img7.jpeg \
    --result CRAFT-pytorch/result/res_img7.txt \
    --api-key YOUR_SARVAM_API_KEY \
    --show
```

**Arguments:**

| Flag | Required | Description |
|------|----------|-------------|
| `--image` | ✅ | Path to input image |
| `--result` | ✅ | Path to CRAFT result `.txt` |
| `--api-key` | ✅ | Your [Sarvam AI](https://sarvam.ai) API key |
| `--output` | ❌ | Output directory (default: `output/`) |
| `--no-gpu` | ❌ | Force CPU mode for EasyOCR |
| `--show` | ❌ | Display before/after comparison plot |

### Output files

```
output/
  img7_inpainted.jpg               # Telugu text removed
  img7_translation_results.json    # Per-area: raw OCR, corrected, Tamil
```

---

## API Key

Get a free Sarvam AI API key at [sarvam.ai](https://sarvam.ai).

**Never commit your API key.** Store it in an environment variable:

```bash
export SARVAM_API_KEY=sk_xxx...
python scripts/run_pipeline.py --api-key $SARVAM_API_KEY ...
```

---

## Using the Package Directly

```python
import cv2
from vtt import (
    load_craft_boxes, deduplicate_craft_boxes,
    build_text_areas, merge_overlapping_areas, purify_areas,
    ocr_area, reconstruct_area_sentence, is_telugu_area,
    inpaint_all_areas, inpaint_noise_boxes,
)

img = cv2.cvtColor(cv2.imread('data/images/img7.jpeg'), cv2.COLOR_BGR2RGB)

# Detection
raw_boxes = load_craft_boxes('result/res_img7.txt')
boxes     = deduplicate_craft_boxes(raw_boxes)
areas     = merge_overlapping_areas(build_text_areas(boxes))
valid, _  = purify_areas(areas, img.shape)

# ... OCR, translation, inpainting
```

---

## Technologies

| Component | Tool |
|-----------|------|
| Text Detection | [CRAFT](https://github.com/clovaai/CRAFT-pytorch) |
| OCR | [EasyOCR](https://github.com/JaidedAI/EasyOCR) (Telugu + English) |
| OCR Normalization | [sarvam-m](https://sarvam.ai) |
| Translation | [sarvam-m](https://sarvam.ai) |
| Inpainting | OpenCV TELEA (`cv2.inpaint`) |
| Deep Learning | PyTorch |
| Image Processing | OpenCV, Pillow |

---

## Scope

**In-scope:**
- Image-level scene text localization and reconstruction
- Video-level extension for short clips (5–15 seconds) *(coming)*
- Static or slow-moving scene text
- Offline processing pipeline
- Language pair: **Telugu → Tamil**

**Out-of-scope:**
- Handwritten or heavily cursive text
- Extreme motion blur or heavy occlusion
- Real-time or streaming inference
- Full-length movie processing

---

## Future Work

- **Phase 3** — Tamil text rendering: re-render translated text matching original font style, colour, perspective and illumination
- Extension to additional language pairs
- Handling complex motion and dynamic scenes in video
- Joint audio-visual localization
- Real-time optimization

---

## Citation

If you use this work, please cite:

```
Manimeghanath A, Shriram Narayanan, Koppesh P.
"Detection & Reconstruction: End-to-End Visual Text Localization and Translation
for Images and Videos Using Deep Learning."
VIT Chennai, Department of Computer Science and Engineering, 2024.
```

---

## License

MIT License — see `LICENSE` for details.
