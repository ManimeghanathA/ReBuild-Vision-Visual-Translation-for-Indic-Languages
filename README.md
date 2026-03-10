# 🌐 Visual Text Translation
### Telugu Scene Text → Tamil | Detection · OCR · Translation · Inpainting

> **2nd World Telugu Authors' Conference** — Project by  
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

# Inpainting Benchmark and Text Removal Strategy

## Overview

This component of the project focuses on **removing detected Telugu text from images while preserving the underlying background**.

After text detection and OCR are completed, the detected text regions are removed using **image inpainting** techniques. Multiple inpainting approaches are evaluated and compared against a **proposed text-aware inpainting strategy** designed specifically for scene text removal.

The goal is to reconstruct the background of the image **as naturally as possible after text removal**.

---

# Inpainting Pipeline

The inpainting stage operates on the text areas detected earlier in the system pipeline.

The process is:

```
Detected Text Areas
        │
        ▼
Stroke Mask Generation
        │
        ▼
Combined Inpainting Mask
        │
        ▼
Background Reconstruction
        │
        ▼
Inpainting Quality Evaluation
```

---

# Proposed Text-Aware Inpainting Strategy

Standard text removal approaches often remove **entire bounding boxes** around text regions. This results in large masked areas that include both text and surrounding background.

```
[ entire bounding box removed ]
```

This forces the inpainting algorithm to reconstruct a large portion of the image, which often introduces visual artifacts.

### Key Idea

The proposed approach removes **only the actual text strokes**, rather than the entire bounding box.

This is achieved by generating **stroke-level masks** that isolate the pixels corresponding to text characters.

Instead of masking:

```
[text + surrounding background]
```

we mask only:

```
[text strokes]
```

This significantly reduces the amount of missing image information that must be reconstructed.

---

# Stroke Mask Construction

For each detected text area:

1. The quadrilateral bounding boxes produced by CRAFT are used to locate character regions.
2. A **stroke mask** is generated that approximates the actual text pixels.
3. These masks are merged into a single **combined inpainting mask**.

The final mask contains:

```
white  → pixels to inpaint
black  → pixels to preserve
```

This ensures that background pixels remain untouched wherever possible.

---

# Multi-Region Inpainting

Images often contain multiple text regions.

Instead of performing inpainting separately for each region, the system:

1. Generates stroke masks for all detected text areas.
2. Merges them into a single mask.
3. Performs **one inpainting operation** over the entire image.

This avoids visible seams between separately reconstructed regions.

---

# Noise Box Cleanup

CRAFT detection may occasionally produce boxes that do not correspond to actual text.

To handle this, an additional cleanup step performs **noise box inpainting**, removing small artifacts introduced by these detections.

---

# Baseline Inpainting Methods

To evaluate the effectiveness of the proposed approach, several baseline inpainting methods were tested:

* Telea Inpainting (OpenCV)
* Navier–Stokes Inpainting (OpenCV)
* Patch-based Exemplar Inpainting
* LaMa Deep Learning Inpainting
* Stable Diffusion Inpainting

These methods provide a range of classical, patch-based, and deep-learning-based reconstruction techniques.

---

# Evaluation Metrics

## Overview

To quantitatively evaluate the effectiveness of the text removal and inpainting pipeline, we introduce a set of metrics that measure three key aspects of the system:

1. **Text Removal Effectiveness** — how well the algorithm removes detected text.
2. **Background Reconstruction Quality** — how realistically the removed regions are filled.
3. **Structural Continuity** — how smoothly the reconstructed region integrates with the surrounding image.

These metrics are combined into a final composite score called the **Text Removal Score (TRS)**.

Evaluation is performed on each processed image using the original image, the inpainted image, and the detected text masks.

---

# 1. Text Removal Rate (TRR)

**Text Removal Rate (TRR)** measures how much of the original detected text has been successfully removed after inpainting.

The metric is computed by comparing the number of text pixels before and after inpainting.

Conceptually:

[
TRR = 1 - \frac{\text{Remaining Text Pixels}}{\text{Original Text Pixels}}
]

Interpretation:

| TRR Value | Meaning                                        |
| --------- | ---------------------------------------------- |
| **1.0**   | All detected text was removed                  |
| **0.0**   | No text was removed                            |
| **< 0**   | OCR falsely detected new text after inpainting |

Negative values can occur when the OCR system mistakenly detects background textures as text.

---

# 2. Background Reconstruction Metrics

After text removal, the algorithm must reconstruct the background realistically. Two perceptual metrics are used to evaluate this.

## Masked LPIPS (Perceptual Similarity)

**LPIPS (Learned Perceptual Image Patch Similarity)** measures perceptual differences between two images using deep neural network features.

The metric is computed **only inside the inpainted region**.

Properties:

* Lower values indicate **more perceptually similar reconstruction**
* Captures human-perceived differences better than pixel metrics.

Typical range:

```
0 → identical reconstruction
>0.3 → noticeable visual difference
```

---

## Masked SSIM (Structural Similarity)

**SSIM (Structural Similarity Index)** measures similarity in image structure, contrast, and luminance.

This metric is also computed **only inside the removed text region**.

Properties:

* Range: **0 to 1**
* Higher values indicate **better structural reconstruction**

Interpretation:

| SSIM        | Quality                  |
| ----------- | ------------------------ |
| 0.95 – 1.0  | Excellent reconstruction |
| 0.85 – 0.95 | Good reconstruction      |
| < 0.85      | Noticeable artifacts     |

---

# 3. Structural Continuity

## Boundary Gradient Error

Removing text should not introduce visible seams or discontinuities. To measure this, we compute the **Boundary Gradient Error**.

This metric compares the gradient magnitude across the boundary between the original image and the inpainted region.

It measures how smoothly edges and textures propagate across the removed region.

Properties:

* Lower values indicate **smoother blending**
* Higher values indicate **visible artifacts or discontinuities**

This metric is especially useful for evaluating classical PDE-based inpainting methods which focus on gradient propagation.

---

# 4. Final Text Removal Score (TRS)

To provide a single overall evaluation metric, the individual metrics are combined into the **Text Removal Score (TRS)**.

TRS balances the competing objectives of:

* Removing text completely
* Preserving perceptual realism
* Maintaining structural continuity

Conceptually:

[
TRS = f(TRR, LPIPS, SSIM, GradientError)
]

Where:

* **Higher TRR improves the score**
* **Lower LPIPS improves the score**
* **Higher SSIM improves the score**
* **Lower Gradient Error improves the score**

Higher TRS values indicate better overall performance.

---

# Dataset Evaluation

The evaluation was performed on a dataset of **10 Telugu signboard images**. Each image was processed using multiple inpainting methods including:

* Telea Inpainting
* Navier-Stokes Inpainting
* Exemplar-based Inpainting
* LaMa Inpainting
* Stable Diffusion Inpainting
* **Proposed Inpainting Pipeline**

Metrics were computed for each image and averaged across the dataset.

Example averaged results:

| Method          | Avg TRR | Avg LPIPS | Avg SSIM | Avg Gradient Error | Avg TRS |
| --------------- | ------- | --------- | -------- | ------------------ | ------- |
| Proposed Method | -1.000  | 0.112     | 0.941    | 48.5               | 0.275   |
| Telea           | -1.378  | 0.115     | 0.940    | 21.7               | 0.216   |
| Navier-Stokes   | -1.301  | 0.114     | 0.940    | 19.9               | 0.222   |
| Exemplar        | -1.273  | 0.115     | 0.940    | 24.3               | 0.325   |

---

# Notes on OCR-based Evaluation

The evaluation relies on OCR to detect remaining text after inpainting. Because OCR models may occasionally detect background patterns as text, **TRR values can sometimes become negative**.

This does not invalidate comparisons between methods because the same OCR pipeline is applied consistently across all approaches.

---

# Benchmark Output

The benchmark produces:

```
full_inpainting_benchmark.csv
```

This file contains all evaluation metrics for each method and image.

Example structure:

| image    | method   | TRR  | Masked_LPIPS | Masked_SSIM | Boundary_Gradient_Error | TRS  |
|----------|----------|------|--------------|-------------|-------------------------|------|
| img1.jpg | telea    | ...  | ...          | ...         | ...                     | ...  |
| img1.jpg | exemplar | ...  | ...          | ...         | ...                     | ...  |
| img1.jpg | proposed | ...  | ...          | ...         | ...                     | ...  |

---

# Summary

The proposed inpainting strategy focuses on **minimizing the masked region** by targeting only the text strokes instead of entire text bounding boxes.

This approach:

* preserves more original background pixels
* reduces reconstruction complexity
* improves structural consistency in many cases

The benchmarking framework evaluates this method against several classical and deep learning inpainting approaches using a wide range of reconstruction metrics.

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
