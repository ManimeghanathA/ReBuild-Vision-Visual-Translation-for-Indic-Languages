# ReBuild Vision Technical README

This document explains the application design, architecture, pipeline, and code responsibilities in detail. It is written as a project walkthrough, not just a list of function names.

## 1. Application Design

ReBuild Vision is designed as a modular image-processing pipeline. Each module owns one stage of the visual text translation problem:

```text
Detection -> OCR -> Normalization/Translation -> Masking -> Inpainting -> Future Rendering
```

The current application has two ways to run the same backend logic:

1. **CLI:** `scripts/run_pipeline.py`
2. **Local browser UI:** `local_web_server.py` plus `frontend/`

Both surfaces use the same Python package under `scripts/vtt/`. This is important because the UI is not a separate implementation. It is only a user-facing wrapper around the same detection, OCR, translation, and inpainting code.

## 2. Data Model Through the Pipeline

The pipeline starts with an image and gradually adds structure:

```text
image path
  -> RGB image array
  -> CRAFT quadrilateral boxes
  -> grouped text areas
  -> OCR words with absolute coordinates
  -> processed Telugu areas
  -> optional corrected Telugu + Tamil translation
  -> Telugu stroke masks
  -> inpainted image
```

The most important object is a `processed_area`. It represents one Telugu-containing text area after OCR:

```python
{
    "area_idx": int,
    "area_bbox": (x1, y1, x2, y2),
    "area_quads": [quad1, quad2, ...],
    "sentence": [word1, word2, ...],
    "full_text": "recognized OCR text",
    "telugu_words": [...],
    "other_words": [...],
    "raw_ocr": [...],
    "mask": np.ndarray,
    "corrected_telugu": "...",      # added only when translation is enabled
    "tamil_translation": "...",     # added only when translation is enabled
}
```

Each word in `sentence` keeps its text and geometry:

```python
{
    "text": str,
    "conf": float,
    "line_idx": int,
    "bbox_abs": [[x, y], [x, y], [x, y], [x, y]],
    "center": (cx, cy),
    "width": float,
    "height": float,
}
```

This structure is the handoff point for inpainting now and Tamil rendering later.

## 3. Full Pipeline Walkthrough

### Stage A: Input Image

The input is a normal image file from `data/images/` or an uploaded image from the local UI.

The code reads it with OpenCV:

```python
img_bgr = cv2.imread(img_path)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

Internally, the project uses RGB arrays for most processing. OpenCV reads/writes BGR, so conversions happen at file boundaries.

### Stage B: CRAFT Text Detection

CRAFT is not reimplemented in this repository. It lives as an external checkout in `CRAFT-pytorch/`.

CRAFT takes an image and writes a result text file:

```text
CRAFT-pytorch/result/res_img1.txt
```

Each line contains four points:

```text
x1,y1,x2,y2,x3,y3,x4,y4
```

Those four points are a quadrilateral around one detected text region. This is better than a rectangle because text in natural scenes can be tilted or perspective-distorted.

The project currently trusts CRAFT for detection because it is working reasonably well. The quality problem is mostly OCR, not detection.

### Stage C: CRAFT Box Parsing and Cleanup

File: `scripts/vtt/detection.py`

This module turns raw CRAFT output into usable text areas.

#### `load_craft_boxes(txt_path)`

Reads the CRAFT `.txt` file line by line.

For each valid line:

1. Parses eight comma-separated numbers.
2. Reshapes them into a `4 x 2` quad.
3. Computes an axis-aligned bounding box.
4. Computes center point, width, and height.
5. Returns a list of dictionaries.

Why this matters:

CRAFT gives geometry as text. The rest of the pipeline needs structured numeric geometry.

#### `bbox_iou(b1, b2)`

Computes two overlap values between two axis-aligned boxes:

- IoU: intersection over union.
- Containment: how much of `b1` is inside `b2`.

Why both exist:

IoU can be small when a small box is inside a large box. Containment catches nested duplicates better.

#### `deduplicate_craft_boxes(boxes, containment_thresh=0.70)`

CRAFT can detect the same text more than once at slightly different scales. This function removes boxes that are mostly contained inside a larger kept box.

How it works:

1. Sort boxes largest first.
2. Keep the first large box.
3. For each later box, compare it to already-kept boxes.
4. If more than 70 percent of it is inside a kept box, suppress it.

Why this matters:

Duplicate CRAFT boxes cause duplicate OCR words, duplicate translations, and overly aggressive masks.

#### `area_median_cy(area)` and `area_median_h(area)`

These compute the median vertical center and median height for a group of boxes.

Why medians:

One noisy box should not drag the line grouping logic away from the real text line.

#### `area_bbox(area)`

Computes the bounding rectangle around every CRAFT box in an area.

This is used for:

- OCR clipping.
- Visualization.
- Inpainting logs.
- Future Tamil rendering placement.

#### `build_text_areas(boxes, v_tol=0.6, h_ratio=0.5, max_line_stride=1.2)`

Groups individual CRAFT boxes into text-line areas.

How it decides whether a new box belongs to an existing area:

1. Compare the box center-y to the area's median center-y.
2. Reject it if the vertical distance is too large.
3. Reject it if the height is too different.
4. Add it to the best matching area.
5. Otherwise start a new area.

Why this matters:

OCR and translation should work on meaningful text lines/areas, not isolated detection fragments.

#### `vertical_overlap_ratio`, `horizontal_overlap_ratio`, and `should_merge`

These functions decide whether two areas overlap enough to be merged.

The code checks both vertical and horizontal overlap so it does not accidentally merge unrelated columns or separate sign regions.

#### `merge_overlapping_areas(areas)`

Repeatedly merges areas that pass the overlap test.

Why repeated:

If A merges with B, the combined area may now overlap C. Iteration catches that.

#### `generate_area_mask(img_shape, area)`

Creates a binary mask for all quads in a text area.

This mask is not the final text-removal mask. It is a broader area mask used during processing/debugging.

#### `is_valid_text_area(area, img_shape)` and `purify_areas(areas, img_shape)`

Filters out tiny detections and likely noise.

Rules include:

- Reject very small width/height.
- Reject areas too small compared with full image size.
- Reject single-box areas that are too tiny.

Why this matters:

Natural images contain small marks, symbols, and detector artifacts. Running OCR on all of them adds junk.

### Stage D: OCR Preprocessing and Recognition

File: `scripts/vtt/ocr.py`

This stage now uses PaddleOCR recognition. The code still preserves the curated crop flow from the original pipeline: each CRAFT quad is clipped, rectified, enhanced, recognized, and mapped back to image space.

#### `PaddleOCRRecognizer`

Thin wrapper around PaddleOCR's recognition-only `TextRecognition` API.

What it does:

1. Sets PaddleX/PaddleOCR cache paths before Paddle imports models.
2. Uses `models/paddleocr/` by default.
3. Loads `te_PP-OCRv5_mobile_rec`.
4. Runs recognition only, not Paddle detection.
5. Returns text/confidence pairs in the shape expected by the existing OCR pipeline.

Why recognition-only matters:

CRAFT is still the detector. PaddleOCR must not choose new crops or boxes in the main pipeline. It only recognizes the rectified CRAFT crop that the project already curated.

#### `create_ocr_reader(use_gpu=True, cache_dir=None, model_name=...)`

Factory used by the CLI and local UI.

It returns a `PaddleOCRRecognizer` and hides the Paddle setup details from the rest of the application.

Default model cache:

```text
models/paddleocr/
```

Default model:

```text
te_PP-OCRv5_mobile_rec
```

#### `parse_paddle_result(result)`

Extracts recognized text and average confidence from PaddleOCR 3.x result objects.

Why it exists:

PaddleOCR result formats can differ slightly across versions. This helper isolates that parsing so the rest of the OCR pipeline receives stable `(text, confidence)` output.

#### Telugu Range Helpers

```python
TELUGU_RANGE = r'[\u0C00-\u0C7F]'
```

Functions:

- `count_telugu_chars(text)`
- `contains_telugu(text)`
- `is_telugu_area(ocr_results, min_telugu_chars=2)`
- `split_telugu_and_other(ocr_results)`

What they do:

They use Unicode ranges to decide whether OCR output contains Telugu characters. This is a script check, not a language understanding check.

Why this matters:

Only Telugu-containing areas should be translated and erased. English and Devanagari text should be protected where possible.

#### `enhance_for_ocr(crop_rgb, clip_limit=2.0)`

Applies CLAHE contrast enhancement on the L channel of LAB color space.

Step by step:

1. Convert RGB to LAB.
2. Split L, A, B channels.
3. Apply CLAHE only on luminance.
4. Merge channels.
5. Convert back to RGB.

Why this matters:

Scene text often has uneven lighting, faded paint, shadows, or low contrast. CLAHE can make the text strokes easier for OCR.

#### `order_quad_points(pts)`

Orders the four CRAFT points as:

```text
top-left, top-right, bottom-right, bottom-left
```

Why this matters:

Perspective transforms require consistent point order. If corners are mixed up, the crop warps incorrectly.

#### `rectify_quad(img, quad, upscale=2.0)`

Flattens a tilted/perspective text quad into a rectangular patch.

Step by step:

1. Order quad corners.
2. Estimate target rectangle width and height.
3. Multiply by upscale factor.
4. Compute perspective transform matrix `M`.
5. Warp the image with `cv2.warpPerspective`.
6. Return the rectified patch, ordered points, and matrix.

Why this matters:

PaddleOCR performs better on horizontal, rectangular text crops than on tilted text inside the full image.

#### `unmap_point(px, py, M_inv)`

Maps a point from the rectified OCR crop back into the original image.

Why this matters:

OCR happens on the warped crop, but masking and rendering must happen in original image coordinates.

#### `ocr_single_quad(img, box_dict, ocr_reader, conf_threshold=0.15, clip_bbox=None)`

Runs OCR on one CRAFT quad.

Step by step:

1. Optionally black out pixels outside the parent area bbox.
2. Rectify the quad with perspective transform.
3. Enhance the rectified crop with CLAHE.
4. Run PaddleOCR recognition on the rectified crop.
5. Drop low-confidence OCR results.
6. Use the full rectified crop as the recognized text bbox.
7. Map that bbox back to original image coordinates.
8. Check that the word center lies inside the source quad.
9. Return word dictionaries.

Why clipping exists:

Without clipping, nearby text from another sign can bleed into the rectified crop. The code calls this the ghost-word problem.

#### `ocr_area(img, area, ocr_reader, conf_threshold=0.15, overlap_thresh=0.50)`

Runs OCR on every CRAFT quad in a grouped area.

Then it deduplicates words inside the area:

1. Sort OCR words by confidence.
2. Keep high-confidence words first.
3. Suppress later words that overlap too much with already-kept words.

Why this matters:

One word can appear multiple times because several CRAFT boxes overlap.

#### `cluster_into_lines(words, v_tol=0.5)`

Groups OCR words into reading lines.

It compares:

- vertical center distance
- median line height
- height compatibility

Why height compatibility matters:

A large headline and small subtitle may be close vertically but should not become one line.

#### `reconstruct_area_sentence(ocr_words)`

Builds a final ordered word list.

Step by step:

1. Cluster words into lines.
2. Sort lines from top to bottom.
3. Sort words in each line from left to right.
4. Store each word with `line_idx`, bbox, center, width, and height.

Why this matters:

Translation and rendering both need reading order.

#### `deduplicate_ocr_across_areas(processed_areas, overlap_thresh=0.50)`

Removes duplicated OCR words that appear in different areas.

How it works:

1. Collect every word from every processed area.
2. Compare absolute OCR bboxes across different areas.
3. If one word is mostly contained in another, keep the higher-confidence one.
4. Rebuild the processed areas with only kept words.

Why this matters:

Area grouping is not perfect. A physical word can leak into two areas.

### Stage E: OCR Normalization and Translation

File: `scripts/vtt/translation.py`

This stage is optional. It is skipped when `--skip-translate` is used.

Important current reality:

Bad OCR produces bad normalization and bad translation. The normalizer can repair some errors, but it cannot reliably recover text that the OCR has heavily corrupted.

#### `_get_client(api_key)`

Creates and caches a Groq client for an API key.

Why caching:

The CLI may call normalization several times. Reusing the client avoids unnecessary setup.

#### `_get_groq_response(client, prompt, json_mode=False)`

Sends a prompt to the Groq model.

What it handles:

- model name
- message format
- temperature
- max tokens
- optional JSON response mode
- retry loop

#### `_clean_json_string(raw_string)`

Removes Markdown code fences if the model returns JSON inside ```json blocks.

Why this exists:

Even with JSON instructions, models sometimes wrap output in Markdown.

#### `detect_image_type(processed_areas, api_key)`

Classifies the image context using OCR text.

Possible categories:

- `signboard`
- `newspaper`
- `road_sign`
- `poster`
- `document`

Why this matters:

The translation prompt can behave differently for a road sign than a poster or document.

#### `normalize_telugu_ocr(raw_text, api_key, retries=3)`

Asks the model to correct Telugu OCR errors without rewriting meaning.

Intended corrections:

- broken spacing
- missing or wrong matras
- split conjuncts
- obvious character-level OCR mistakes

What it should not do:

- invent new content
- paraphrase
- translate
- explain its reasoning

Current limitation:

This is a generative repair step. It is helpful only if the OCR output is close enough to the real Telugu. For very noisy OCR, a better OCR recognizer is the correct fix.

#### `translate_areas(corrected_texts, image_type, api_key, retries=3)`

Translates all corrected Telugu text areas in one JSON-mode call.

Why one call:

The model sees all text areas together, which gives context.

Expected behavior:

- Translate native Telugu into natural Tamil.
- Transliterate English loanwords into Tamil script.
- Transliterate proper nouns into Tamil script.
- Return empty strings for unreadable garbage.

### Stage F: Telugu Quad Classification for Inpainting

File: `scripts/vtt/inpainting.py`

This module decides what pixels to erase.

The hard part is not inpainting itself. The hard part is avoiding removal of non-Telugu text, especially English and Devanagari text near Telugu text.

#### `contains_devanagari(text)`

Checks whether text contains Devanagari Unicode characters.

Why this matters:

Hindi/Devanagari text should be protected from Telugu removal.

#### `is_purely_ascii_noise(text)`

Detects short punctuation or ASCII junk that is probably OCR noise.

Examples treated as noise:

- `-`
- `.`
- `|`
- short corrupted ASCII tokens

Examples protected:

- `DANGER`
- `India`
- `SyndicateBank`

#### `is_protected_non_telugu(text)`

Returns true for text that should not be erased:

- Devanagari
- clean English words

Why this matters:

Mixed-language signs can contain Telugu plus English/Hindi. The goal is Telugu-to-Tamil replacement, not deleting every script.

#### Geometry Helpers

Functions:

- `quad_centre(quad)`
- `word_bbox_centre(bbox_abs)`
- `point_in_quad(px, py, quad)`
- `quad_height(quad)`

What they do:

They compute centers, heights, and point-inside-polygon checks used to match OCR words to CRAFT quads.

#### `is_telugu_quad(quad, area)`

This is the main decision gate for inpainting.

Case 1: OCR word center falls inside the quad.

- If any matched word is protected non-Telugu: skip.
- If matched words are ASCII noise: erase.
- If matched word contains Telugu: erase.

Case 2: No OCR word center falls inside the quad.

- If the whole area looks like pure Telugu: erase the unmatched quad.
- If the area is mixed-script: erase only when the quad is close to a Telugu word center.

Why this is important:

OCR can miss large stylized Telugu text. If the area is pure Telugu, the code erases unmatched CRAFT quads anyway. But in mixed-script areas, that would be dangerous, so it uses proximity to known Telugu words.

### Stage G: Stroke Mask Generation

File: `scripts/vtt/inpainting.py`

#### `build_stroke_mask_for_quad(img_rgb, quad)`

Creates a pixel-level mask of the text ink inside one CRAFT quad.

Step by step:

1. Rectify the quad to a flat patch.
2. Convert to grayscale.
3. Use Otsu thresholding to split ink from background.
4. Estimate threshold confidence.
5. If confidence is weak, use the full quad polygon as fallback.
6. Ensure ink is white in the binary mask.
7. Dilate to catch antialiasing edges.
8. Map the white stroke pixels back to original image coordinates.

Why this is better than filling the whole box:

The background around letters is preserved. Only the text strokes are erased.

Current limitation:

If CRAFT misses the top or bottom of stylized letters, those missed pixels are outside all quads and cannot be masked.

#### `build_stroke_mask_for_area(img_rgb, area)`

Builds one combined mask for all Telugu quads in an area.

Step by step:

1. Start with an empty mask.
2. For every quad in `area_quads`, call `is_telugu_quad`.
3. Skip protected non-Telugu quads.
4. Build a stroke mask for Telugu quads.
5. Combine masks with bitwise OR.

### Stage H: Inpainting

File: `scripts/vtt/inpainting.py`

#### `inpaint_area(img_rgb, area)`

Erases Telugu text in one processed area.

Step by step:

1. Build the stroke mask for the area.
2. Choose an inpaint radius based on text height.
3. Dilate the mask slightly.
4. Convert RGB to BGR for OpenCV.
5. Run `cv2.inpaint` with `cv2.INPAINT_TELEA`.
6. Convert result back to RGB.

Radius selection:

- small text: radius 5
- medium text: radius 8
- large text: radius 12

#### `inpaint_all_areas(img_rgb, processed_areas)`

Runs `inpaint_area` over every processed Telugu area.

Why sequential:

Each area is removed from the current result image, then the next area uses that updated image.

#### `inpaint_noise_boxes(img_rgb, craft_boxes, processed_areas, radius=5)`

Final cleanup for tiny CRAFT boxes that were filtered out earlier but sit inside confirmed Telugu areas.

It only erases small boxes that:

1. Were not already processed.
2. Are very small.
3. Have centers near confirmed Telugu area bboxes.

Why this exists:

Separators, small marks, and punctuation can be left behind after area purification.

## 4. Entry Points

### `scripts/run_pipeline.py`

This is the CLI orchestrator.

Responsibilities:

- Parse arguments.
- Collect image/result pairs.
- Initialize PaddleOCR once.
- Run detection cleanup.
- Run OCR.
- Filter Telugu areas.
- Optionally normalize and translate.
- Run inpainting.
- Save outputs.

Important functions:

#### `parse_args()`

Defines CLI modes and flags.

Modes:

- single image: `--image` + `--result`
- batch/selective: `--image-dir` + `--result-dir`

#### `collect_pairs(args)`

Pairs input images with matching CRAFT result files.

For batch mode, it expects:

```text
image: data/images/img1.jpg
result: CRAFT-pytorch/result/res_img1.txt
```

#### `process_one(img_path, res_path, args, ocr_reader)`

Runs the full pipeline for one image.

It is the best place to read if you want to understand the execution order.

#### `main()`

Creates output folder, initializes PaddleOCR, loops over image pairs, and reports success/failure.

### `local_web_server.py`

This is the local UI backend.

Responsibilities:

- Serve `frontend/index.html`, `styles.css`, and `app.js`.
- Accept image upload at `/api/translate`.
- Ensure CRAFT exists and is patched.
- Run CRAFT for the uploaded image.
- Run the same modular pipeline.
- Return JSON and base64 image data to the browser.

Important functions:

#### `setup_craft()`

Checks whether `CRAFT-pytorch/` exists, downloads weights if needed, and patches compatibility issues.

#### `run_craft(img_path)`

Runs CRAFT on a temporary single-image folder and returns the generated result file.

#### `load_ocr_reader()`

Initializes PaddleOCR once and caches the recognizer.

#### `run_pipeline(img_path, api_key, skip_translate)`

Local-server version of the pipeline orchestration.

It mirrors the CLI flow but returns in-memory data instead of writing files directly.

#### `Handler`

HTTP request handler.

Important routes:

- `GET /` serves the UI.
- `GET /api/health` returns health status.
- `POST /api/translate` runs the pipeline.

### `frontend/index.html`

Defines the UI structure:

- image file picker
- Groq API key input
- skip-translation checkbox
- run and clear buttons
- result summary
- inpainted image preview
- translation result list
- download links

### `frontend/app.js`

Handles browser behavior:

- validates image/API key input
- submits upload using `fetch`
- displays progress state
- renders returned base64 image
- renders translation JSON into expandable sections
- creates download links

### `frontend/styles.css`

Styles the local UI. It does not affect backend behavior.

### `scripts/vtt/visualisation.py`

Debugging and inspection helpers:

- `show_craft_results`
- `visualize_areas`
- `visualize_final_areas`
- `visualise_stroke_masks`
- `visualize_inpainted`

These are useful in notebooks or manual debugging but are not required for headless CLI runs.

### `scripts/vtt/__init__.py`

Exports the functions from each module so callers can import from `vtt` directly:

```python
from vtt import load_craft_boxes, ocr_area, inpaint_all_areas
```

## 5. OCR Engine Status

The main pipeline now uses PaddleOCR's Telugu recognition model, `te_PP-OCRv5_mobile_rec`.

EasyOCR is still available in the benchmark script for comparison, but it is no longer the production recognizer. PaddleOCR is better on the initial sample, but Telugu scene text recognition is still difficult because:

- Telugu characters have many visually similar shapes.
- Matras, vowel signs, and conjunct marks are small and easy to miss.
- Scene text fonts differ from clean printed text.
- Signboards have perspective, blur, shadows, compression, reflections, and stylized fonts.
- OCR on individual CRAFT quads can lose word-level context.
- Some detected regions are partial words or split characters.

The current preprocessing helps, but it cannot fully fix recognizer limitations.

PaddleOCR model cache:

```text
models/paddleocr/
```

This is the default cache used by `scripts/vtt/ocr.py`. It is intentionally separate from `CRAFT-pytorch/` because CRAFT is an external detector checkout, while PaddleOCR is a project dependency/cache. The folder is gitignored. For offline setup, keep a Drive copy of `models/paddleocr/official_models/te_PP-OCRv5_mobile_rec` and restore it into the same path.

## 6. What To Do About OCR Next

Do not start by changing the translation prompt. Translation is downstream. First we need a better OCR signal.

Recommended plan:

### Step 1: Create an OCR evaluation set

For 20-50 representative Telugu text regions:

- Save crop image.
- Write exact ground-truth Telugu text.
- Track image name and bbox/quad.

Then measure:

- Character Error Rate (CER)
- Word Error Rate (WER)
- Telugu-character recall
- common confusion patterns

### Step 2: Test OCR alternatives on the same crops

Candidates to test:

- PaddleOCR as the current local baseline, using `te_PP-OCRv5_mobile_rec` on CRAFT crops.
- Tesseract Telugu as a classical baseline.
- Google Cloud Vision OCR if cloud API use is acceptable.
- Azure OCR if Telugu support is confirmed for the current API.
- TrOCR/Donut-style transformer recognizers fine-tuned on Telugu crops.
- A custom CRNN/SVTR recognizer trained or fine-tuned using Telugu synthetic text plus your real crops.

The repository now includes a benchmark harness:

```text
scripts/benchmark_ocr.py
```

It uses the same CRAFT result files as the production pipeline, rectifies the
same text quads, and sends the same crops to each OCR engine. That means we are
comparing recognition quality, not changing detection or inpainting.

EasyOCR-only benchmark:

```powershell
vision\Scripts\python.exe scripts\benchmark_ocr.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1,img3,img7 `
  --engines easyocr `
```

EasyOCR + PaddleOCR benchmark, after PaddleOCR is installed:

```powershell
vision\Scripts\python.exe scripts\benchmark_ocr.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1,img3,img7 `
  --engines easyocr,paddleocr `
  --paddle-cache models/paddleocr
```

Outputs are written to `output/ocr_benchmark/`:

```text
output/ocr_benchmark/
|-- crops_metadata.csv    # crop id, source image, bbox, quad
|-- ground_truth_template.tsv
|-- ocr_results.csv       # one row per engine per crop
|-- ocr_results.json      # raw OCR output for inspection
|-- ocr_comparison.json   # clean crop-level EasyOCR vs PaddleOCR comparison
`-- summary.json          # aggregate non-empty rate, timing, CER/WER if labeled
```

Crop images are not saved by default. Add `--save-crops` only when you want
rectified PNGs for manual inspection or labeling.

Optional ground truth:

Create a TSV file with this shape:

```text
crop_id	text
img1_q0000	<correct Telugu text here>
img1_q0001	<correct Telugu text here>
```

Then rerun the same benchmark command. The script automatically loads
`ground_truth.tsv`, `ground_truth.csv`, or a filled `ground_truth_template.tsv`
from the output folder. You can still pass `--ground-truth` when you want to use
a truth file stored somewhere else:

```powershell
vision\Scripts\python.exe scripts\benchmark_ocr.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1 `
  --engines easyocr,paddleocr `
  --ground-truth output/ocr_benchmark/ground_truth.tsv
```

When ground truth is present, the script reports CER and WER. Without ground
truth, the comparison is still useful for manual review because it writes each
engine's recognized text side by side.

### Step 3: Separate detection from recognition

Keep CRAFT if detection is good. Swap only `ocr.py`.

The interface to preserve:

```python
ocr_area(img, area, ocr_reader) -> list[dict]
```

As long as a new OCR module returns the same word dictionaries, the rest of the pipeline can stay stable.

### Step 4: Add Telugu-specific normalization before LLM repair

The `docs/Word Normalization pipeline.docx` design points toward a stronger approach:

1. Unicode normalization.
2. Remove illegal intra-word spaces.
3. Reattach split vowel marks and vattulu.
4. Token stabilization.
5. Fuzzy lemma and inflection capture.
6. Proper noun and loanword protection.
7. Only then use a generative model for final polish.

This is better than asking an LLM to repair raw OCR directly.

### Step 5: Revisit translation after OCR improves

Once OCR is better:

- improve the Telugu correction prompt
- compare Groq model choices
- consider IndicTrans2 or another Indic NMT model
- add terminology/proper-noun preservation rules

## 7. Future Rendering Handoff

Tamil rendering should become a new module, not part of inpainting.

Suggested file:

```text
scripts/vtt/rendering.py
```

Inputs:

- inpainted RGB image
- `processed_areas`
- `tamil_translation`
- original `area_bbox`
- original `area_quads`

Responsibilities:

- choose Tamil font
- estimate font size
- wrap text into available region
- choose readable color
- preserve line placement
- optionally match perspective using quads

The rendering module should not change CRAFT, OCR, translation, or inpainting logic.
