# ReBuild Studio

<p align="center">
  <img width="1434" height="625" alt="image" src="https://github.com/user-attachments/assets/9b40146c-9bef-4d87-a19c-1c58b119e710" />
</p>

**Visual scene-text localization and reconstruction for Telugu to Tamil media.**

ReBuild Vision is a proof-of-concept system for translating text that is part of an image itself: signboards, posters, notices, road signs, captions, banners, and other text that appears inside the visual scene.

Most localization systems translate speech or subtitles. They usually ignore text embedded in the video frame. Existing image-translation tools often place translated text over the original text, which can look cluttered and artificial. This project takes a harder route: detect the original text, read it, translate it, remove it from the scene, and prepare the image so translated Tamil text can later be rendered back in a visually consistent way.

The current implementation works at the **image level**. That is intentional. Images are the proof of concept before extending the same idea to videos, where we must also solve tracking, motion, flicker, and temporal consistency.

## Why This Project Exists

India is highly multilingual. A Tamil speaker traveling through Telugu-speaking regions may see important public information in Telugu script: bus signs, institution boards, warning notices, posters, road directions, and local announcements. The text is not a subtitle; it is part of the physical world captured by the camera.

For film, educational content, tourism, navigation, and accessibility, this embedded visual text matters. If it remains untranslated, the viewer loses context. If we simply overlay a translation, the result often looks messy and breaks visual realism.

ReBuild Vision explores a more complete visual localization pipeline:

1. Find text in the scene.
2. Recognize the source text.
3. Translate it into the target language.
4. Remove the original text cleanly.
5. Render translated text back into the same visual region.

The fifth step, Tamil rendering, is the next major phase. The current code has built and tested the first four pieces.

## Why Images First, Then Videos

The full long-term goal is video-level visual text dubbing. But video adds several hard problems on top of the image problem:

- The same text must be tracked across frames.
- Inpainting must not flicker.
- Rendered Tamil text must stay stable as the camera moves.
- Motion blur, perspective, and lighting can change frame by frame.
- A bad decision in one frame can create visible temporal drift.

So the project starts with still images because they let us validate the core idea:

- Can we detect Telugu scene text?
- Can we OCR Telugu from natural images?
- Can we translate the recognized text?
- Can we remove the original text without destroying the background?
- Can we preserve enough geometry for future rendering?

Once the image-level system is reliable, the video system can reuse the same stages frame-by-frame and add tracking plus temporal smoothing.

## Current Scope

Current language pair:

- Source: Telugu
- Target: Tamil

Current input:

- Still images containing Telugu scene text.

Current output:

- Per-image output folder under `output/<image_name>/`.
- Inpainted image with Telugu text removed.
- Metadata JSON containing detection counts, CRAFT geometry, OCR word data, raw OCR, corrected Telugu, and Tamil translation.
- Clean text-results JSON containing raw text, corrected Telugu, and Tamil translation per area.

Current status:

- Text detection is working reasonably well with CRAFT.
- Inpainting is often acceptable because the stroke masks are tight.
- OCR has been switched from EasyOCR to PaddleOCR after an initial benchmark showed better recognition on the same CRAFT crops.
- Translation quality still depends heavily on OCR quality, so OCR evaluation remains important before prompt/model tuning.
- Tamil text rendering has not yet been rebuilt.

Current metrics:

- On hold. The code needs a repeatable evaluation set before reporting meaningful CER/WER, translation, and image-quality numbers.

## Application Pipeline

At a high level:

```text
Input image
  -> CRAFT text detection
  -> CRAFT box cleanup and grouping
  -> OCR preprocessing and PaddleOCR recognition
  -> Telugu-area filtering and OCR deduplication
  -> optional OCR normalization and Telugu-to-Tamil translation
  -> Telugu stroke-mask generation
  -> OpenCV TELEA inpainting
  -> cleaned image + translation JSON
```

What each stage produces:

| Stage | Input | Output |
|---|---|---|
| Detection | Image | Quadrilateral boxes around text |
| Grouping | CRAFT boxes | Larger text areas/lines |
| OCR | Image + text areas | Ordered Telugu/English OCR words |
| Normalization | Raw Telugu OCR | Corrected Telugu text |
| Translation | Corrected Telugu | Tamil translation |
| Masking | Image + OCR + CRAFT quads | Binary mask of Telugu ink strokes |
| Inpainting | Image + stroke mask | Image with Telugu text removed |

Each processed image is saved as:

```text
output/<image_name>/
|-- inpainted.jpg
|-- metadata.json
`-- text_results.json
```

The detailed implementation guide is in [TECHNICAL_README.md](TECHNICAL_README.md).

## Active Repository Layout

```text
.
|-- README.md                    # project overview and motivation
|-- TECHNICAL_README.md           # detailed architecture and code walkthrough
|-- Visual_Translation.ipynb      # original notebook prototype, ignored by git
|-- scripts/
|   |-- run_pipeline.py           # CLI entry point
|   `-- vtt/
|       |-- detection.py          # CRAFT parsing, box cleanup, area grouping
|       |-- ocr.py                # OCR preprocessing, PaddleOCR, line reconstruction
|       |-- translation.py        # Groq-based normalization and translation
|       |-- inpainting.py         # Telugu filtering, stroke masks, TELEA inpainting
|       |-- visualisation.py      # debugging visualizations
|       `-- __init__.py           # package exports
|-- local_web_server.py           # local browser UI backend
|-- frontend/
|   |-- index.html
|   |-- styles.css
|   `-- app.js
|-- data/images/                  # sample input images
|-- output/                       # generated outputs
|-- docs/                         # project proposal and research notes
`-- CRAFT-pytorch/                # external detector checkout, ignored by git
`-- models/paddleocr/             # local PaddleOCR model cache, ignored by git
```

Historical notebooks, training experiments, checkpoints, and deployment notes live in `Others/` and `Helper_files/`. They are useful project memory, but the active maintainable code path is `scripts/`, `frontend/`, `local_web_server.py`, and the two root README files.

## Setup

```text
Check out  [instruction_to_run.md](TECHNICAL_README.md) for direct commands for testing and for running the main pipeline
```

Create a Python environment:

```powershell
python -m venv vision
vision\Scripts\Activate.ps1
pip install -r requirements.txt
```

Clone CRAFT and download its weights:

```powershell
git clone https://github.com/clovaai/CRAFT-pytorch.git
cd CRAFT-pytorch
gdown https://drive.google.com/uc?id=1YsaQIpqePU3hFPt_4EunNsBbwNUyBRDD
cd ..
```

Patch CRAFT for modern torchvision:

```powershell
(Get-Content CRAFT-pytorch\basenet\vgg16_bn.py) `
  -replace '^from torchvision.models.vgg import model_urls', '# from torchvision.models.vgg import model_urls' `
  | Set-Content CRAFT-pytorch\basenet\vgg16_bn.py

$file = Get-Content CRAFT-pytorch\basenet\vgg16_bn.py
$file[24] = '#' + $file[24]
$file | Set-Content CRAFT-pytorch\basenet\vgg16_bn.py

(Get-Content CRAFT-pytorch\craft.py) `
  -replace 'vgg16_bn\(pretrained=True, freeze=True\)', 'vgg16_bn(pretrained=False, freeze=True)' `
  | Set-Content CRAFT-pytorch\craft.py
```

PaddleOCR model cache:

The main pipeline uses PaddleOCR's Telugu recognition model:

```text
te_PP-OCRv5_mobile_rec
```

By default, the code stores/downloads PaddleOCR model files under:

```text
models/paddleocr/
```

This folder is ignored by git. If `models/paddleocr/` is missing, the code creates it automatically. If the Telugu model is missing and network is available, PaddleOCR downloads the model on first run.

You can handle it like CRAFT weights: keep a copy in Drive if you want offline setup, then restore it into `models/paddleocr/` before running.

GPU behavior:

- By default, the CLI, benchmark, and UI try to use GPU when the installed OCR/runtime stack can use it.
- If GPU is not available, they fall back to CPU.
- Use `--no-gpu` only when you deliberately want CPU.
- PaddleOCR can use GPU only when the installed Paddle package is GPU-enabled. A CPU Paddle install will still run correctly, but on CPU.

## CLI Usage

For full copy-paste command sets covering OCR comparison, one-image runs, selected-image runs, all-image runs, GPU/CPU behavior, and the local UI, see [instruction_to_run.md](instruction_to_run.md).

First run CRAFT detection:

```powershell
cd CRAFT-pytorch
python test.py --trained_model=craft_mlt_25k.pth --test_folder=../data/images --cuda=True
cd ..
```

Run the pipeline without translation, useful for smoke testing:

```powershell
python scripts/run_pipeline.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1 `
  --skip-translate
```

Run translation and inpainting:

```powershell
python scripts/run_pipeline.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1,img3,img7 `
  --api-key YOUR_GROQ_API_KEY
```

## Local Browser UI

Detailed UI startup instructions are in [instruction_to_run.md](instruction_to_run.md).

Start the server:

```powershell
python local_web_server.py
```

Open:

```text
http://127.0.0.1:8000
```

The UI accepts an image upload, optionally accepts a Groq API key, runs the same modular pipeline, and returns an inpainted image plus translation JSON. If the key is blank, it runs inpainting-only mode.

## OCR Status

CRAFT usually detects text well, and inpainting can remove detected text well. The earlier EasyOCR recognizer often misread Telugu scene text, so the main pipeline now uses PaddleOCR recognition on the same curated CRAFT crops.

This does not mean OCR is solved. It means PaddleOCR is now the better baseline. The next engineering focus should be measuring OCR with labeled crops before tuning translation prompts.

The benchmark script compares EasyOCR and PaddleOCR on the same CRAFT crops:

```powershell
vision\Scripts\python.exe scripts\benchmark_ocr.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1,img3,img7 `
  --engines easyocr,paddleocr `
  --paddle-cache models/paddleocr `
  --output output/ocr_compare_selected
```

Main comparison output:

```text
output/ocr_compare_selected/ocr_comparison.json
```

## Next Phase

After OCR quality is improved, the next product phase is Tamil rendering:

- Choose a Tamil font.
- Fit translated text into the original area after correcting it.
- Preserve line breaks and perspective where possible.
- Estimate readable color/contrast.
- Render onto the inpainted image.

For video, the same rendered result must later become temporally stable across frames.
