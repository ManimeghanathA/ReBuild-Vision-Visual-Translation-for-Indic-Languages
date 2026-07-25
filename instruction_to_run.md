# ReBuild Vision - Run Instructions

Do not paste real API keys into this file.
Use `--api-key YOUR_GROQ_API_KEY` on the CLI, or paste the key into the local UI.

All commands below assume you are in the project root:

```powershell
cd "C:\Users\manim\OneDrive\Desktop\ReBuild Stuido"
vision\Scripts\Activate.ps1
```

## 1. One-Time Setup

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

CRAFT setup, if `CRAFT-pytorch/` does not already exist:

```powershell
git clone https://github.com/clovaai/CRAFT-pytorch.git
cd CRAFT-pytorch
gdown https://drive.google.com/uc?id=1YsaQIpqePU3hFPt_4EunNsBbwNUyBRDD
cd ..
```

Patch CRAFT once for modern torchvision:

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

```text
models/paddleocr/official_models/te_PP-OCRv5_mobile_rec/
```

If `models/paddleocr/` is missing, the code creates it automatically. If the Telugu model is missing and network is available, PaddleOCR downloads the model on first run and stores it there.

You can treat this like the CRAFT weight folder for offline setup:

```text
models/paddleocr/
```

Back it up to Drive if you want, then restore it into the same path before running on another machine.

GPU behavior:

- By default, the benchmark, CLI, and UI try to use GPU when Paddle/PyTorch can see one.
- If GPU is not available, they fall back to CPU.
- Add `--no-gpu` only when you want to force CPU.
- PaddleOCR can use GPU only when the installed Paddle package is GPU-enabled. If you installed the CPU Paddle wheel, the code still runs, but it runs PaddleOCR on CPU.

Quick GPU check:

```powershell
vision\Scripts\python.exe -c "import paddle; print('Paddle CUDA:', paddle.is_compiled_with_cuda())"
```

## 2. Run CRAFT Detection

The comparison script and the main CLI both use CRAFT result files. Run CRAFT first whenever `CRAFT-pytorch/result/res_<image>.txt` files are missing or stale.

Run CRAFT on all images:

```powershell
cd CRAFT-pytorch
python test.py --trained_model=craft_mlt_25k.pth --test_folder=../data/images --cuda=True
cd ..
```

CPU fallback:

```powershell
cd CRAFT-pytorch
python test.py --trained_model=craft_mlt_25k.pth --test_folder=../data/images --cuda=False
cd ..
```

Important: CRAFT creates the boxes/crops. EasyOCR and PaddleOCR only recognize text inside those CRAFT crops in our comparison.

## 3. OCR Comparison: EasyOCR vs PaddleOCR

The comparison uses the same CRAFT-derived, rectified crop sections as the main pipeline. It does not OCR the whole image.

Primary output to inspect:

```text
output/<your_folder>/ocr_comparison.json
```

Other useful files:

```text
output/<your_folder>/summary.json
output/<your_folder>/ocr_results.csv
output/<your_folder>/ground_truth_template.tsv
```

By default, the commands below do not save cropped images. Add `--save-crops` only if you want crop PNGs for manual labeling.

Ground-truth automation:

- Each benchmark run writes `output/<your_folder>/ground_truth_template.tsv`.
- Fill the `text` column in that same file.
- Run the same benchmark command again.
- The script automatically detects the filled `ground_truth_template.tsv` in the output folder and adds CER/WER to `ocr_comparison.json`, `ocr_results.csv`, and `summary.json`.
- You can also create `output/<your_folder>/ground_truth.tsv`; it will be auto-detected before the template file.

### 3.1 Compare OCR on one image

Example with `img1`:

```powershell
vision\Scripts\python.exe scripts\benchmark_ocr.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1 `
  --engines easyocr,paddleocr `
  --paddle-cache models/paddleocr `
  --output output/ocr_compare_img1
```

Inspect:

```powershell
Get-Content output\ocr_compare_img1\ocr_comparison.json
Get-Content output\ocr_compare_img1\summary.json
```

### 3.2 Compare OCR on selected multiple images

Example with `img1,img3,img7`:

```powershell
vision\Scripts\python.exe scripts\benchmark_ocr.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1,img3,img7 `
  --engines easyocr,paddleocr `
  --paddle-cache models/paddleocr `
  --output output/ocr_compare_selected
```

### 3.3 Compare OCR on all images in `data/images/`

```powershell
vision\Scripts\python.exe scripts\benchmark_ocr.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --engines easyocr,paddleocr `
  --paddle-cache models/paddleocr `
  --output output/ocr_compare_all
```

### 3.4 Add ground truth and get CER/WER automatically

First run a comparison. Then fill:

```text
output/<your_folder>/ground_truth_template.tsv
```

Keep the columns:

```text
crop_id	text
```

Then rerun the same command. Example:

```powershell
vision\Scripts\python.exe scripts\benchmark_ocr.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1,img3,img7 `
  --engines easyocr,paddleocr `
  --paddle-cache models/paddleocr `
  --output output/ocr_compare_selected
```

Because the output folder already contains a filled `ground_truth_template.tsv`, CER/WER are added automatically.

Force CPU only if needed:

```powershell
vision\Scripts\python.exe scripts\benchmark_ocr.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1 `
  --engines easyocr,paddleocr `
  --paddle-cache models/paddleocr `
  --no-gpu `
  --output output/ocr_compare_img1_cpu
```

## 4. Main Pipeline CLI

The main pipeline now uses PaddleOCR. EasyOCR remains only in `scripts/benchmark_ocr.py`.

Use `--skip-translate` to test detection, PaddleOCR, Telugu filtering, and inpainting without Groq API calls.

Every processed image is saved under `output/<image_name>/`:

```text
output/<image_name>/
|-- inpainted.jpg
|-- metadata.json
`-- text_results.json
```

`metadata.json` is always written, even with `--skip-translate`. It contains the source paths, detection/grouping counts, CRAFT quads, OCR word metadata, raw OCR text, corrected Telugu when available, and Tamil translation when available.

`text_results.json` is the cleaner inspection file. It contains only the area bbox, raw OCR text, corrected Telugu, and Tamil translation.

### 4.1 Run one image

Example with `img7`:

```powershell
vision\Scripts\python.exe scripts\run_pipeline.py `
  --image data/images/img7.jpeg `
  --result CRAFT-pytorch/result/res_img7.txt `
  --skip-translate `
  --output output
```

With translation:

```powershell
vision\Scripts\python.exe scripts\run_pipeline.py `
  --image data/images/img7.jpeg `
  --result CRAFT-pytorch/result/res_img7.txt `
  --api-key YOUR_GROQ_API_KEY `
  --output output
```

### 4.2 Run selected multiple images

```powershell
vision\Scripts\python.exe scripts\run_pipeline.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1,img3,img7 `
  --skip-translate `
  --output output
```

With translation:

```powershell
vision\Scripts\python.exe scripts\run_pipeline.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --select img1,img3,img7 `
  --api-key YOUR_GROQ_API_KEY `
  --output output
```

### 4.3 Run all images in `data/images/`

```powershell
vision\Scripts\python.exe scripts\run_pipeline.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --skip-translate `
  --output output
```

After this finishes, inspect:

```text
output/img1/inpainted.jpg
output/img1/metadata.json
output/img1/text_results.json
output/img2/inpainted.jpg
output/img2/metadata.json
output/img2/text_results.json
...
```

With translation:

```powershell
vision\Scripts\python.exe scripts\run_pipeline.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --api-key YOUR_GROQ_API_KEY `
  --output output
```

Force CPU only if needed:

```powershell
vision\Scripts\python.exe scripts\run_pipeline.py `
  --image-dir data/images `
  --result-dir CRAFT-pytorch/result `
  --skip-translate `
  --no-gpu `
  --output output
```

## 5. Local UI / UX

Start the local web server:

```powershell
vision\Scripts\python.exe local_web_server.py
```

The UI uses the same automatic device behavior: GPU if available, CPU fallback otherwise.

Open this in your browser:

```text
http://127.0.0.1:8000
```

How to use the UI:

1. Choose an image file.
2. Leave the Groq key empty and keep "Skip translation" checked for inpainting-only testing.
3. Paste `YOUR_GROQ_API_KEY` and uncheck "Skip translation" when you want translation.
4. Click "Run Translation Pipeline".
5. Download the inpainted image and JSON results.

If port `8000` is already in use, stop the existing server or change the port inside `local_web_server.py`.
