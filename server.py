"""
server.py
---------
Web server for Visual Translation for Indic Languages (ReBuild Studio).

When an image is uploaded:
1. Saves it into `data/images/{image_name}.jpg`
2. Runs CRAFT text detection to generate `CRAFT-pytorch/result/res_{image_name}.txt`
3. Executes `scripts/run_pipeline.py`:
     --image data/images/{image_name}.jpg
     --result CRAFT-pytorch/result/res_{image_name}.txt
     --api-key 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
     --output output
4. Displays the rendered image stored in `render_output/{image_name}/rendered.jpg`
"""

import os
import sys
import re
import json
import shutil
import subprocess
import tempfile
import threading
import mimetypes
import traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
CRAFT_DIR = os.path.join(BASE_DIR, "CRAFT-pytorch")
WEIGHTS = os.path.join(CRAFT_DIR, "craft_mlt_25k.pth")
RESULT_DIR = os.path.join(CRAFT_DIR, "result")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RENDER_OUTPUT_DIR = os.path.join(BASE_DIR, "render_output")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

DEFAULT_API_KEY = "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"

os.makedirs(DATA_IMAGES_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RENDER_OUTPUT_DIR, exist_ok=True)

_craft_ready = False
_craft_ready_lock = threading.Lock()


def has_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def setup_craft() -> None:
    """One-time CRAFT setup (clone weights + patch torchvision compatibility if needed)."""
    global _craft_ready
    with _craft_ready_lock:
        if _craft_ready:
            return

        if not os.path.exists(CRAFT_DIR):
            print("[setup] Cloning CRAFT-pytorch...")
            subprocess.run(
                ["git", "clone", "https://github.com/clovaai/CRAFT-pytorch.git", CRAFT_DIR],
                check=True,
                capture_output=True,
                text=True,
            )

        os.makedirs(RESULT_DIR, exist_ok=True)
        if not os.path.exists(WEIGHTS):
            print("[setup] Downloading CRAFT weights...")
            try:
                import gdown
                gdown.download(
                    "https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ",
                    WEIGHTS,
                    quiet=False,
                )
            except Exception as e:
                print(f"[warning] CRAFT weights download error: {e}")

        # Patch torchvision compatibility if needed
        vgg_path = os.path.join(CRAFT_DIR, "basenet", "vgg16_bn.py")
        craft_py = os.path.join(CRAFT_DIR, "craft.py")

        if os.path.exists(vgg_path):
            try:
                with open(vgg_path, "r", encoding="utf-8", errors="ignore") as f:
                    vgg_src = f.read()
                vgg_src2 = vgg_src.replace(
                    "from torchvision.models.vgg import model_urls",
                    "# from torchvision.models.vgg import model_urls"
                )
                if vgg_src2 != vgg_src:
                    with open(vgg_path, "w", encoding="utf-8") as f:
                        f.write(vgg_src2)
            except Exception as e:
                print(f"[warning] Patching vgg16_bn.py failed: {e}")

        if os.path.exists(craft_py):
            try:
                with open(craft_py, "r", encoding="utf-8", errors="ignore") as f:
                    craft_src = f.read()
                craft_src2 = craft_src.replace(
                    "vgg16_bn(pretrained=True, freeze=True)",
                    "vgg16_bn(pretrained=False, freeze=True)"
                )
                if craft_src2 != craft_src:
                    with open(craft_py, "w", encoding="utf-8") as f:
                        f.write(craft_src2)
            except Exception as e:
                print(f"[warning] Patching craft.py failed: {e}")

        _craft_ready = True


def run_craft(image_name: str, image_path: str) -> str:
    """Runs CRAFT detection on image_path and produces res_{image_name}.txt."""
    setup_craft()
    res_path = os.path.join(RESULT_DIR, f"res_{image_name}.txt")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_img = os.path.join(tmp_dir, f"{image_name}.jpg")
        shutil.copy2(image_path, tmp_img)

        cmd = [
            sys.executable,
            "test.py",
            f"--trained_model={WEIGHTS}",
            f"--test_folder={tmp_dir}",
            f"--cuda={'True' if has_gpu() else 'False'}",
        ]
        print(f"[CRAFT] Running detection: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=CRAFT_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[CRAFT error]\n{result.stderr}")
            raise RuntimeError(f"CRAFT detection failed: {result.stderr[-1000:]}")

    if not os.path.exists(res_path):
        raise FileNotFoundError(f"CRAFT result file expected at {res_path} was not created.")

    print(f"[CRAFT] Result generated at {res_path}")
    return res_path


def run_pipeline(image_name: str, api_key: str = DEFAULT_API_KEY) -> str:
    """
    Executes scripts/run_pipeline.py with the exact arguments:
      --image data/images/{image_name}.jpg
      --result CRAFT-pytorch/result/res_{image_name}.txt
      --api-key {api_key}
      --output output
    """
    image_rel_path = os.path.join("data", "images", f"{image_name}.jpg")
    image_full_path = os.path.join(BASE_DIR, image_rel_path)
    res_rel_path = os.path.join("CRAFT-pytorch", "result", f"res_{image_name}.txt")
    res_full_path = os.path.join(BASE_DIR, res_rel_path)

    if not os.path.exists(image_full_path):
        raise FileNotFoundError(f"Image not found at {image_full_path}")

    # Ensure CRAFT text file exists
    if not os.path.exists(res_full_path):
        print(f"[Pipeline] CRAFT result {res_full_path} missing. Running CRAFT first...")
        run_craft(image_name, image_full_path)

    run_pipeline_script = os.path.join(SCRIPTS_DIR, "run_pipeline.py")

    cmd = [
        sys.executable,
        run_pipeline_script,
        "--image", image_rel_path,
        "--result", res_rel_path,
        "--api-key", api_key,
        "--output", "output"
    ]

    print(f"[Pipeline] Executing: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )

    print(f"[Pipeline stdout]\n{result.stdout}")
    if result.returncode != 0:
        print(f"[Pipeline stderr]\n{result.stderr}")
        raise RuntimeError(f"run_pipeline.py failed:\n{result.stderr[-1500:]}")

    expected_render_path = os.path.join(RENDER_OUTPUT_DIR, image_name, "rendered.jpg")
    if not os.path.exists(expected_render_path):
        raise FileNotFoundError(f"Rendered output not found at: {expected_render_path}")

    print(f"[Pipeline] Successfully rendered image at: {expected_render_path}")
    return expected_render_path


def parse_multipart_data(content_type: str, body: bytes):
    """Robust parser for multipart/form-data image upload."""
    boundary_str = None
    for token in content_type.split(";"):
        token = token.strip()
        if token.lower().startswith("boundary="):
            boundary_str = token.split("=", 1)[1].strip()
            if boundary_str.startswith('"') and boundary_str.endswith('"'):
                boundary_str = boundary_str[1:-1]
            break

    if not boundary_str:
        raise ValueError(f"Could not extract boundary from Content-Type: {content_type}")

    boundary_bytes = ("--" + boundary_str).encode("latin-1")
    parts = body.split(boundary_bytes)

    file_bytes = None
    filename = "uploaded_image.jpg"
    api_key = DEFAULT_API_KEY

    for part in parts:
        if not part or part in (b"--\r\n", b"--\n", b"--", b"\r\n", b"\n"):
            continue

        if b"\r\n\r\n" in part:
            header_part, content = part.split(b"\r\n\r\n", 1)
            if content.endswith(b"\r\n"):
                content = content[:-2]
            elif content.endswith(b"\n"):
                content = content[:-1]
        elif b"\n\n" in part:
            header_part, content = part.split(b"\n\n", 1)
            if content.endswith(b"\n"):
                content = content[:-1]
        else:
            continue

        header_text = header_part.decode("latin-1", errors="ignore")

        if 'name="api_key"' in header_text:
            val = content.decode("utf-8", errors="ignore").strip()
            if val:
                api_key = val
        elif 'filename="' in header_text or 'name="image"' in header_text:
            match = re.search(r'filename="([^"]+)"', header_text, re.IGNORECASE)
            if match:
                filename = match.group(1)
            file_bytes = content

    return filename, file_bytes, api_key


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ReBuild Studio - Visual Translation for Indic Languages</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0b0f19;
            --card-bg: rgba(22, 29, 47, 0.7);
            --card-border: rgba(255, 255, 255, 0.1);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --accent: #06b6d4;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --success: #10b981;
            --error: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(6, 182, 212, 0.15) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 2rem 1.5rem;
            text-align: center;
            border-bottom: 1px solid var(--card-border);
            backdrop-filter: blur(10px);
            background: rgba(11, 15, 25, 0.6);
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #818cf8 0%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 1rem;
        }

        main {
            flex: 1;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .upload-area {
            border: 2px dashed var(--card-border);
            border-radius: 12px;
            padding: 3rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.02);
            position: relative;
        }

        .upload-area:hover, .upload-area.dragover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.05);
        }

        .upload-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
        }

        .file-input {
            display: none;
        }

        .form-group {
            margin-top: 1.5rem;
        }

        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: var(--text-main);
        }

        input[type="text"] {
            width: 100%;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: 1px solid var(--card-border);
            background: rgba(15, 23, 42, 0.6);
            color: var(--text-main);
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }

        input[type="text"]:focus {
            border-color: var(--primary);
        }

        .btn-submit {
            margin-top: 1.5rem;
            width: 100%;
            padding: 1rem;
            border: none;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
            color: #fff;
            font-family: inherit;
            font-weight: 600;
            font-size: 1.05rem;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .status-container {
            display: none;
            margin-top: 1.5rem;
            padding: 1.25rem;
            border-radius: 10px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--card-border);
        }

        .status-spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
            margin-right: 8px;
            vertical-align: middle;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .results-section {
            display: none;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .results-section {
                grid-template-columns: 1fr;
            }
        }

        .image-box {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid var(--card-border);
            text-align: center;
        }

        .image-box h3 {
            margin-bottom: 0.75rem;
            font-size: 1.1rem;
            color: #818cf8;
        }

        .image-box img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
    </style>
</head>
<body>

    <header>
        <h1>ReBuild Studio Visual Translation</h1>
        <p>Telugu Scene Text Detection & Inpainting & Tamil Visual Translation Pipeline</p>
    </header>

    <main>
        <section class="card">
            <form id="pipeline-form">
                <div class="upload-area" id="drop-zone">
                    <span class="upload-icon">📸</span>
                    <h3>Select or Drag & Drop Image</h3>
                    <p id="file-name-display" style="color: var(--text-muted); margin-top: 0.5rem;">JPG, JPEG, PNG supported</p>
                </div>
                <input type="file" id="image-input" name="image" accept="image/*" class="file-input" required>

                <div class="form-group">
                    <label for="api-key">Groq API Key</label>
                    <input type="text" id="api-key" name="api_key" value="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">
                </div>

                <button type="submit" id="submit-btn" class="btn-submit">Run Pipeline & Translate</button>
            </form>

            <div id="status-box" class="status-container">
                <span id="status-spinner" class="status-spinner"></span>
                <span id="status-msg">Processing pipeline... (CRAFT -> OCR -> Groq -> Inpaint -> Render)</span>
            </div>
        </section>

        <section id="results-card" class="card" style="display:none;">
            <h2 style="margin-bottom: 1.5rem; color: #38bdf8;">Pipeline Results</h2>
            <div class="results-section" style="display: grid;">
                <div class="image-box">
                    <h3>Original Input Image</h3>
                    <img id="original-img" src="" alt="Original Image">
                </div>
                <div class="image-box">
                    <h3>Rendered Output (render_output/{image_name}/rendered.jpg)</h3>
                    <img id="rendered-img" src="" alt="Rendered Image">
                </div>
            </div>
        </section>
    </main>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('image-input');
        const fileNameDisplay = document.getElementById('file-name-display');
        const pipelineForm = document.getElementById('pipeline-form');
        const submitBtn = document.getElementById('submit-btn');
        const statusBox = document.getElementById('status-box');
        const statusSpinner = document.getElementById('status-spinner');
        const statusMsg = document.getElementById('status-msg');
        const resultsCard = document.getElementById('results-card');
        const originalImg = document.getElementById('original-img');
        const renderedImg = document.getElementById('rendered-img');

        dropZone.addEventListener('click', (e) => {
            fileInput.click();
        });

        fileInput.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updateFileName();
            }
        });

        fileInput.addEventListener('change', updateFileName);

        function updateFileName() {
            if (fileInput.files.length > 0) {
                fileNameDisplay.textContent = fileInput.files[0].name;
                fileNameDisplay.style.color = '#38bdf8';
            }
        }

        pipelineForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInput.files.length) {
                alert('Please select an image file first.');
                return;
            }

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            formData.append('api_key', document.getElementById('api-key').value);

            submitBtn.disabled = true;
            statusBox.style.display = 'block';
            statusSpinner.style.display = 'inline-block';
            statusMsg.style.color = '#f3f4f6';
            statusMsg.textContent = 'Uploading image and running pipeline... Please wait (~10-30s)';
            resultsCard.style.display = 'none';

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    statusSpinner.style.display = 'none';
                    statusMsg.style.color = '#10b981';
                    statusMsg.textContent = 'Pipeline execution completed successfully!';
                    
                    originalImg.src = data.original_url + '?t=' + Date.now();
                    renderedImg.src = data.rendered_url + '?t=' + Date.now();
                    resultsCard.style.display = 'grid';
                } else {
                    throw new Error(data.error || 'Pipeline execution failed.');
                }
            } catch (err) {
                statusSpinner.style.display = 'none';
                statusMsg.style.color = '#ef4444';
                statusMsg.textContent = 'Error: ' + err.message;
            } finally {
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


class WebServerHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

    def serve_file(self, file_path: str, default_content_type: str = "application/octet-stream"):
        if not os.path.exists(file_path):
            self.send_error(404, f"File not found: {file_path}")
            return

        content_type, _ = mimetypes.guess_type(file_path)
        if not content_type:
            content_type = default_content_type

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_UI.encode("utf-8"))
            return

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if path.startswith("/render_output/"):
            rel_path = path[len("/render_output/"):]
            full_path = os.path.join(RENDER_OUTPUT_DIR, rel_path)
            self.serve_file(full_path)
            return

        if path.startswith("/data/images/"):
            rel_path = path[len("/data/images/"):]
            full_path = os.path.join(DATA_IMAGES_DIR, rel_path)
            self.serve_file(full_path)
            return

        if path.startswith("/output/"):
            rel_path = path[len("/output/"):]
            full_path = os.path.join(OUTPUT_DIR, rel_path)
            self.serve_file(full_path)
            return

        self.send_error(404, "Page Not Found")

    def do_POST(self):
        path = self.path.split("?")[0]

        if path in ("/upload", "/api/upload"):
            try:
                content_type = self.headers.get("Content-Type", "")
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)

                if not content_type.startswith("multipart/form-data"):
                    raise ValueError(f"Content-Type must be multipart/form-data, got: {content_type}")

                filename, file_bytes, api_key = parse_multipart_data(content_type, body)

                if not file_bytes:
                    raise ValueError("No image file provided in upload request.")

                # Extract image name stem
                stem = Path(filename).stem
                image_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', stem)
                if not image_name:
                    image_name = "uploaded_image"

                dest_image_path = os.path.join(DATA_IMAGES_DIR, f"{image_name}.jpg")
                print(f"[Upload] Saving image ({len(file_bytes)} bytes) to: {dest_image_path}")

                nparr = np.frombuffer(file_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    cv2.imwrite(dest_image_path, img_bgr)
                else:
                    with open(dest_image_path, "wb") as f:
                        f.write(file_bytes)

                # Execute CRAFT + run_pipeline.py
                rendered_path = run_pipeline(image_name, api_key=api_key)

                original_url = f"/data/images/{image_name}.jpg"
                rendered_url = f"/render_output/{image_name}/rendered.jpg"

                response_data = {
                    "success": True,
                    "image_name": image_name,
                    "original_url": original_url,
                    "rendered_url": rendered_url,
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            except Exception as e:
                print(f"[Upload Error] {e}")
                traceback.print_exc()
                err_response = {
                    "success": False,
                    "error": str(e)
                }
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(err_response).encode("utf-8"))
            return

        self.send_error(404, "Endpoint Not Found")


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, WebServerHandler)
    print(f"=" * 60)
    print(f"Server started on http://localhost:{port}")
    print(f"Feed images to the web server to throw into data/images/{{image_name}}.jpg")
    print(f"and execute scripts/run_pipeline.py")
    print(f"=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server.")
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
