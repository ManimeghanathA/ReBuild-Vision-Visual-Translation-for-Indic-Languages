const els = {
  uploadForm: document.getElementById("uploadForm"),
  imageInput: document.getElementById("imageInput"),
  apiKeyInput: document.getElementById("apiKeyInput"),
  skipTranslateCheckbox: document.getElementById("skipTranslateCheckbox"),
  runButton: document.getElementById("runButton"),
  clearButton: document.getElementById("clearButton"),

  statusText: document.getElementById("statusText"),
  progressBarWrap: document.getElementById("progressBarWrap"),
  progressBar: document.getElementById("progressBar"),
  errorBox: document.getElementById("errorBox"),

  summary: document.getElementById("summary"),
  imageType: document.getElementById("imageType"),
  areaCount: document.getElementById("areaCount"),
  modeText: document.getElementById("modeText"),

  outputGrid: document.getElementById("outputGrid"),
  inpaintedImg: document.getElementById("inpaintedImg"),
  downloadInpaintedLink: document.getElementById("downloadInpaintedLink"),

  translationList: document.getElementById("translationList"),
  translationMeta: document.getElementById("translationMeta"),
  downloadTranslationsLink: document.getElementById("downloadTranslationsLink"),
};

function setBusy(busy) {
  els.runButton.disabled = busy;
  els.clearButton.disabled = busy;
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function showProgress(show) {
  els.progressBarWrap.hidden = !show;
}

function setProgress(percent) {
  const p = Math.max(0, Math.min(100, percent));
  els.progressBar.style.width = `${p}%`;
}

function showError(msg) {
  els.errorBox.hidden = false;
  els.errorBox.textContent = msg;
}

function clearError() {
  els.errorBox.hidden = true;
  els.errorBox.textContent = "";
}

function clearResults() {
  els.outputGrid.hidden = true;
  els.translationList.innerHTML = "";
  els.translationMeta.innerHTML = "";
  els.downloadInpaintedLink.hidden = true;
  els.downloadTranslationsLink.hidden = true;
  els.inpaintedImg.removeAttribute("src");
  els.summary.hidden = true;
}

function makeInpaintedDownloadLink(base64, filename) {
  els.downloadInpaintedLink.hidden = false;
  els.downloadInpaintedLink.href = `data:image/jpeg;base64,${base64}`;
  els.downloadInpaintedLink.download = filename;
}

function makeTranslationsDownloadLink(translationData, filename) {
  const jsonStr = JSON.stringify(translationData, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  els.downloadTranslationsLink.hidden = false;
  els.downloadTranslationsLink.href = url;
  els.downloadTranslationsLink.download = filename;
}

function renderTranslationList(translationData) {
  if (!translationData || translationData.length === 0) {
    els.translationList.innerHTML =
      `<div class="translationItem" style="color: #9ca3af;">No Telugu areas detected in the image.</div>`;
    return;
  }

  els.translationList.innerHTML = "";
  for (const row of translationData) {
    const areaNum = row.area ?? "?";
    const rawOcr = row.raw_ocr ?? "";
    const corrected = row.corrected_telugu ?? "";
    const tamil = row.tamil_translation ?? "";

    const item = document.createElement("div");
    item.className = "translationItem";

    const summaryText = `Area ${areaNum} — ${rawOcr.length > 60 ? rawOcr.slice(0, 60) + "…" : rawOcr}`;
    const details = document.createElement("details");

    const sum = document.createElement("summary");
    sum.textContent = summaryText;
    details.appendChild(sum);

    const raw = document.createElement("div");
    raw.className = "rowTitle";
    raw.textContent = "Raw OCR (Telugu)";
    details.appendChild(raw);
    const rawPre = document.createElement("pre");
    rawPre.textContent = rawOcr || "(empty)";
    details.appendChild(rawPre);

    const corr = document.createElement("div");
    corr.className = "rowTitle";
    corr.textContent = "Corrected Telugu";
    details.appendChild(corr);
    const corrPre = document.createElement("pre");
    corrPre.textContent = corrected || "(empty)";
    details.appendChild(corrPre);

    const tamilDiv = document.createElement("div");
    tamilDiv.className = "rowTitle";
    tamilDiv.textContent = "Tamil Translation";
    details.appendChild(tamilDiv);
    const tamilPre = document.createElement("pre");
    tamilPre.textContent = tamil || "(empty)";
    details.appendChild(tamilPre);

    item.appendChild(details);
    els.translationList.appendChild(item);
  }
}

async function runPipeline() {
  clearError();
  clearResults();

  const file = els.imageInput.files && els.imageInput.files[0];
  if (!file) {
    showError("Please choose an image file.");
    return;
  }

  const apiKey = (els.apiKeyInput.value || "").trim();
  const skipTranslate = !!els.skipTranslateCheckbox.checked;

  // If user chose inpainting-only, skip_translate=true can work without api key.
  if (!skipTranslate && !apiKey) {
    showError("Please enter your Sarvam AI API key (or enable 'Skip translation').");
    return;
  }

  setBusy(true);
  setStatus("Uploading & running pipeline… (this can take 30–90 seconds)");
  showProgress(true);
  setProgress(10);

  try {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("api_key", apiKey);
    formData.append("skip_translate", skipTranslate ? "1" : "0");

    const res = await fetch("/api/translate", {
      method: "POST",
      body: formData,
    });

    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = payload && payload.error ? payload.error : "Request failed.";
      throw new Error(msg);
    }

    const base64 = payload.inpainted_image_jpeg_base64;
    if (!base64) throw new Error("Server did not return inpainted image.");

    // Show summary
    els.summary.hidden = false;
    els.imageType.textContent = payload.image_type || "unknown";
    els.areaCount.textContent = String(payload.n_areas ?? 0);
    els.modeText.textContent = payload.skip_translate ? "inpainting-only" : "translation + inpainting";

    setProgress(70);

    // Show inpainted image
    els.outputGrid.hidden = false;
    els.inpaintedImg.src = `data:image/jpeg;base64,${base64}`;

    const stem = file.name ? file.name.replace(/\.[^/.]+$/, "") : "image";
    makeInpaintedDownloadLink(base64, `${stem}_inpainted.jpg`);

    // Render translation table
    const td = payload.translation_data || [];
    if (!payload.skip_translate && td.length > 0) {
      els.translationMeta.textContent = `Found ${td.length} Telugu area(s). Click an area to expand details.`;
    } else if (payload.skip_translate) {
      els.translationMeta.textContent = `Inpainting-only mode. Translation fields may be empty.`;
    } else {
      els.translationMeta.textContent = `No Telugu areas detected.`;
    }

    renderTranslationList(td);

    // Download translations
    makeTranslationsDownloadLink(td, `${stem}_translations.json`);

    setStatus(`Done in ${payload.elapsed_sec ?? "?"}s.`);
    setProgress(100);
  } catch (e) {
    showError(String(e && e.message ? e.message : e));
    setStatus("Failed.");
    setProgress(0);
  } finally {
    setBusy(false);
    showProgress(false);
  }
}

function init() {
  els.uploadForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    runPipeline();
  });

  els.clearButton.addEventListener("click", () => {
    els.imageInput.value = "";
    els.apiKeyInput.value = "";
    els.skipTranslateCheckbox.checked = true;
    clearError();
    clearResults();
    setStatus("Idle.");
  });

  setStatus("Idle.");
  showProgress(false);
  clearResults();
}

init();

