"""
vtt/inpainting.py
─────────────────
Phase 2 — Telugu text removal via tight stroke masks.

Synced with Visual_Translation_v2.ipynb (cell 56 + cell 60).

Key decisions:
  - Stroke mask: per-quad Otsu thresholding isolates ink pixels,
    unmapped via M_inv back to image space.
  - is_telugu_quad v2: hard-protects Devanagari AND clean English words
    via is_protected_non_telugu(). Pure-Telugu areas erase all unmatched
    quads unconditionally (Case A). Mixed areas use quad-height-scaled
    proximity threshold (Case B).
  - Noise cleanup pass: erases tiny CRAFT boxes (e.g. '-' separator).

Known limitation (documented, accepted):
  - CRAFT underestimates quad height for large 3D/stylised text (e.g.
    the blue "భారత్" in img4). Upper/lower glyph strokes fall outside
    all detected quads and are therefore not in any mask. Vertical quad
    expansion was evaluated and did not reliably improve results.
"""

import cv2
import numpy as np

from .ocr import (
    contains_telugu,
    rectify_quad,
    order_quad_points,
)


# ── Script detection helpers ──────────────────────────────────────────────────

def contains_devanagari(text: str) -> bool:
    """True if text contains any Devanagari (Hindi, Marathi, Sanskrit)."""
    return any(0x0900 <= ord(c) <= 0x097F for c in text)


def is_purely_ascii_noise(text: str) -> bool:
    """
    True for single punctuation chars and short ASCII tokens that are
    clearly OCR noise, NOT real English words.

    True:  '-', '.', '|', 'SOll', '3UhH', 'G0v'
    False: 'DANGER', 'India', 'SyndicateBank'
    """
    if not text:
        return False
    if len(text.strip()) == 1 and not text.strip().isalpha():
        return True
    telugu_punct = set('॥।०१२३४५६७८९')
    has_tp = any(c in telugu_punct for c in text)
    all_ap = all(ord(c) < 128 or c in telugu_punct for c in text)
    if has_tp and all_ap and len(text.strip()) <= 6:
        return True
    if all(ord(c) < 128 for c in text):
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if len(text.strip()) <= 4:
            return True
        if alpha_ratio < 0.70:
            return True
    return False


def is_protected_non_telugu(text: str) -> bool:
    """
    True for text that must NEVER be erased:
      • Devanagari: सिंडिकेटबैंक, भारत सरकार का उपक्रम, etc.
      • Clean real English words (>4 chars, ≥70% alpha)
        e.g. SyndicateBank, India, Undertaking, DANGER, Picxy
    """
    if contains_devanagari(text):
        return True
    if all(ord(c) < 128 for c in text) and len(text.strip()) > 4:
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio >= 0.70:
            return True
    return False


# ── Quad geometry helpers ─────────────────────────────────────────────────────

def quad_centre(quad: np.ndarray) -> tuple[float, float]:
    q = np.array(quad)
    return float(q[:, 0].mean()), float(q[:, 1].mean())


def word_bbox_centre(bbox_abs: list) -> tuple[float, float]:
    pts = np.array(bbox_abs)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def point_in_quad(px: float, py: float, quad: np.ndarray) -> bool:
    contour = np.array(quad, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0


def quad_height(quad: np.ndarray) -> float:
    q = np.array(quad)
    return float(q[:, 1].max() - q[:, 1].min())


# ── Telugu quad filter (v2) ───────────────────────────────────────────────────

def is_telugu_quad(quad: np.ndarray, area: dict) -> bool:
    """
    v2 — Decides whether a CRAFT quad should be inpainted.

    MATCHED WORD CASES (an OCR word centre falls inside the quad):
      • Any matched word has Telugu chars               → ERASE
      • Any matched word is Devanagari / clean English  → SKIP (hard protect)
      • All matched words are ASCII noise               → ERASE (OCR misread)

    NO MATCH CASES (EasyOCR missed this quad entirely):

      Case A — Pure Telugu area (no protected words in sentence):
        → ERASE unconditionally.
        Catches large/bold/decorative Telugu fonts that EasyOCR misses.

      Case B — Mixed area (has Devanagari or clean English words):
        Erase only if a Telugu word centre is within quad_height × 3.5 px.
        Quad-height-relative threshold scales to the actual unmatched quad.
        Protects Hindi logos and English text in distant rows.
    """
    qcx, qcy = quad_centre(quad)
    quad_arr  = quad if isinstance(quad, np.ndarray) else np.array(quad)

    matched_texts = []
    for word in area.get('sentence', []):
        bbox = word.get('bbox_abs')
        if bbox is None:
            continue
        wcx, wcy = word_bbox_centre(bbox)
        if point_in_quad(wcx, wcy, quad_arr):
            matched_texts.append(word['text'])

    if matched_texts:
        if any(is_protected_non_telugu(t) for t in matched_texts):
            return False
        if all(is_purely_ascii_noise(t) for t in matched_texts):
            return True
        return any(contains_telugu(t) for t in matched_texts)

    # No OCR match — classify by area context
    sentence = area.get('sentence', [])
    non_telugu_words = [
        w for w in sentence
        if (w.get('text', '') and
            not contains_telugu(w['text']) and
            not is_purely_ascii_noise(w['text']) and
            is_protected_non_telugu(w['text']))
    ]

    # Case A: pure Telugu area
    if not non_telugu_words:
        return True

    # Case B: mixed area — proximity check
    q_h       = max(quad_height(quad_arr), 20.0)
    threshold = q_h * 3.5
    telugu_words = [
        w for w in sentence
        if contains_telugu(w.get('text', '')) and w.get('bbox_abs')
    ]
    if not telugu_words:
        return False
    for w in telugu_words:
        wcx, wcy = word_bbox_centre(w['bbox_abs'])
        if ((qcx - wcx) ** 2 + (qcy - wcy) ** 2) ** 0.5 < threshold:
            return True
    return False


# ── Stroke mask ───────────────────────────────────────────────────────────────

def build_stroke_mask_for_quad(img_rgb: np.ndarray,
                                quad: np.ndarray) -> np.ndarray:
    """
    Pixel-level mask of text ink strokes in a single CRAFT quad.

    Steps:
      1. Rectify quad patch (warpPerspective, 2× upscale)
      2. Grayscale + Otsu threshold → isolates ink
      3. Confidence check: if < 0.15 fall back to full quad polygon
      4. Polarity check → ink = white
      5. Dilate 2 px to catch anti-aliased edges
      6. Unmap stroke pixels to original image space via M_inv
    """
    quad = quad if isinstance(quad, np.ndarray) else np.array(quad)
    rectified, ordered, M = rectify_quad(img_rgb, quad, upscale=2.0)
    if rectified.size == 0:
        return np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    M_inv = np.linalg.inv(M)
    gray  = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
    thresh_val, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    hist  = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total > 0:
        t   = int(thresh_val)
        w0  = hist[:t].sum() / total
        w1  = hist[t:].sum() / total
        mu0 = (np.arange(t)     * hist[:t]).sum() / max(hist[:t].sum(), 1)
        mu1 = (np.arange(t,256) * hist[t:]).sum() / max(hist[t:].sum(), 1)
        mu  = mu0 * w0 + mu1 * w1
        bcv = w0 * w1 * (mu0 - mu1) ** 2
        total_var = ((np.arange(256) - mu) ** 2 * hist).sum() / total
        confidence = bcv / max(total_var, 1e-6)
        if confidence < 0.15:
            H, W = img_rgb.shape[:2]
            fallback = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(fallback, [ordered.astype(np.int32)], 255)
            return fallback

    if binary[:8, :8].mean() > 128:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)

    ys_r, xs_r = np.where(binary == 255)
    if len(xs_r) == 0:
        return np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    pts_r   = np.array([[[float(x), float(y)] for x, y in zip(xs_r, ys_r)]],
                       dtype=np.float32)
    pts_img = cv2.perspectiveTransform(pts_r, M_inv)[0]

    H, W = img_rgb.shape[:2]
    stroke_mask = np.zeros((H, W), dtype=np.uint8)
    xi = np.clip(np.round(pts_img[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pts_img[:, 1]).astype(int), 0, H - 1)
    stroke_mask[yi, xi] = 255
    return stroke_mask


def build_stroke_mask_for_area(img_rgb: np.ndarray,
                                area: dict) -> np.ndarray:
    """Combined stroke mask for all Telugu quads in one area."""
    H, W = img_rgb.shape[:2]
    combined    = np.zeros((H, W), dtype=np.uint8)
    total_quads = skipped = 0

    for quad in area['area_quads']:
        quad_arr = quad if isinstance(quad, np.ndarray) else np.array(quad)
        total_quads += 1
        if not is_telugu_quad(quad_arr, area):
            skipped += 1
            continue
        combined = cv2.bitwise_or(
            combined, build_stroke_mask_for_quad(img_rgb, quad_arr)
        )

    if skipped > 0:
        print(f'    Telugu filter: {total_quads-skipped}/{total_quads} quads masked '
              f'({skipped} non-Telugu skipped)')
    return combined


# ── Inpainting ────────────────────────────────────────────────────────────────

def inpaint_area(img_rgb: np.ndarray, area: dict) -> np.ndarray:
    """Erase Telugu text in one area using stroke mask + TELEA inpainting."""
    stroke_mask = build_stroke_mask_for_area(img_rgb, area)
    if stroke_mask.sum() == 0:
        return img_rgb

    x1, y1, x2, y2 = area['area_bbox']
    text_h = y2 - y1

    if   text_h < 40:  radius = 5
    elif text_h < 80:  radius = 8
    else:              radius = 12

    kernel      = np.ones((3, 3), np.uint8)
    dil_iters   = 3 if text_h >= 60 else 2
    stroke_mask = cv2.dilate(stroke_mask, kernel, iterations=dil_iters)

    bgr       = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, stroke_mask,
                            inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def inpaint_all_areas(img_rgb: np.ndarray,
                      processed_areas: list[dict]) -> np.ndarray:
    """Run inpainting over all processed areas."""
    result = img_rgb.copy()
    print(f'Inpainting {len(processed_areas)} areas...')
    for i, area in enumerate(processed_areas):
        bb = area['area_bbox']
        print(f'  [Area {i}]  bbox=({bb[0]},{bb[1]})→({bb[2]},{bb[3]})')
        result = inpaint_area(result, area)
    return result


def inpaint_noise_boxes(img_rgb: np.ndarray,
                        craft_boxes: list[dict],
                        processed_areas: list[dict],
                        radius: int = 5) -> np.ndarray:
    """
    Erase small CRAFT boxes (e.g. '-' separator) that were filtered as
    noise during area grouping and never entered processed_areas.

    Guards (all must pass):
      1. Box area ≤ 800 px²  — only true punctuation marks
         ('-' ≈ 120 px²  → erased ✅
          उपक्रम ≈ 3 000 px² → skipped ✅)
      2. Centre inside a confirmed Telugu area bbox (±20 px margin)
      3. Not already handled by inpaint_all_areas
    """
    processed_quads = set()
    for area in processed_areas:
        for q in area['area_quads']:
            arr = np.array(q)
            processed_quads.add(tuple(arr.flatten().astype(int).tolist()))

    area_bboxes = [area['area_bbox'] for area in processed_areas]
    result = img_rgb.copy()
    erased = 0

    for box in craft_boxes:
        quad = np.array(box['quad'])
        key  = tuple(quad.flatten().astype(int).tolist())
        if key in processed_quads:
            continue
        cx, cy = quad_centre(quad)
        xs, ys = quad[:, 0], quad[:, 1]
        box_w  = float(xs.max() - xs.min())
        box_h  = float(ys.max() - ys.min())
        if box_w * box_h > 800:
            continue
        inside = any(
            (x1 - 20) <= cx <= (x2 + 20) and (y1 - 20) <= cy <= (y2 + 20)
            for x1, y1, x2, y2 in area_bboxes
        )
        if not inside:
            continue
        mask = build_stroke_mask_for_quad(result, quad)
        if mask.sum() == 0:
            continue
        kernel = np.ones((3, 3), np.uint8)
        mask   = cv2.dilate(mask, kernel, iterations=2)
        bgr    = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = cv2.cvtColor(
            cv2.inpaint(bgr, mask, inpaintRadius=radius,
                        flags=cv2.INPAINT_TELEA),
            cv2.COLOR_BGR2RGB
        )
        erased += 1

    if erased:
        print(f'  Noise cleanup: erased {erased} small isolated box(es)')
    return result
