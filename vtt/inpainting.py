"""
vtt/inpainting.py
─────────────────
Phase 2 — Telugu text removal via tight stroke masks.

Key decisions:
  - Stroke mask: per-quad Otsu thresholding isolates ink pixels,
    unmapped via M_inv back to image space (user's idea).
  - Telugu-only filter: skips quads whose matched OCR words are
    non-Telugu (English logos, numbers, Hindi text).
  - Noise cleanup pass: erases tiny CRAFT boxes (e.g. '-' separator)
    that were filtered as noise before area grouping.
  - Devanagari protected: quads with no OCR match in mixed-script
    areas are skipped to preserve Hindi text (उपक्रम etc.).
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
    """Returns True if text contains Devanagari (Hindi, Sanskrit, Marathi)."""
    return any(0x0900 <= ord(c) <= 0x097F for c in text)


def is_purely_ascii_noise(text: str) -> bool:
    """
    Returns True for short OCR strings that are ASCII noise standing
    in place of a Telugu character.
    e.g. 'SOll' (misread from 'కం॥'), '-', ',', 'Il', 'S0'
    """
    if not text:
        return False
    # Single non-letter character → always noise (-, ,, ., |)
    if len(text.strip()) == 1 and not text.strip().isalpha():
        return True
    # Telugu punctuation mixed with ASCII (e.g. 'S0॥')
    telugu_punct = set('॥।०१२३४५६७८९')
    has_tp = any(c in telugu_punct for c in text)
    all_ap = all(ord(c) < 128 or c in telugu_punct for c in text)
    if has_tp and all_ap and len(text.strip()) <= 6:
        return True
    # Short pure ASCII (≤ 4 chars)
    if all(ord(c) < 128 for c in text) and len(text.strip()) <= 4:
        return True
    return False


def area_is_pure_telugu(area: dict) -> bool:
    """
    Returns True if ALL OCR words in this area are Telugu or ASCII noise.
    Returns False if the area contains real English / mixed-script words.
    Used as context signal for the no-match fallback in is_telugu_quad().
    """
    for word in area.get('sentence', []):
        text = word.get('text', '')
        if not text:
            continue
        if (all(ord(c) < 128 for c in text)
                and len(text.strip()) > 4
                and not is_purely_ascii_noise(text)):
            return False
    return True


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


# ── Telugu quad filter ────────────────────────────────────────────────────────

def is_telugu_quad(quad: np.ndarray, area: dict) -> bool:
    """
    Decides whether a CRAFT quad should be inpainted.

    Decision table:
      matched word has Telugu chars            → ERASE
      matched word is ASCII noise ('SOll')     → ERASE
      matched word is clean English/numbers    → SKIP
      no match + pure Telugu area              → ERASE  (missed OCR like 'అంద')
      no match + mixed area + same row
        as Telugu words                        → ERASE  (separator like '-')
      no match + mixed area + different row    → SKIP   (Hindi, logos)
    """
    qcx, qcy = quad_centre(quad)

    matched_texts = []
    for word in area.get('sentence', []):
        bbox = word.get('bbox_abs')
        if bbox is None:
            continue
        wcx, wcy = word_bbox_centre(bbox)
        if point_in_quad(wcx, wcy, quad):
            matched_texts.append(word['text'])

    # ── No OCR match: use area context ────────────────────────────────────────
    if not matched_texts:
        if area_is_pure_telugu(area):
            return True   # Pure Telugu area → missed OCR → erase
        # Mixed area: erase only if this quad sits on the same row as Telugu words
        telugu_centres = []
        for word in area.get('sentence', []):
            if contains_telugu(word.get('text', '')) and word.get('bbox_abs'):
                cx, cy = word_bbox_centre(word['bbox_abs'])
                telugu_centres.append((cx, cy))
        same_row = [(cx, cy) for cx, cy in telugu_centres
                    if abs(cy - qcy) < 60]
        return len(same_row) >= 1

    # ── ASCII noise standing in for Telugu → erase ────────────────────────────
    if all(is_purely_ascii_noise(t) for t in matched_texts):
        return True

    # ── Normal case ───────────────────────────────────────────────────────────
    return any(contains_telugu(t) for t in matched_texts)


# ── Stroke mask ───────────────────────────────────────────────────────────────

def build_stroke_mask_for_quad(img_rgb: np.ndarray,
                                quad: np.ndarray) -> np.ndarray:
    """
    Build a tight pixel mask of text strokes inside one CRAFT quad.

    Steps:
      1. Rectify the quad patch (warpPerspective, 2× upscale)
      2. Grayscale + Otsu threshold → isolates ink from background
      3. Otsu confidence check — falls back to full quad polygon
         if histogram is unimodal (text over complex background)
      4. Polarity check → ensure text pixels = white
      5. Dilate 2px to catch anti-aliased edges
      6. Unmap stroke pixels back to original image via M_inv
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

    # Otsu confidence (between-class variance / total variance)
    hist  = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total > 0:
        t   = int(thresh_val)
        w0  = hist[:t].sum() / total
        w1  = hist[t:].sum() / total
        mu0 = (np.arange(t)     * hist[:t]).sum() / max(hist[:t].sum(), 1)
        mu1 = (np.arange(t,256) * hist[t:]).sum() / max(hist[t:].sum(), 1)
        mu  = mu0*w0 + mu1*w1
        bcv = w0 * w1 * (mu0 - mu1)**2
        total_var = ((np.arange(256) - mu)**2 * hist).sum() / total
        confidence = bcv / max(total_var, 1e-6)
        if confidence < 0.15:           # low contrast → fall back to full poly
            H, W = img_rgb.shape[:2]
            fallback = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(fallback, [ordered.astype(np.int32)], 255)
            return fallback

    # Polarity: text must be white
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
    xi = np.clip(np.round(pts_img[:, 0]).astype(int), 0, W-1)
    yi = np.clip(np.round(pts_img[:, 1]).astype(int), 0, H-1)
    stroke_mask[yi, xi] = 255
    return stroke_mask


def build_stroke_mask_for_area(img_rgb: np.ndarray, area: dict) -> np.ndarray:
    """Build combined stroke mask for all Telugu quads in one area."""
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
    Erase small CRAFT boxes (e.g. '-' separator) that were filtered as noise
    during area grouping and never entered processed_areas.

    Size threshold 800px² is intentionally strict:
      '-' separator  ≈  15×8  =  120px²  → erased ✅
      उपक्रम word   ≈ 120×25 = 3000px²  → skipped ✅
      सिंडिकेटबैंक  ≈ 200×40 = 8000px²  → skipped ✅
    """
    # Collect quads already handled by inpaint_all_areas
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

        if box_w * box_h > 800:     # not tiny punctuation
            continue

        inside = any(
            (x1-20) <= cx <= (x2+20) and (y1-20) <= cy <= (y2+20)
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
            cv2.inpaint(bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA),
            cv2.COLOR_BGR2RGB
        )
        erased += 1

    if erased:
        print(f'  Noise cleanup: erased {erased} small isolated box(es)')
    return result
