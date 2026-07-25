"""
vtt/ocr.py
──────────
Quad-rectified PaddleOCR, line reconstruction, Telugu helpers,
and cross-area OCR deduplication.

Fixes implemented:
  B4  — Cross-area OCR deduplication
  B5  — Consistent 'sentence' key
  B6  — Positions from bbox_abs (truly absolute)
  B8  — cx/cy derived from bbox_abs
  B9  — Quad-rectified OCR per CRAFT box (warpPerspective)
  B10 — conf_threshold=0.15 drops junk detections
  B11 — Height-adaptive line clustering (h_compat > 0.40)
  B12 — CLAHE on LAB L-channel for OCR quality
  B13 — Pixel clip to area bbox before rectifying (ghost word prevention)
"""

import re
import os
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from .detection import bbox_iou, area_bbox

# Telugu Unicode range
TELUGU_RANGE = r'[\u0C00-\u0C7F]'

DEFAULT_PADDLE_CACHE = Path(__file__).resolve().parents[2] / 'models' / 'paddleocr'
DEFAULT_PADDLE_MODEL = 'te_PP-OCRv5_mobile_rec'


class PaddleOCRRecognizer:
    """Recognition-only PaddleOCR wrapper for already-rectified CRAFT crops."""

    def __init__(self,
                 model_name: str = DEFAULT_PADDLE_MODEL,
                 cache_dir: str | os.PathLike | None = None,
                 use_gpu: bool = True):
        cache_path = Path(cache_dir) if cache_dir else DEFAULT_PADDLE_CACHE
        cache_path.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(cache_path.resolve()))
        os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
        os.environ.setdefault('MODELSCOPE_CACHE', str((cache_path / 'modelscope').resolve()))
        os.environ.setdefault('HF_HOME', str((cache_path / 'huggingface').resolve()))
        os.environ.setdefault(
            'HUGGINGFACE_HUB_CACHE',
            str((cache_path / 'huggingface' / 'hub').resolve()),
        )

        from paddleocr import TextRecognition

        self.device = 'cpu'
        if use_gpu:
            try:
                import paddle  # type: ignore

                if paddle.is_compiled_with_cuda():
                    self.device = 'gpu'
            except Exception:
                self.device = 'gpu'

        init_attempts = [
            {'model_name': model_name, 'device': self.device},
            {'model_name': model_name, 'device': 'cpu'},
            {'model_name': model_name},
        ]
        last_error: Exception | None = None
        for kwargs in init_attempts:
            try:
                self.model = TextRecognition(**kwargs)
                self.device = kwargs.get('device', self.device)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f'Could not initialize PaddleOCR: {last_error}')

    def readtext(self, crop_rgb: np.ndarray) -> list[tuple[str, float]]:
        """Return PaddleOCR recognition as [(text, confidence)]."""
        result = self.model.predict(crop_rgb)
        text, confidence = parse_paddle_result(result)
        if not text:
            return []
        return [(text, confidence if confidence is not None else 1.0)]


def create_ocr_reader(use_gpu: bool = True,
                      cache_dir: str | os.PathLike | None = None,
                      model_name: str = DEFAULT_PADDLE_MODEL) -> PaddleOCRRecognizer:
    """Create the OCR recognizer used by the main pipeline."""
    return PaddleOCRRecognizer(
        model_name=model_name,
        cache_dir=cache_dir,
        use_gpu=use_gpu,
    )


def parse_paddle_result(result: Any) -> tuple[str, float | None]:
    """Extract text and average confidence from PaddleOCR 3.x results."""
    texts: list[str] = []
    confs: list[float] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            for key in ('rec_text', 'text'):
                value = node.get(key)
                if value:
                    texts.append(str(value).strip())
            for key in ('rec_score', 'score', 'confidence'):
                value = node.get(key)
                if value is not None:
                    try:
                        confs.append(float(value))
                    except (TypeError, ValueError):
                        pass
            for value in node.values():
                visit(value)
            return
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[1], (list, tuple)) and node[1]:
                if isinstance(node[1][0], str):
                    texts.append(node[1][0].strip())
                    if len(node[1]) > 1:
                        try:
                            confs.append(float(node[1][1]))
                        except (TypeError, ValueError):
                            pass
            for item in node:
                visit(item)

    visit(result)
    text = ' '.join(t for t in texts if t).strip()
    confidence = sum(confs) / len(confs) if confs else None
    return text, confidence


# ── CLAHE contrast enhancement ────────────────────────────────────────────────

def enhance_for_ocr(crop_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE on LAB L-channel to improve OCR contrast. (Fix B12)"""
    if crop_rgb.size == 0:
        return crop_rgb
    lab  = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)


# ── Quad geometry ─────────────────────────────────────────────────────────────

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order quad corners: TL, TR, BR, BL."""
    pts  = pts.astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def rectify_quad(img: np.ndarray,
                 quad: np.ndarray,
                 upscale: float = 2.0) -> tuple:
    """Perspective-rectify a single CRAFT quad. (Fix B9)"""
    ordered = order_quad_points(quad)
    tl, tr, br, bl = ordered
    w = max(1, int(max(np.linalg.norm(tr - tl),
                       np.linalg.norm(br - bl)) * upscale))
    h = max(1, int(max(np.linalg.norm(bl - tl),
                       np.linalg.norm(br - tr)) * upscale))
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]],
                   dtype=np.float32)
    M   = cv2.getPerspectiveTransform(ordered, dst)
    rectified = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)
    return rectified, ordered, M

def unmap_point(px: float, py: float, M_inv: np.ndarray) -> tuple[float, float]:
    """Map a point from rectified space back to image space."""
    pt     = np.array([[[px, py]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, M_inv)
    return float(mapped[0][0][0]), float(mapped[0][0][1])


# ── Single-quad OCR ───────────────────────────────────────────────────────────

def ocr_single_quad(img: np.ndarray,
                    box_dict: dict,
                    ocr_reader,
                    conf_threshold: float = 0.15,
                    clip_bbox: tuple | None = None) -> list[dict]:
    """
    OCR one CRAFT quad with perspective rectification.

    Fix B13 — clip image pixels to area bbox before rectifying to prevent
    adjacent sign's text bleeding into this box's rectified patch.
    Fix B10 — confidence threshold 0.15 drops junk detections.
    """
    quad = box_dict['quad']

    if clip_bbox is not None:
        cx1, cy1, cx2, cy2 = clip_bbox
        img_clipped = img.copy()
        if cy1 > 0:             img_clipped[:cy1, :]  = 0
        if cy2 < img.shape[0]:  img_clipped[cy2:, :]  = 0
        if cx1 > 0:             img_clipped[:, :cx1]  = 0
        if cx2 < img.shape[1]:  img_clipped[:, cx2:]  = 0
    else:
        img_clipped = img

    rectified, ordered, M = rectify_quad(img_clipped, quad, upscale=2.0)
    if rectified.size == 0:
        return []

    enhanced  = enhance_for_ocr(rectified, clip_limit=2.0)
    M_inv     = np.linalg.inv(M)
    quad_poly = ordered.astype(np.float32)

    results = ocr_reader.readtext(enhanced)
    words   = []

    h_rect, w_rect = enhanced.shape[:2]
    full_bbox_rect = [
        [0.0, 0.0],
        [float(max(w_rect - 1, 0)), 0.0],
        [float(max(w_rect - 1, 0)), float(max(h_rect - 1, 0))],
        [0.0, float(max(h_rect - 1, 0))],
    ]

    for text, conf in results:
        text = text.strip()
        if not text or float(conf) < conf_threshold:
            continue

        bbox_abs = [list(unmap_point(p[0], p[1], M_inv)) for p in full_bbox_rect]
        xs = [p[0] for p in bbox_abs]; ys = [p[1] for p in bbox_abs]
        x1a, x2a = min(xs), max(xs)
        y1a, y2a = min(ys), max(ys)
        cx = (x1a + x2a) / 2
        cy = (y1a + y2a) / 2

        inside = cv2.pointPolygonTest(quad_poly, (cx, cy), measureDist=False)
        if inside < 0:
            continue

        words.append({
            'text':     text,
            'conf':     float(conf),
            'bbox_rel': [[float(p[0]), float(p[1])] for p in full_bbox_rect],
            'bbox_abs': bbox_abs,
            'cx': cx, 'cy': cy,
            'w':  x2a - x1a,
            'h':  y2a - y1a,
        })
    return words


def ocr_area(img: np.ndarray,
             area: list[dict],
             ocr_reader,
             conf_threshold: float = 0.15,
             overlap_thresh: float = 0.50) -> list[dict]:
    """OCR all quads in an area, deduplicate within area."""
    all_words = []
    ab = area_bbox(area)
    for box in area:
        words = ocr_single_quad(img, box, ocr_reader,
                                conf_threshold, clip_bbox=ab)
        all_words.extend(words)

    all_words.sort(key=lambda w: w['conf'], reverse=True)
    kept = []
    for w in all_words:
        duplicate = False
        wx1 = w['cx'] - w['w']/2;  wx2 = w['cx'] + w['w']/2
        wy1 = w['cy'] - w['h']/2;  wy2 = w['cy'] + w['h']/2
        for k in kept:
            kx1 = k['cx']-k['w']/2; kx2 = k['cx']+k['w']/2
            ky1 = k['cy']-k['h']/2; ky2 = k['cy']+k['h']/2
            ix = max(0, min(wx2, kx2) - max(wx1, kx1))
            iy = max(0, min(wy2, ky2) - max(wy1, ky1))
            if ix * iy / max(1, w['w'] * w['h']) > overlap_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(w)
    return kept


# ── Line reconstruction ───────────────────────────────────────────────────────

def cluster_into_lines(words: list[dict], v_tol: float = 0.5) -> list[list[dict]]:
    """Height-adaptive line clustering. (Fix B11)"""
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: w['cy'])
    lines = []
    for w in words_sorted:
        best_line  = None
        best_score = float('inf')
        for line in lines:
            line_cy  = float(np.median([x['cy'] for x in line]))
            line_h   = float(np.median([x['h']  for x in line]))
            cy_dist  = abs(w['cy'] - line_cy)
            h_compat = min(w['h'], line_h) / max(w['h'], line_h)
            if cy_dist < v_tol * line_h and h_compat > 0.40:
                score = cy_dist / max(line_h, 1)
                if score < best_score:
                    best_score = score
                    best_line  = line
        if best_line is not None:
            best_line.append(w)
        else:
            lines.append([w])
    return lines


def reconstruct_area_sentence(ocr_words: list[dict]) -> list[dict]:
    """Build ordered word list with line_idx. (Fix B5, B6, B8)"""
    lines = cluster_into_lines(ocr_words)
    lines = sorted(lines,
                   key=lambda l: float(np.median([w['cy'] for w in l])))
    structured = []
    for line_idx, line in enumerate(lines):
        for w in sorted(line, key=lambda w: w['cx']):
            structured.append({
                'text':     w['text'],
                'conf':     w['conf'],
                'line_idx': line_idx,
                'bbox_rel': w['bbox_rel'],
                'bbox_abs': w['bbox_abs'],
                'center':   (w['cx'], w['cy']),
                'width':    w['w'],
                'height':   w['h'],
            })
    return structured


# ── Telugu helpers ────────────────────────────────────────────────────────────

def count_telugu_chars(text: str) -> int:
    return len(re.findall(TELUGU_RANGE, text))

def contains_telugu(text: str) -> bool:
    return count_telugu_chars(text) > 0

def is_telugu_area(ocr_results: list[dict], min_telugu_chars: int = 2) -> bool:
    return count_telugu_chars(
        ' '.join(r['text'] for r in ocr_results)
    ) >= min_telugu_chars

def split_telugu_and_other(ocr_results: list[dict]) -> tuple[list, list]:
    telugu = [r['text'] for r in ocr_results if     contains_telugu(r['text'])]
    other  = [r['text'] for r in ocr_results if not contains_telugu(r['text'])]
    return telugu, other


# ── Cross-area OCR deduplication ──────────────────────────────────────────────

def _ocr_bbox_iou_abs(w1: dict, w2: dict) -> float:
    def to_rect(w):
        xs = [p[0] for p in w['bbox_abs']]
        ys = [p[1] for p in w['bbox_abs']]
        return min(xs), min(ys), max(xs), max(ys)
    _, contained = bbox_iou(to_rect(w1), to_rect(w2))
    return contained


def deduplicate_ocr_across_areas(processed_areas: list[dict],
                                  overlap_thresh: float = 0.50) -> list[dict]:
    """Remove duplicate words that appear in multiple areas. (Fix B4)"""
    all_words = []
    for i, area in enumerate(processed_areas):
        for w in area['sentence']:
            all_words.append((i, w))

    keep_flags = [True] * len(all_words)
    for i in range(len(all_words)):
        if not keep_flags[i]:
            continue
        for j in range(i + 1, len(all_words)):
            if not keep_flags[j]:
                continue
            area_i, wi = all_words[i]
            area_j, wj = all_words[j]
            if area_i == area_j:
                continue
            if _ocr_bbox_iou_abs(wi, wj) > overlap_thresh:
                if wi['conf'] >= wj['conf']:
                    keep_flags[j] = False
                else:
                    keep_flags[i] = False
                    break

    keep_per_area = {i: [] for i in range(len(processed_areas))}
    for flag, (area_idx, w) in zip(keep_flags, all_words):
        if flag:
            keep_per_area[area_idx].append(w)

    deduped = []
    for i, area in enumerate(processed_areas):
        cleaned = keep_per_area[i]
        if cleaned:
            deduped.append({
                **area,
                'sentence':  cleaned,
                'full_text': ' '.join(w['text'] for w in cleaned),
            })
    return deduped
