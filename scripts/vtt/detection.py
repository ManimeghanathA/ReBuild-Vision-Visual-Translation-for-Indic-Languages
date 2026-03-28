"""
vtt/detection.py
────────────────
CRAFT box loading, deduplication, area grouping, merging and purification.

Fixes implemented:
  B1 — Running-median cy + stride cap prevents area drift
  B2 — CRAFT box deduplication (IoU containment > 70%)
  B3 — Iterative merge with v_thresh=0.40, h_thresh=0.15
  B7 — v/h thresholds prevent cross-board merging
"""

import numpy as np
import cv2


# ── CRAFT box loading ─────────────────────────────────────────────────────────

def load_craft_boxes(txt_path: str) -> list[dict]:
    """Parse CRAFT result .txt file into a list of box dicts."""
    boxes = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pts = np.array(list(map(int, line.split(',')))).reshape(4, 2)
            except ValueError:
                continue
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            boxes.append({
                'quad': pts,
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'cx':   float((x1 + x2) / 2),
                'cy':   float((y1 + y2) / 2),
                'w':    int(x2 - x1),
                'h':    int(y2 - y1),
            })
    return boxes


# ── IoU / containment ─────────────────────────────────────────────────────────

def bbox_iou(b1: tuple, b2: tuple) -> tuple[float, float]:
    """Returns (iou, containment_of_b1_inside_b2)."""
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter   = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a  = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b  = max(1, (bx2 - bx1) * (by2 - by1))
    union   = area_a + area_b - inter
    iou         = inter / union if union > 0 else 0.0
    containment = inter / area_a
    return iou, containment


def deduplicate_craft_boxes(boxes: list[dict],
                             containment_thresh: float = 0.70) -> list[dict]:
    """Remove boxes largely contained inside a bigger box. (Fix B2)"""
    boxes = sorted(boxes, key=lambda b: b['w'] * b['h'], reverse=True)
    keep  = []
    for box in boxes:
        suppressed = False
        for kept in keep:
            _, contained = bbox_iou(box['bbox'], kept['bbox'])
            if contained > containment_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(box)
    return keep


# ── Area helpers ──────────────────────────────────────────────────────────────

def area_median_cy(area: list[dict]) -> float:
    return float(np.median([b['cy'] for b in area]))

def area_median_h(area: list[dict]) -> float:
    return float(np.median([b['h'] for b in area]))

def area_bbox(area: list[dict]) -> tuple[int, int, int, int]:
    x1 = int(min(b['bbox'][0] for b in area))
    y1 = int(min(b['bbox'][1] for b in area))
    x2 = int(max(b['bbox'][2] for b in area))
    y2 = int(max(b['bbox'][3] for b in area))
    return (x1, y1, x2, y2)


# ── Area grouping ─────────────────────────────────────────────────────────────

def build_text_areas(boxes: list[dict],
                     v_tol: float = 0.6,
                     h_ratio: float = 0.5,
                     max_line_stride: float = 1.2) -> list[list[dict]]:
    """Group CRAFT boxes into text-line areas. (Fix B1)"""
    boxes_sorted = sorted(boxes, key=lambda b: (b['cy'], b['cx']))
    areas = []
    for box in boxes_sorted:
        best_area = None
        best_dist = float('inf')
        for area in areas:
            med_cy = area_median_cy(area)
            med_h  = area_median_h(area)
            cy_dist = abs(box['cy'] - med_cy)
            if cy_dist > max_line_stride * med_h:
                continue
            if cy_dist > v_tol * med_h:
                continue
            h_sim = min(box['h'], med_h) / max(box['h'], med_h)
            if h_sim < h_ratio:
                continue
            if cy_dist < best_dist:
                best_dist = cy_dist
                best_area = area
        if best_area is not None:
            best_area.append(box)
        else:
            areas.append([box])
    areas.sort(key=lambda a: area_median_cy(a))
    return areas


# ── Area merging ──────────────────────────────────────────────────────────────

def vertical_overlap_ratio(bb1: tuple, bb2: tuple) -> float:
    y_ov = max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))
    return y_ov / min(max(1, bb1[3]-bb1[1]), max(1, bb2[3]-bb2[1]))

def horizontal_overlap_ratio(bb1: tuple, bb2: tuple) -> float:
    x_ov = max(0, min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]))
    return x_ov / min(max(1, bb1[2]-bb1[0]), max(1, bb2[2]-bb2[0]))

def should_merge(bb1: tuple, bb2: tuple,
                 v_thresh: float = 0.40,
                 h_thresh: float = 0.15) -> bool:
    return (vertical_overlap_ratio(bb1, bb2)   > v_thresh and
            horizontal_overlap_ratio(bb1, bb2) > h_thresh)

def merge_overlapping_areas(areas: list[list[dict]],
                             v_thresh: float = 0.40,
                             h_thresh: float = 0.15) -> list[list[dict]]:
    """Iteratively merge overlapping areas. (Fix B3, B7)"""
    changed = True
    while changed:
        changed = False
        merged = []
        used   = set()
        for i in range(len(areas)):
            if i in used:
                continue
            current = areas[i]
            for j in range(i + 1, len(areas)):
                if j in used:
                    continue
                if should_merge(area_bbox(current), area_bbox(areas[j]),
                                v_thresh, h_thresh):
                    current = current + areas[j]
                    used.add(j)
                    changed = True
            merged.append(current)
            used.add(i)
        areas = sorted(merged, key=lambda a: area_median_cy(a))
    return areas


# ── Area purification ─────────────────────────────────────────────────────────

def generate_area_mask(img_shape: tuple, area: list[dict]):
    """Build a binary pixel mask for all quads in an area."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for b in area:
        cv2.fillPoly(mask, [b['quad'].astype(np.int32)], 255)
    return mask

def is_valid_text_area(area: list[dict], img_shape: tuple) -> bool:
    """Filter out noise/tiny areas."""
    h_img, w_img = img_shape[:2]
    img_area = w_img * h_img
    x1, y1, x2, y2 = area_bbox(area)
    aw, ah = x2 - x1, y2 - y1
    a = aw * ah
    if aw < 20 or ah < 20:
        return False
    if a < 0.0004 * img_area:
        return False
    if len(area) == 1 and a < 0.002 * img_area:
        return False
    return True

def purify_areas(areas: list[list[dict]],
                 img_shape: tuple) -> tuple[list, list]:
    """Split into valid and noise areas."""
    valid = [a for a in areas if     is_valid_text_area(a, img_shape)]
    noise = [a for a in areas if not is_valid_text_area(a, img_shape)]
    return valid, noise
