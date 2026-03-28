"""
vtt — Visual Text Translation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
End-to-end Telugu scene text → Tamil pipeline.

Modules:
  detection      — CRAFT box loading, dedup, area grouping & purification
  ocr            — Quad-rectified EasyOCR, line reconstruction, Telugu helpers
  translation    — Sarvam AI: image-type detection, normalization, translation
  inpainting     — Stroke mask generation (v2), Telugu filter, TELEA inpainting
  visualisation  — Matplotlib/CV2 visualisation helpers
"""

from .detection import (
    load_craft_boxes,
    deduplicate_craft_boxes,
    build_text_areas,
    merge_overlapping_areas,
    purify_areas,
    area_bbox,
    generate_area_mask,
    bbox_iou,
)

from .ocr import (
    enhance_for_ocr,
    rectify_quad,
    order_quad_points,
    ocr_single_quad,
    ocr_area,
    cluster_into_lines,
    reconstruct_area_sentence,
    contains_telugu,
    count_telugu_chars,
    is_telugu_area,
    split_telugu_and_other,
    deduplicate_ocr_across_areas,
    TELUGU_RANGE,
)

from .translation import (
    detect_image_type,
    normalize_telugu_ocr,
    translate_areas,
    IMAGE_TYPE_DESCRIPTIONS,
)

from .inpainting import (
    # stroke mask
    build_stroke_mask_for_quad,
    build_stroke_mask_for_area,
    # inpainting
    inpaint_area,
    inpaint_all_areas,
    inpaint_noise_boxes,
    # helpers / classifiers
    is_telugu_quad,
    is_purely_ascii_noise,
    is_protected_non_telugu,
    contains_devanagari,
    quad_centre,
    quad_height,
)

from .visualisation import (
    show_craft_results,
    visualize_areas,
    visualize_final_areas,
    visualise_stroke_masks,
    visualize_inpainted,
)

__version__ = '2.0.0'
__all__ = [
    # detection
    'load_craft_boxes', 'deduplicate_craft_boxes', 'build_text_areas',
    'merge_overlapping_areas', 'purify_areas', 'area_bbox',
    'generate_area_mask', 'bbox_iou',
    # ocr
    'enhance_for_ocr', 'rectify_quad', 'order_quad_points',
    'ocr_single_quad', 'ocr_area', 'cluster_into_lines',
    'reconstruct_area_sentence', 'contains_telugu', 'count_telugu_chars',
    'is_telugu_area', 'split_telugu_and_other', 'deduplicate_ocr_across_areas',
    'TELUGU_RANGE',
    # translation
    'detect_image_type', 'normalize_telugu_ocr', 'translate_areas',
    'IMAGE_TYPE_DESCRIPTIONS',
    # inpainting
    'build_stroke_mask_for_quad', 'build_stroke_mask_for_area',
    'inpaint_area', 'inpaint_all_areas', 'inpaint_noise_boxes',
    'is_telugu_quad', 'is_purely_ascii_noise', 'is_protected_non_telugu',
    'contains_devanagari', 'quad_centre', 'quad_height',
    # visualisation
    'show_craft_results', 'visualize_areas', 'visualize_final_areas',
    'visualise_stroke_masks', 'visualize_inpainted',
]
