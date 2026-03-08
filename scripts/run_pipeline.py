#!/usr/bin/env python3
"""
scripts/run_pipeline.py
────────────────────────
Full end-to-end pipeline: detection → OCR → translation → inpainting.

Usage:
    python scripts/run_pipeline.py \
        --image  data/images/img7.jpeg \
        --result result/res_img7.txt \
        --api-key YOUR_SARVAM_API_KEY

Outputs (in output/ folder):
    {img_stem}_inpainted.jpg
    {img_stem}_translation_results.json
"""

import argparse
import json
import os
import re
import time

import cv2
import easyocr
import matplotlib.pyplot as plt

from vtt import (
    load_craft_boxes,
    deduplicate_craft_boxes,
    build_text_areas,
    merge_overlapping_areas,
    purify_areas,
    area_bbox,
    generate_area_mask,
    ocr_area,
    reconstruct_area_sentence,
    is_telugu_area,
    split_telugu_and_other,
    deduplicate_ocr_across_areas,
    detect_image_type,
    normalize_telugu_ocr,
    translate_areas,
    IMAGE_TYPE_DESCRIPTIONS,
    inpaint_all_areas,
    inpaint_noise_boxes,
    visualize_inpainted,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Telugu scene text → Tamil visual translation pipeline'
    )
    p.add_argument('--image',   required=True,  help='Path to source image')
    p.add_argument('--result',  required=True,  help='Path to CRAFT result .txt')
    p.add_argument('--api-key', required=True,  help='Sarvam AI API key')
    p.add_argument('--output',  default='output', help='Output directory')
    p.add_argument('--no-gpu',  action='store_true', help='Force CPU for EasyOCR')
    p.add_argument('--show',    action='store_true', help='Show before/after plot')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ── Load image ────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f'Image not found: {args.image}')
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_stem = os.path.splitext(os.path.basename(args.image))[0]
    print(f'Loaded : {args.image}  ({img.shape[1]}w × {img.shape[0]}h)')

    # ── PHASE 1: Detection ────────────────────────────────────────────────────
    print('\n── Phase 1: Detection ──')
    raw_boxes = load_craft_boxes(args.result)
    boxes     = deduplicate_craft_boxes(raw_boxes)
    print(f'CRAFT boxes: {len(raw_boxes)} raw → {len(boxes)} deduped')

    areas_raw    = build_text_areas(boxes)
    areas_merged = merge_overlapping_areas(areas_raw)
    valid_areas, noise_areas = purify_areas(areas_merged, img.shape)
    print(f'Areas: {len(areas_raw)} raw → {len(areas_merged)} merged '
          f'→ {len(valid_areas)} valid, {len(noise_areas)} noise')

    # ── PHASE 1: OCR ─────────────────────────────────────────────────────────
    print('\n── Phase 1: OCR ──')
    use_gpu = not args.no_gpu
    ocr_reader = easyocr.Reader(['te', 'en'], gpu=use_gpu)
    print(f'EasyOCR ready (GPU={use_gpu})')

    processed_areas_raw = []
    for idx, area in enumerate(valid_areas):
        ocr_results = ocr_area(img, area, ocr_reader)
        if not ocr_results:
            continue
        if not is_telugu_area(ocr_results, min_telugu_chars=2):
            continue
        sentence = reconstruct_area_sentence(ocr_results)
        if not sentence:
            continue
        mask = generate_area_mask(img.shape, area)
        telugu_words, other_words = split_telugu_and_other(ocr_results)
        processed_areas_raw.append({
            'area_idx':     idx,
            'area_bbox':    area_bbox(area),
            'area_quads':   [b['quad'] for b in area],
            'sentence':     sentence,
            'full_text':    ' '.join(w['text'] for w in sentence),
            'telugu_words': telugu_words,
            'other_words':  other_words,
            'raw_ocr':      ocr_results,
            'mask':         mask,
        })

    processed_areas = deduplicate_ocr_across_areas(processed_areas_raw)
    print(f'Telugu areas: {len(processed_areas_raw)} → {len(processed_areas)} after dedup')

    # ── PHASE 2: Translation ──────────────────────────────────────────────────
    print('\n── Phase 2: Translation ──')
    image_type = detect_image_type(processed_areas, args.api_key)
    print(f'Image type (auto): {image_type}')

    print('Normalizing OCR text...')
    for area in processed_areas:
        raw = area.get('full_text', '').strip()
        area['corrected_telugu'] = normalize_telugu_ocr(raw, args.api_key) if raw else ''
        time.sleep(0.3)

    print('Translating all areas...')
    corrected     = [a.get('corrected_telugu', '') for a in processed_areas]
    tamil_results = translate_areas(corrected, image_type, args.api_key)
    for i, area in enumerate(processed_areas):
        area['tamil_translation'] = tamil_results[i]

    # Save translation JSON
    def strip_think(t):
        return re.sub(r'<think>.*?</think>', '', t or '', flags=re.DOTALL).strip()

    save_data = []
    for i, area in enumerate(processed_areas):
        save_data.append({
            'area_index':        i,
            'area_bbox':         list(area['area_bbox']),
            'raw_ocr':           area.get('full_text', ''),
            'corrected_telugu':  strip_think(area.get('corrected_telugu', '')),
            'tamil_translation': area.get('tamil_translation', ''),
        })

    json_path = os.path.join(args.output, f'{img_stem}_translation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f'Saved: {json_path}')

    # ── PHASE 2: Inpainting ───────────────────────────────────────────────────
    print('\n── Phase 2: Inpainting ──')
    inpainted = inpaint_all_areas(img, processed_areas)
    inpainted = inpaint_noise_boxes(inpainted, raw_boxes, processed_areas)

    img_path = os.path.join(args.output, f'{img_stem}_inpainted.jpg')
    cv2.imwrite(img_path, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
    print(f'Saved: {img_path}')

    if args.show:
        visualize_inpainted(img, inpainted)

    print('\n✅ Pipeline complete.')
    print(f'   Outputs in: {args.output}/')


if __name__ == '__main__':
    main()
