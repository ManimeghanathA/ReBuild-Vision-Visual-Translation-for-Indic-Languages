#!/usr/bin/env python3
"""
scripts/run_pipeline.py
────────────────────────
Full end-to-end pipeline: detection -> OCR -> translation -> inpainting.

Three modes
-----------
1. Single image:
       python scripts/run_pipeline.py \
           --image  data/images/img7.jpeg \
           --result CRAFT-pytorch/result/res_img7.txt \
           --api-key YOUR_KEY

2. All images in a folder:
       python scripts/run_pipeline.py \
           --image-dir  data/images \
           --result-dir CRAFT-pytorch/result \
           --api-key    YOUR_KEY

3. Selective images (comma-separated stems):
       python scripts/run_pipeline.py \
           --image-dir  data/images \
           --result-dir CRAFT-pytorch/result \
           --select     img1,img3,img7 \
           --api-key    YOUR_KEY

Optional flags
--------------
  --output DIR        output directory (default: output)
  --no-gpu            force CPU for EasyOCR
  --show              show before/after plot for each image
  --skip-translate    skip Sarvam AI calls, inpainting only (good for testing)

Outputs (in --output folder)
-----------------------------
  {stem}_inpainted.jpg
  {stem}_translation_results.json   (skipped with --skip-translate)
"""

import argparse
import json
import os
import re
import sys
import time

import cv2
import easyocr

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

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff',
    '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF',
}


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Telugu scene text -> Tamil visual translation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    single = p.add_argument_group('Single-image mode')
    single.add_argument('--image',  default=None,
                        help='Path to a single source image')
    single.add_argument('--result', default=None,
                        help='Path to matching CRAFT result .txt')

    batch = p.add_argument_group('Batch / selective mode')
    batch.add_argument('--image-dir',  default=None,
                       help='Folder of source images')
    batch.add_argument('--result-dir', default=None,
                       help='Folder of CRAFT result .txt files '
                            '(expects res_{stem}.txt naming)')
    batch.add_argument('--select', default=None,
                       help='Comma-separated image stems to process '
                            '(e.g. img1,img3,img7). '
                            'Omit to process ALL images in --image-dir.')

    p.add_argument('--api-key',        default='',
                   help='Sarvam AI API key (required unless --skip-translate)')
    p.add_argument('--output',         default='output',
                   help='Output directory (default: output)')
    p.add_argument('--no-gpu',         action='store_true',
                   help='Force CPU for EasyOCR')
    p.add_argument('--show',           action='store_true',
                   help='Show before/after plot for each image')
    p.add_argument('--skip-translate', action='store_true',
                   help='Skip Sarvam AI translation (inpainting only)')

    return p.parse_args()


# ── Image pair collection ─────────────────────────────────────────────────────

def collect_pairs(args):
    """
    Return list of (image_path, result_path) pairs.
    Validates existence; skips with a warning if CRAFT result is missing.
    """
    pairs = []

    # Single-image mode
    if args.image and args.result:
        if not os.path.exists(args.image):
            sys.exit(f'[ERROR] Image not found: {args.image}')
        if not os.path.exists(args.result):
            sys.exit(f'[ERROR] CRAFT result not found: {args.result}')
        pairs.append((args.image, args.result))
        return pairs

    # Batch / selective mode
    if not args.image_dir or not args.result_dir:
        sys.exit(
            '[ERROR] Provide either:\n'
            '  --image + --result          (single image)\n'
            '  --image-dir + --result-dir  (batch or selective)'
        )

    if not os.path.isdir(args.image_dir):
        sys.exit(f'[ERROR] --image-dir not found: {args.image_dir}')
    if not os.path.isdir(args.result_dir):
        sys.exit(f'[ERROR] --result-dir not found: {args.result_dir}')

    # Build the set of stems to process
    if args.select:
        selected_stems = {s.strip() for s in args.select.split(',') if s.strip()}
    else:
        selected_stems = None  # all images

    for fname in sorted(os.listdir(args.image_dir)):
        ext  = os.path.splitext(fname)[1]
        stem = os.path.splitext(fname)[0]

        if ext not in IMAGE_EXTENSIONS:
            continue
        if selected_stems is not None and stem not in selected_stems:
            continue

        img_path = os.path.join(args.image_dir, fname)
        res_path = os.path.join(args.result_dir, f'res_{stem}.txt')

        if not os.path.exists(res_path):
            print(f'  [SKIP] No CRAFT result for {fname} '
                  f'(expected {res_path})')
            continue

        pairs.append((img_path, res_path))

    # Warn about --select stems that were never matched
    if selected_stems is not None:
        found_stems = {os.path.splitext(os.path.basename(p))[0] for p, _ in pairs}
        for m in sorted(selected_stems - found_stems):
            print(f'  [WARN] --select stem "{m}" not found in {args.image_dir}')

    return pairs


# ── Per-image pipeline ────────────────────────────────────────────────────────

def process_one(img_path, res_path, args, ocr_reader):
    """Run the full pipeline on a single image. Returns True on success."""
    img_stem = os.path.splitext(os.path.basename(img_path))[0]
    sep = '─' * 60
    print(f'\n{sep}')
    print(f'  Image : {img_path}')
    print(f'  Result: {res_path}')
    print(sep)

    # Load
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f'  [ERROR] Cannot read image — skipping.')
        return False
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f'  Size: {img.shape[1]}w x {img.shape[0]}h px')

    # ── Phase 1: Detection ────────────────────────────────────────────────────
    print('\n── Phase 1: Detection ──')
    raw_boxes = load_craft_boxes(res_path)
    boxes     = deduplicate_craft_boxes(raw_boxes)
    print(f'  CRAFT boxes: {len(raw_boxes)} raw -> {len(boxes)} deduped')

    areas_raw    = build_text_areas(boxes)
    areas_merged = merge_overlapping_areas(areas_raw)
    valid_areas, noise_areas = purify_areas(areas_merged, img.shape)
    print(f'  Areas: {len(areas_raw)} raw -> {len(areas_merged)} merged '
          f'-> {len(valid_areas)} valid, {len(noise_areas)} noise')

    # ── Phase 1: OCR ──────────────────────────────────────────────────────────
    print('\n── Phase 1: OCR ──')
    processed_raw = []
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
        processed_raw.append({
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

    processed_areas = deduplicate_ocr_across_areas(processed_raw)
    print(f'  Telugu areas: {len(processed_raw)} -> '
          f'{len(processed_areas)} after cross-area dedup')

    if not processed_areas:
        print('  No Telugu text detected — skipping translation & inpainting.')
        return True

    # ── Phase 2: Translation ──────────────────────────────────────────────────
    if not args.skip_translate:
        if not args.api_key:
            print('  [WARN] --api-key not set; skipping translation. '
                  'Use --skip-translate to suppress this warning.')
        else:
            print('\n── Phase 2: Translation ──')
            image_type = detect_image_type(processed_areas, args.api_key)
            print(f'  Image type (auto): {image_type}')

            print('  Normalizing OCR text...')
            for area in processed_areas:
                raw = area.get('full_text', '').strip()
                area['corrected_telugu'] = (
                    normalize_telugu_ocr(raw, args.api_key) if raw else ''
                )
                time.sleep(0.3)

            print('  Translating all areas in one call...')
            corrected     = [a.get('corrected_telugu', '') for a in processed_areas]
            tamil_results = translate_areas(corrected, image_type, args.api_key)
            for i, area in enumerate(processed_areas):
                area['tamil_translation'] = tamil_results[i]

            def _strip(t):
                return re.sub(r'<think>.*?</think>', '', t or '',
                              flags=re.DOTALL).strip()

            save_data = [{
                'area_index':        i,
                'area_bbox':         list(area['area_bbox']),
                'raw_ocr':           area.get('full_text', ''),
                'corrected_telugu':  _strip(area.get('corrected_telugu', '')),
                'tamil_translation': area.get('tamil_translation', ''),
            } for i, area in enumerate(processed_areas)]

            json_path = os.path.join(args.output,
                                     f'{img_stem}_translation_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            ok = sum(1 for d in save_data if d['tamil_translation'])
            print(f'  Saved: {json_path}  ({ok}/{len(save_data)} areas translated)')

    # ── Phase 2: Inpainting ───────────────────────────────────────────────────
    print('\n── Phase 2: Inpainting ──')
    inpainted = inpaint_all_areas(img, processed_areas)
    inpainted = inpaint_noise_boxes(inpainted, raw_boxes, processed_areas)

    out_path = os.path.join(args.output, f'{img_stem}_inpainted.jpg')
    cv2.imwrite(out_path, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
    print(f'  Saved: {out_path}')

    if args.show:
        visualize_inpainted(img, inpainted)

    print(f'  Done: {img_stem}')
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    pairs = collect_pairs(args)
    if not pairs:
        print('[ERROR] No images to process. '
              'Check --image-dir / --result-dir / --select.')
        sys.exit(1)

    total = len(pairs)
    print(f'\nImages to process: {total}')
    for img_path, _ in pairs:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f'  • {stem}')

    # Init EasyOCR once — shared across all images (expensive)
    use_gpu = not args.no_gpu
    print(f'\nInitialising EasyOCR (gpu={use_gpu})...')
    ocr_reader = easyocr.Reader(['te', 'en'], gpu=use_gpu)
    print('EasyOCR ready.')

    succeeded = failed = 0
    for img_path, res_path in pairs:
        try:
            ok = process_one(img_path, res_path, args, ocr_reader)
            if ok:
                succeeded += 1
            else:
                failed += 1
        except Exception as exc:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            print(f'\n  [ERROR] {stem}: {exc}')
            failed += 1

    print(f'\n{"=" * 60}')
    print(f'  Pipeline complete.')
    print(f'  Processed : {succeeded}/{total}')
    if failed:
        print(f'  Failed    : {failed}/{total}')
    print(f'  Outputs   : {args.output}/')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
