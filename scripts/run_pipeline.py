#!/usr/bin/env python3
"""
scripts/run_pipeline.py
------------------------
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
  --no-gpu            force CPU for PaddleOCR
  --show              show before/after plot for each image
  --skip-translate    skip Groq API calls, inpainting only (good for testing)

Outputs (in --output folder)
-----------------------------
  {stem}/
    inpainted.jpg
    metadata.json
    text_results.json
"""

import argparse
import json
import os
import re
import sys
import time

import cv2
from vtt import (
    load_craft_boxes,
    deduplicate_craft_boxes,
    build_text_areas,
    merge_overlapping_areas,
    purify_areas,
    area_bbox,
    generate_area_mask,
    create_ocr_reader,
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

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff',
    '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF',
}


def json_safe(value):
    """Convert numpy/OpenCV values into plain JSON-safe Python objects."""
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if k != 'mask'}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def area_to_metadata(area, area_index):
    """Build the persistent per-area OCR/translation record."""
    return {
        'area_index': area_index,
        'source_area_index': area.get('area_idx'),
        'area_bbox': list(area.get('area_bbox', [])),
        'area_quads': json_safe(area.get('area_quads', [])),
        'raw_text': area.get('full_text', ''),
        'corrected_telugu': area.get('corrected_telugu', ''),
        'tamil_translation': area.get('tamil_translation', ''),
        'telugu_words': json_safe(area.get('telugu_words', [])),
        'other_words': json_safe(area.get('other_words', [])),
        'raw_ocr': json_safe(area.get('raw_ocr', [])),
    }


def write_image_outputs(output_root, img_stem, img_path, res_path, img, inpainted,
                        raw_boxes, boxes, areas_raw, areas_merged,
                        valid_areas, noise_areas, processed_raw,
                        processed_areas, image_type, skip_translate):
    """Write each image's image output and metadata under output/<image_stem>/."""
    image_dir = os.path.join(output_root, img_stem)
    os.makedirs(image_dir, exist_ok=True)

    inpainted_path = os.path.join(image_dir, 'inpainted.jpg')
    cv2.imwrite(inpainted_path, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))

    text_results = [
        {
            'area_index': i,
            'area_bbox': list(area.get('area_bbox', [])),
            'raw_text': area.get('full_text', ''),
            'corrected_telugu': area.get('corrected_telugu', ''),
            'tamil_translation': area.get('tamil_translation', ''),
        }
        for i, area in enumerate(processed_areas)
    ]

    metadata = {
        'image_stem': img_stem,
        'source_image': img_path,
        'craft_result': res_path,
        'image_width': int(img.shape[1]),
        'image_height': int(img.shape[0]),
        'skip_translate': bool(skip_translate),
        'image_type': image_type,
        'counts': {
            'craft_boxes_raw': len(raw_boxes),
            'craft_boxes_deduped': len(boxes),
            'areas_raw': len(areas_raw),
            'areas_merged': len(areas_merged),
            'areas_valid': len(valid_areas),
            'areas_noise': len(noise_areas),
            'telugu_areas_before_ocr_dedup': len(processed_raw),
            'telugu_areas_after_ocr_dedup': len(processed_areas),
        },
        'outputs': {
            'inpainted_image': inpainted_path,
            'metadata': os.path.join(image_dir, 'metadata.json'),
            'text_results': os.path.join(image_dir, 'text_results.json'),
        },
        'areas': [
            area_to_metadata(area, i)
            for i, area in enumerate(processed_areas)
        ],
    }

    metadata_path = os.path.join(image_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe(metadata), f, ensure_ascii=False, indent=2)

    text_results_path = os.path.join(image_dir, 'text_results.json')
    with open(text_results_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe(text_results), f, ensure_ascii=False, indent=2)

    return image_dir, inpainted_path, metadata_path, text_results_path


# -- Argument parsing ----------------------------------------------------------

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
                   help='Groq API key (required unless --skip-translate)')
    p.add_argument('--output',         default='output',
                   help='Output directory (default: output)')
    p.add_argument('--no-gpu',         action='store_true',
                   help='Force CPU for PaddleOCR')
    p.add_argument('--show',           action='store_true',
                   help='Show before/after plot for each image')
    p.add_argument('--skip-translate', action='store_true',
                   help='Skip Groq translation (inpainting only)')

    return p.parse_args()


# -- Image pair collection -----------------------------------------------------

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


# -- Per-image pipeline --------------------------------------------------------

def process_one(img_path, res_path, args, ocr_reader):
    """Run the full pipeline on a single image. Returns True on success."""
    img_stem = os.path.splitext(os.path.basename(img_path))[0]
    sep = '-' * 60
    print(f'\n{sep}')
    print(f'  Image : {img_path}')
    print(f'  Result: {res_path}')
    print(sep)

    # Load
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f'  [ERROR] Cannot read image - skipping.')
        return False
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f'  Size: {img.shape[1]}w x {img.shape[0]}h px')

    # -- Phase 1: Detection ----------------------------------------------------
    print('\n-- Phase 1: Detection --')
    raw_boxes = load_craft_boxes(res_path)
    boxes     = deduplicate_craft_boxes(raw_boxes)
    print(f'  CRAFT boxes: {len(raw_boxes)} raw -> {len(boxes)} deduped')

    areas_raw    = build_text_areas(boxes)
    areas_merged = merge_overlapping_areas(areas_raw)
    valid_areas, noise_areas = purify_areas(areas_merged, img.shape)
    print(f'  Areas: {len(areas_raw)} raw -> {len(areas_merged)} merged '
          f'-> {len(valid_areas)} valid, {len(noise_areas)} noise')

    # -- Phase 1: OCR ----------------------------------------------------------
    print('\n-- Phase 1: OCR --')
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
        print('  No Telugu text detected - saving unchanged image and metadata.')
        image_dir, out_path, metadata_path, text_results_path = write_image_outputs(
            output_root=args.output,
            img_stem=img_stem,
            img_path=img_path,
            res_path=res_path,
            img=img,
            inpainted=img,
            raw_boxes=raw_boxes,
            boxes=boxes,
            areas_raw=areas_raw,
            areas_merged=areas_merged,
            valid_areas=valid_areas,
            noise_areas=noise_areas,
            processed_raw=processed_raw,
            processed_areas=processed_areas,
            image_type='unknown',
            skip_translate=args.skip_translate or not bool(args.api_key),
        )
        print(f'  Saved folder : {image_dir}')
        print(f'  Saved image  : {out_path}')
        print(f'  Saved metadata: {metadata_path}')
        print(f'  Saved text   : {text_results_path}')
        return True

    image_type = 'unknown'

    # -- Phase 2: Translation --------------------------------------------------
    if not args.skip_translate:
        if not args.api_key:
            print('  [WARN] --api-key not set; skipping translation. '
                  'Use --skip-translate to suppress this warning.')
        else:
            print('\n-- Phase 2: Translation --')
            image_type = detect_image_type(processed_areas, args.api_key)
            print(f'  Image type (auto): {image_type}')

            print('  Normalizing OCR text...')
            for area in processed_areas:
                raw = area.get('full_text', '').strip()
                # The LLM prompt handles OCR correction; keep the pipeline stage explicit.
                area['corrected_telugu'] = (
                    normalize_telugu_ocr(raw, args.api_key) if raw else ''
                )
                # Small sleep to avoid hitting rate limits
                time.sleep(0.5) 

            print('  Translating all areas in one call...')
            corrected = [a.get('corrected_telugu', '') for a in processed_areas]
            tamil_results = translate_areas(corrected, image_type, args.api_key)
            
            # Map results back
            for i, area in enumerate(processed_areas):
                area['tamil_translation'] = tamil_results[i]
            ok = sum(1 for area in processed_areas if area.get('tamil_translation'))
            print(f'  Translated: {ok}/{len(processed_areas)} areas')
    else:
        for area in processed_areas:
            area['corrected_telugu'] = ''
            area['tamil_translation'] = ''

    # -- Phase 2: Inpainting ---------------------------------------------------
    print('\n-- Phase 2: Inpainting --')
    inpainted = inpaint_all_areas(img, processed_areas)
    inpainted = inpaint_noise_boxes(inpainted, raw_boxes, processed_areas)

    image_dir, out_path, metadata_path, text_results_path = write_image_outputs(
        output_root=args.output,
        img_stem=img_stem,
        img_path=img_path,
        res_path=res_path,
        img=img,
        inpainted=inpainted,
        raw_boxes=raw_boxes,
        boxes=boxes,
        areas_raw=areas_raw,
        areas_merged=areas_merged,
        valid_areas=valid_areas,
        noise_areas=noise_areas,
        processed_raw=processed_raw,
        processed_areas=processed_areas,
        image_type=image_type,
        skip_translate=args.skip_translate or not bool(args.api_key),
    )
    print(f'  Saved folder : {image_dir}')
    print(f'  Saved image  : {out_path}')
    print(f'  Saved metadata: {metadata_path}')
    print(f'  Saved text   : {text_results_path}')

    if args.show:
        visualize_inpainted(img, inpainted)

    print(f'  Done: {img_stem}')
    return True


# -- Entry point ---------------------------------------------------------------

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
        print(f'  - {stem}')

    # Init PaddleOCR once - shared across all images (expensive)
    use_gpu = not args.no_gpu
    print(f'\nInitialising PaddleOCR Telugu recognizer (gpu={use_gpu})...')
    ocr_reader = create_ocr_reader(use_gpu=use_gpu)
    print('PaddleOCR ready.')

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

