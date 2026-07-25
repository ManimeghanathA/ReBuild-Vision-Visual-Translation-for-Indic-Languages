#!/usr/bin/env python3
"""
Compare EasyOCR and PaddleOCR on the same CRAFT-derived Telugu text crops.

This script does not change the main ReBuild Vision pipeline. It creates a
repeatable OCR benchmark so we can decide whether PaddleOCR should replace
EasyOCR later.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from vtt import enhance_for_ocr, load_craft_boxes, rectify_quad  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF",
}


@dataclass
class CropRecord:
    image_stem: str
    image_path: Path
    result_path: Path
    crop_id: str
    quad_index: int
    bbox: tuple[int, int, int, int]
    quad: list[list[int]]
    crop_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark EasyOCR vs PaddleOCR on shared CRAFT crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", default="data/images",
                        help="Folder containing source images.")
    parser.add_argument("--result-dir", default="CRAFT-pytorch/result",
                        help="Folder containing CRAFT res_<stem>.txt files.")
    parser.add_argument("--select", default=None,
                        help="Comma-separated image stems to benchmark.")
    parser.add_argument("--output", default="output/ocr_benchmark",
                        help="Folder for crops, CSV, and JSON outputs.")
    parser.add_argument("--max-crops", type=int, default=0,
                        help="Maximum crops to process. 0 means no limit.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU for OCR engines where supported.")
    parser.add_argument("--engines", default="easyocr,paddleocr",
                        help="Comma-separated engines: easyocr,paddleocr.")
    parser.add_argument("--ground-truth", default=None,
                        help=("Optional TSV/CSV with crop_id,text columns. If omitted, "
                              "the script auto-loads ground_truth.tsv or a filled "
                              "ground_truth_template.tsv from the output folder."))
    parser.add_argument("--save-crops", action="store_true",
                        help="Save rectified crops for manual labeling.")
    parser.add_argument("--paddle-cache", default="models/paddleocr",
                        help="Shared PaddleX/PaddleOCR model cache folder.")
    return parser.parse_args()


def collect_image_pairs(image_dir: Path,
                        result_dir: Path,
                        select: str | None) -> list[tuple[Path, Path]]:
    selected = None
    if select:
        selected = {item.strip() for item in select.split(",") if item.strip()}

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix not in IMAGE_EXTENSIONS:
            continue
        stem = image_path.stem
        if selected is not None and stem not in selected:
            continue
        result_path = result_dir / f"res_{stem}.txt"
        if not result_path.exists():
            print(f"[WARN] Missing CRAFT result for {image_path.name}: {result_path}")
            continue
        pairs.append((image_path, result_path))
    return pairs


def safe_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def edit_distance(a: Any, b: Any) -> int:
    a = a or ""
    b = b or ""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float | None:
    if truth is None:
        return None
    truth = truth.strip()
    if not truth:
        return None
    return edit_distance(pred, truth) / max(len(truth), 1)


def wer(pred: str, truth: str) -> float | None:
    if truth is None:
        return None
    truth_words = truth.split()
    if not truth_words:
        return None
    pred_words = pred.split()
    return edit_distance(pred_words, truth_words) / max(len(truth_words), 1)


def load_ground_truth(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    gt_path = Path(path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    delimiter = "\t" if gt_path.suffix.lower() in {".tsv", ".txt"} else ","
    truth: dict[str, str] = {}
    with gt_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        required = {"crop_id", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Ground-truth file missing columns: {sorted(missing)}")
        for row in reader:
            crop_id = safe_text(row.get("crop_id", ""))
            text = safe_text(row.get("text", ""))
            if crop_id:
                truth[crop_id] = text
    return truth


def discover_ground_truth(output_dir: Path, explicit_path: str | None) -> tuple[dict[str, str], Path | None]:
    """Load explicit truth, or auto-load labels already present in the output folder."""
    if explicit_path:
        path = Path(explicit_path)
        return load_ground_truth(str(path)), path

    for candidate in (
        output_dir / "ground_truth.tsv",
        output_dir / "ground_truth.csv",
        output_dir / "ground_truth_template.tsv",
    ):
        if not candidate.exists():
            continue
        truth = load_ground_truth(str(candidate))
        filled_truth = {crop_id: text for crop_id, text in truth.items() if text}
        if filled_truth:
            return filled_truth, candidate

    return {}, None


def make_crops(pairs: list[tuple[Path, Path]],
               output_dir: Path,
               save_crops: bool,
               max_crops: int = 0) -> list[CropRecord]:
    crops_dir = output_dir / "crops"
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)
    records: list[CropRecord] = []

    for image_path, result_path in pairs:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes = load_craft_boxes(str(result_path))

        for idx, box in enumerate(boxes):
            crop_id = f"{image_path.stem}_q{idx:04d}"
            rectified, _, _ = rectify_quad(img_rgb, box["quad"], upscale=2.0)
            if rectified.size == 0:
                continue
            crop = enhance_for_ocr(rectified)
            crop_path = crops_dir / f"{crop_id}.png"
            if save_crops:
                cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

            x1, y1, x2, y2 = box["bbox"]
            records.append(CropRecord(
                image_stem=image_path.stem,
                image_path=image_path,
                result_path=result_path,
                crop_id=crop_id,
                quad_index=idx,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                quad=np.array(box["quad"]).astype(int).tolist(),
                crop_path=crop_path,
            ))

            if max_crops and len(records) >= max_crops:
                return records
    return records


def crop_image_for_record(record: CropRecord) -> np.ndarray:
    img_bgr = cv2.imread(str(record.image_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {record.image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    box = {"quad": np.array(record.quad, dtype=np.int32)}
    rectified, _, _ = rectify_quad(img_rgb, box["quad"], upscale=2.0)
    return enhance_for_ocr(rectified)


def run_easyocr(records: list[CropRecord], no_gpu: bool) -> list[dict[str, Any]]:
    import easyocr

    reader = easyocr.Reader(["te", "en"], gpu=not no_gpu)
    rows = []
    for record in records:
        crop = crop_image_for_record(record)
        started = time.time()
        raw = reader.readtext(crop, detail=1, paragraph=False)
        elapsed = time.time() - started
        parts = []
        confs = []
        for _, text, conf in raw:
            text = safe_text(text)
            if text:
                parts.append(text)
                confs.append(float(conf))
        rows.append({
            "engine": "easyocr",
            "crop_id": record.crop_id,
            "text": safe_text(" ".join(parts)),
            "confidence": sum(confs) / len(confs) if confs else None,
            "elapsed_sec": elapsed,
            "raw": raw,
        })
    return rows


def create_paddleocr(no_gpu: bool, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir.resolve()))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("MODELSCOPE_CACHE", str((cache_dir / "modelscope").resolve()))
    os.environ.setdefault("HF_HOME", str((cache_dir / "huggingface").resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((cache_dir / "huggingface" / "hub").resolve()))

    try:
        from paddleocr import TextRecognition
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install it in the vision environment "
            "before running the paddleocr engine."
        ) from exc

    device = "cpu"
    if not no_gpu:
        try:
            import paddle  # type: ignore

            if paddle.is_compiled_with_cuda():
                device = "gpu"
        except Exception:
            device = "gpu"

    init_attempts = [
        {"model_name": "te_PP-OCRv5_mobile_rec", "device": device},
        {"model_name": "te_PP-OCRv5_mobile_rec", "device": "cpu"},
        {"model_name": "te_PP-OCRv5_mobile_rec"},
        {"model_name": "PP-OCRv5_mobile_rec", "device": device},
        {"model_name": "PP-OCRv5_mobile_rec", "device": "cpu"},
    ]
    last_error: Exception | None = None
    for kwargs in init_attempts:
        try:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return TextRecognition(**kwargs)
        except Exception as exc:  # PaddleOCR changes kwargs between versions.
            last_error = exc
    raise RuntimeError(f"Could not initialize PaddleOCR: {last_error}")


def parse_paddle_result(result: Any) -> tuple[str, float | None]:
    texts: list[str] = []
    confs: list[float] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            for key in ("rec_text", "text"):
                if key in node and node[key]:
                    texts.append(safe_text(str(node[key])))
            for key in ("rec_score", "score", "confidence"):
                if key in node and node[key] is not None:
                    try:
                        confs.append(float(node[key]))
                    except (TypeError, ValueError):
                        pass
            for value in node.values():
                visit(value)
            return
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[1], (list, tuple)) and node[1]:
                if isinstance(node[1][0], str):
                    texts.append(safe_text(node[1][0]))
                    if len(node[1]) > 1:
                        try:
                            confs.append(float(node[1][1]))
                        except (TypeError, ValueError):
                            pass
            for item in node:
                visit(item)

    visit(result)
    texts = [t for t in texts if t]
    return safe_text(" ".join(texts)), (sum(confs) / len(confs) if confs else None)


def run_paddleocr(records: list[CropRecord],
                  no_gpu: bool,
                  cache_dir: Path) -> list[dict[str, Any]]:
    ocr = create_paddleocr(no_gpu=no_gpu, cache_dir=cache_dir)
    rows = []
    for record in records:
        crop = crop_image_for_record(record)
        started = time.time()
        try:
            result = ocr.predict(crop)
        except TypeError:
            fallback_path = record.crop_path
            if not fallback_path.exists():
                fallback_path = cache_dir / "tmp_crops" / f"{record.crop_id}.png"
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(fallback_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            result = ocr.predict(str(fallback_path))
        elapsed = time.time() - started
        text, confidence = parse_paddle_result(result)
        rows.append({
            "engine": "paddleocr",
            "crop_id": record.crop_id,
            "text": text,
            "confidence": confidence,
            "elapsed_sec": elapsed,
            "raw": result,
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_engine: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_engine.setdefault(row["engine"], []).append(row)

    summary = []
    for engine, engine_rows in sorted(by_engine.items()):
        with_truth = [r for r in engine_rows if r.get("cer") is not None]
        summary.append({
            "engine": engine,
            "crops": len(engine_rows),
            "non_empty": sum(1 for r in engine_rows if r.get("text")),
            "avg_confidence": avg([r.get("confidence") for r in engine_rows]),
            "avg_elapsed_sec": avg([r.get("elapsed_sec") for r in engine_rows]),
            "avg_cer": avg([r.get("cer") for r in with_truth]) if with_truth else None,
            "avg_wer": avg([r.get("wer") for r in with_truth]) if with_truth else None,
        })
    return summary


def avg(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def write_outputs(output_dir: Path,
                  records: list[CropRecord],
                  rows: list[dict[str, Any]],
                  summary: list[dict[str, Any]],
                  truth: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "crops_metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "crop_id", "image_stem", "image_path", "result_path",
            "quad_index", "bbox", "quad", "crop_path",
        ])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "crop_id": record.crop_id,
                "image_stem": record.image_stem,
                "image_path": str(record.image_path),
                "result_path": str(record.result_path),
                "quad_index": record.quad_index,
                "bbox": json.dumps(record.bbox),
                "quad": json.dumps(record.quad),
                "crop_path": str(record.crop_path),
            })

    template_path = output_dir / "ground_truth_template.tsv"
    with template_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_id", "text"], delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow({"crop_id": record.crop_id, "text": truth.get(record.crop_id, "")})

    csv_path = output_dir / "ocr_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "engine", "crop_id", "truth", "text", "confidence",
            "elapsed_sec", "cer", "wer",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "engine": row["engine"],
                "crop_id": row["crop_id"],
                "truth": row.get("truth"),
                "text": row.get("text"),
                "confidence": row.get("confidence"),
                "elapsed_sec": row.get("elapsed_sec"),
                "cer": row.get("cer"),
                "wer": row.get("wer"),
            })

    json_path = output_dir / "ocr_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)

    comparison = build_comparison(records, rows)
    comparison_path = output_dir / "ocr_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2, default=str)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved crop metadata: {metadata_path}")
    print(f"Saved GT template  : {template_path}")
    print(f"Saved OCR CSV      : {csv_path}")
    print(f"Saved OCR JSON     : {json_path}")
    print(f"Saved comparison   : {comparison_path}")
    print(f"Saved summary      : {summary_path}")


def build_comparison(records: list[CropRecord],
                     rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group EasyOCR and PaddleOCR results side by side per CRAFT crop."""
    by_crop: dict[str, dict[str, Any]] = {}
    for record in records:
        by_crop[record.crop_id] = {
            "crop_id": record.crop_id,
            "image_stem": record.image_stem,
            "source_image": str(record.image_path),
            "craft_result": str(record.result_path),
            "quad_index": record.quad_index,
            "bbox": list(record.bbox),
            "quad": record.quad,
            "truth": None,
            "easyocr": None,
            "paddleocr": None,
        }

    for row in rows:
        crop_id = row["crop_id"]
        if crop_id not in by_crop:
            continue
        by_crop[crop_id]["truth"] = row.get("truth")
        by_crop[crop_id][row["engine"]] = {
            "text": row.get("text"),
            "confidence": row.get("confidence"),
            "elapsed_sec": row.get("elapsed_sec"),
            "cer": row.get("cer"),
            "wer": row.get("wer"),
        }

    return [by_crop[record.crop_id] for record in records]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    engines = {engine.strip().lower() for engine in args.engines.split(",") if engine.strip()}

    pairs = collect_image_pairs(Path(args.image_dir), Path(args.result_dir), args.select)
    if not pairs:
        sys.exit("[ERROR] No image/result pairs found.")

    records = make_crops(
        pairs=pairs,
        output_dir=output_dir,
        save_crops=args.save_crops,
        max_crops=args.max_crops,
    )
    if not records:
        sys.exit("[ERROR] No crops generated from CRAFT results.")

    truth, truth_path = discover_ground_truth(output_dir, args.ground_truth)
    if truth_path:
        print(f"Loaded ground truth: {truth_path}")
    else:
        print("No filled ground truth found yet. CER/WER will stay empty until labels are added.")

    print(f"Benchmark crops: {len(records)}")
    rows: list[dict[str, Any]] = []

    if "easyocr" in engines:
        print("Running EasyOCR...")
        rows.extend(run_easyocr(records, no_gpu=args.no_gpu))

    if "paddleocr" in engines:
        print("Running PaddleOCR...")
        try:
            rows.extend(run_paddleocr(
                records,
                no_gpu=args.no_gpu,
                cache_dir=Path(args.paddle_cache),
            ))
        except RuntimeError as exc:
            print(f"[WARN] {exc}")

    for row in rows:
        gt = truth.get(row["crop_id"])
        row["truth"] = gt
        row["cer"] = cer(row.get("text", ""), gt) if gt is not None else None
        row["wer"] = wer(row.get("text", ""), gt) if gt is not None else None

    summary = summarize(rows)
    write_outputs(output_dir, records, rows, summary, truth)

    print("\nSummary:")
    for item in summary:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
