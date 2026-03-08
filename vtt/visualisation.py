"""
vtt/visualisation.py
────────────────────
Visualisation helpers for all pipeline stages.
"""

import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .detection import area_bbox
from .inpainting import build_stroke_mask_for_area


def show_craft_results(result_dir: str) -> None:
    """Display all CRAFT result images from result_dir."""
    import os
    for img_name in sorted(os.listdir(result_dir)):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            path = os.path.join(result_dir, img_name)
            img_v = cv2.imread(path)
            if img_v is not None:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(img_v, cv2.COLOR_BGR2RGB))
                plt.title(f'CRAFT Detection: {img_name}', fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.show()


def visualize_areas(img: np.ndarray,
                    valid_areas: list,
                    noise_areas: list | None = None,
                    title: str = 'Text Areas') -> None:
    """Draw bounding boxes for valid (coloured) and noise (red) areas."""
    vis = img.copy()
    random.seed(42)
    for idx, area in enumerate(valid_areas):
        color = tuple(random.randint(80, 255) for _ in range(3))
        x1, y1, x2, y2 = area_bbox(area)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, str(idx), (x1 + 3, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if noise_areas:
        for area in noise_areas:
            x1, y1, x2, y2 = area_bbox(area)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 50, 50), 1)
    plt.figure(figsize=(14, 10))
    plt.imshow(vis)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_final_areas(img: np.ndarray,
                           processed_areas: list[dict]) -> None:
    """Draw final processed areas with OCR word centres."""
    vis = img.copy()
    random.seed(99)
    for a in processed_areas:
        color = tuple(random.randint(80, 220) for _ in range(3))
        x1, y1, x2, y2 = a['area_bbox']
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"#{a['area_idx']}", (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        for w in a['sentence']:
            cx, cy = w['center']
            cv2.circle(vis, (int(cx), int(cy)), 3, (0, 255, 0), -1)
    plt.figure(figsize=(14, 10))
    plt.imshow(vis)
    plt.title('Final Processed Areas (clean, deduplicated)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualise_stroke_masks(img_rgb: np.ndarray,
                            processed_areas: list[dict],
                            max_areas: int = 4) -> None:
    """Overlay stroke masks in green on the image."""
    vis = img_rgb.copy()
    for area in processed_areas[:max_areas]:
        stroke_mask = build_stroke_mask_for_area(img_rgb, area)
        vis[stroke_mask == 255] = [0, 220, 80]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original', fontsize=13)
    axes[0].axis('off')
    axes[1].imshow(vis)
    axes[1].set_title('Stroke masks (green = will be erased)', fontsize=13)
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    print(f'Showing stroke masks for {min(len(processed_areas), max_areas)} areas')


def visualize_inpainted(original: np.ndarray,
                         inpainted: np.ndarray) -> None:
    """Side-by-side before/after comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(original)
    axes[0].set_title('BEFORE — Original Telugu', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(inpainted)
    axes[1].set_title('AFTER — Telugu Text Removed', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
