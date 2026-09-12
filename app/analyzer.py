from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .image_metrics import (
    aspect_ratio,
    brightness,
    contrast,
    dominant_colors,
    edge_density,
    grayscale,
    pil_to_rgb_array,
    quality_score,
    saturation,
    sharpness,
)
from .ui_detector import detect_ui_regions


def analyze_image(image: Image.Image) -> dict:
    rgb = pil_to_rgb_array(image)
    gray = grayscale(rgb)

    h, w = gray.shape
    edge_density_value, edges = edge_density(gray)

    b = brightness(gray)
    c = contrast(gray)
    s = saturation(rgb)
    sh = sharpness(gray)

    colors = dominant_colors(rgb)
    regions, coverage = detect_ui_regions(rgb)

    metrics = {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio(w, h),
        "brightness": b,
        "contrast": c,
        "saturation": s,
        "sharpness": sh,
        "edge_density": edge_density_value,
        "quality_score": quality_score(b, c, sh, edge_density_value),
    }

    return {
        "metrics": metrics,
        "palette": colors,
        "gray": gray,
        "edges": edges,
        "ui": {
            "detected": len(regions) > 0,
            "coverage": coverage,
            "regions": regions,
        },
    }


def compare_images(image_a: Image.Image, image_b: Image.Image) -> dict:
    a = analyze_image(image_a)
    b = analyze_image(image_b)

    ma = a["metrics"]
    mb = b["metrics"]

    # Resize B to A for a simple structural difference calculation.
    arr_a = pil_to_rgb_array(image_a)
    arr_b = pil_to_rgb_array(image_b)
    arr_b = cv2.resize(arr_b, (arr_a.shape[1], arr_a.shape[0]))

    gray_a = grayscale(arr_a).astype(np.float32)
    gray_b = grayscale(arr_b).astype(np.float32)

    difference = np.mean(np.abs(gray_a - gray_b)) / 255.0 * 100.0
    similarity = max(0.0, 100.0 - difference)

    table = pd.DataFrame(
        [
            ["Brightness", f"{ma['brightness']:.1f}%", f"{mb['brightness']:.1f}%"],
            ["Contrast", f"{ma['contrast']:.1f}%", f"{mb['contrast']:.1f}%"],
            ["Saturation", f"{ma['saturation']:.1f}%", f"{mb['saturation']:.1f}%"],
            ["Sharpness", f"{ma['sharpness']:.1f}", f"{mb['sharpness']:.1f}"],
            ["Edge Density", f"{ma['edge_density']:.1f}%", f"{mb['edge_density']:.1f}%"],
            ["Quality Score", f"{ma['quality_score']:.0f}/100", f"{mb['quality_score']:.0f}/100"],
            ["UI Coverage", f"{a['ui']['coverage']:.1f}%", f"{b['ui']['coverage']:.1f}%"],
        ],
        columns=["Metric", "Screenshot A", "Screenshot B"],
    )

    return {
        "brightness_difference": abs(ma["brightness"] - mb["brightness"]),
        "difference_score": difference,
        "similarity": similarity,
        "table": table,
    }
