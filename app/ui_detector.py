from __future__ import annotations

import cv2
import numpy as np


def detect_ui_regions(
    image_rgb: np.ndarray,
) -> tuple[list[dict], float]:
    """Estimate likely HUD/UI regions using simple CV heuristics.

    This intentionally avoids game-specific assumptions and ML models.
    The detector looks for:
    - strong edge/text-like activity,
    - bright or highly contrasting areas,
    - concentration near common HUD zones.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    edges = cv2.Canny(gray, 120, 220)
    kernel = np.ones((5, 5), np.uint8)
    edge_map = cv2.dilate(edges, kernel, iterations=1)

    # Candidate zones: corners + top/bottom strips.
    zones = [
        ("top-left", 0.00, 0.00, 0.32, 0.22),
        ("top-right", 0.68, 0.00, 0.32, 0.22),
        ("bottom-left", 0.00, 0.78, 0.38, 0.22),
        ("bottom-right", 0.62, 0.78, 0.38, 0.22),
        ("top-center", 0.25, 0.00, 0.50, 0.14),
        ("bottom-center", 0.25, 0.86, 0.50, 0.14),
    ]

    regions = []
    coverage = 0.0

    for name, x0, y0, x1, y1 in zones:
        xa, ya = int(x0 * w), int(y0 * h)
        xb, yb = int(x1 * w), int(y1 * h)

        zone_edges = edge_map[ya:yb, xa:xb]
        if zone_edges.size == 0:
            continue

        density = np.count_nonzero(zone_edges) / zone_edges.size

        # Threshold is intentionally conservative to reduce false positives.
        if density > 0.035:
            area = (xb - xa) * (yb - ya)
            regions.append(
                {
                    "name": name,
                    "x": xa,
                    "y": ya,
                    "w": xb - xa,
                    "h": yb - ya,
                    "density": float(density),
                    "area": area,
                }
            )
            coverage += area / (w * h) * 100.0

    # Cap the estimate because overlapping candidate zones can overcount.
    coverage = min(coverage, 100.0)

    return regions, coverage
