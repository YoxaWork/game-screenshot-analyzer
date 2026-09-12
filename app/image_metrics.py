from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def grayscale(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray) / 255.0 * 100.0)


def contrast(gray: np.ndarray) -> float:
    # Standard deviation is a simple global contrast proxy.
    return float(np.std(gray) / 128.0 * 100.0)


def saturation(image_rgb: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 1]) / 255.0 * 100.0)


def sharpness(gray: np.ndarray) -> float:
    # Variance of Laplacian is a common focus/sharpness heuristic.
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_density(gray: np.ndarray) -> tuple[float, np.ndarray]:
    edges = cv2.Canny(gray, 100, 200)
    density = float(np.count_nonzero(edges) / edges.size * 100.0)
    return density, edges


def aspect_ratio(width: int, height: int) -> str:
    ratio = width / height
    common = {
        16 / 9: "16:9",
        16 / 10: "16:10",
        4 / 3: "4:3",
        5 / 4: "5:4",
        21 / 9: "21:9",
        32 / 9: "32:9",
    }
    closest = min(common, key=lambda x: abs(x - ratio))
    return common[closest] if abs(closest - ratio) < 0.03 else f"{ratio:.2f}:1"


def dominant_colors(image_rgb: np.ndarray, k: int = 6) -> list[tuple[int, int, int]]:
    pixels = image_rgb.reshape((-1, 3)).astype(np.float32)

    # Downsample for speed.
    if len(pixels) > 50000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pixels), 50000, replace=False)
        pixels = pixels[indices]

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.5,
    )

    _, _, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    return [tuple(map(int, center)) for center in centers]


def quality_score(
    brightness_value: float,
    contrast_value: float,
    sharpness_value: float,
    edge_density_value: float,
) -> float:
    # Deliberately heuristic rather than a claim of perceptual quality.
    exposure = 100.0 - min(abs(brightness_value - 50.0) * 2.0, 100.0)
    contrast_score = min(max(contrast_value, 0.0), 100.0)
    sharpness_score = min(sharpness_value / 8.0, 100.0)
    edge_score = min(edge_density_value * 3.0, 100.0)

    score = (
        exposure * 0.25
        + contrast_score * 0.20
        + sharpness_score * 0.35
        + edge_score * 0.20
    )
    return float(np.clip(score, 0, 100))
