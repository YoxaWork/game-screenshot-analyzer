from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_color_palette(colors: list[tuple[int, int, int]]):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    rgb = np.array(colors, dtype=np.uint8)[None, :, :] / 255.0
    ax.imshow(rgb, aspect="auto")
    ax.set_xticks(range(len(colors)))
    ax.set_xticklabels([f"RGB{c}" for c in colors], rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_title("Dominant Color Palette")
    fig.tight_layout()
    return fig


def plot_brightness_heatmap(gray: np.ndarray):
    # Downsample to make the heatmap lightweight.
    small = cv2.resize(gray, (40, 24), interpolation=cv2.INTER_AREA)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.imshow(small, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Brightness Distribution")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_edge_heatmap(edges: np.ndarray):
    small = cv2.resize(edges, (80, 45), interpolation=cv2.INTER_AREA)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(small, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Edge / Detail Distribution")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def draw_ui_regions(
    image: Image.Image,
    regions: list[dict],
) -> np.ndarray:
    canvas = np.array(image.convert("RGB")).copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    for region in regions:
        x, y = region["x"], region["y"]
        w, h = region["w"], region["h"]

        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            (255, 180, 0),
            3,
        )
        cv2.putText(
            canvas,
            region["name"],
            (x + 8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
