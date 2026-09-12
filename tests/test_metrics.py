import numpy as np
from PIL import Image

from app.analyzer import analyze_image


def test_analyzer_returns_expected_fields():
    image = Image.fromarray(np.full((720, 1280, 3), 128, dtype=np.uint8))
    result = analyze_image(image)

    assert result["metrics"]["width"] == 1280
    assert result["metrics"]["height"] == 720
    assert "quality_score" in result["metrics"]
    assert "ui" in result


def test_aspect_ratio_is_detected():
    image = Image.fromarray(np.zeros((720, 1280, 3), dtype=np.uint8))
    result = analyze_image(image)
    assert result["metrics"]["aspect_ratio"] == "16:9"
