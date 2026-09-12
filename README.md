# 🎮 Game Screenshot Analyzer

A computer vision toolkit for analyzing visual characteristics and UI composition in game screenshots.

## Features

- Resolution and aspect-ratio detection
- Brightness, contrast, saturation, and sharpness metrics
- Dominant-color extraction
- Edge-density analysis
- Heuristic HUD/UI detection
- HUD coverage estimation
- Visual quality score
- Brightness and edge heatmaps
- Screenshot comparison mode

> **Note:** HUD detection in this V1 is heuristic-based. It does not claim to understand a game's UI semantically. The goal is to provide a clean CV portfolio project that can be extended with OCR or object detection later.

## Tech stack

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
- Matplotlib

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
streamlit run main.py
```

Then open the local Streamlit URL shown in the terminal.

## Project structure

```text
game-screenshot-analyzer/
├── app/
│   ├── analyzer.py
│   ├── image_metrics.py
│   ├── ui_detector.py
│   └── visualizer.py
├── samples/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Roadmap

- [x] V1: image metrics
- [x] V1: heuristic UI/HUD detection
- [x] V1: visualizations
- [x] V1: screenshot comparison
- [ ] V2: OCR text detection
- [ ] V2: YOLO object detection
- [ ] V3: game-specific HUD classifiers
- [ ] V3: batch screenshot analysis
- [ ] V4: exportable HTML/JSON reports
