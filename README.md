# Object-Detection-System-Identifies-Objects-in-Real-Time-

# Object Detection System – Real-Time Video
### Project by: Mohd Aditya mall| Python · OpenCV · YOLOv8 / COCO-SSD / Haar Cascades

---

## Quick Start (zero setup)

```bash
pip install opencv-python numpy
python object_detection.py
```

The system launches immediately using built-in **Haar Cascades** — no model download needed.
It detects: **faces, profiles, bodies, upper body, lower body, eyes, smiles, cats**.

---

## Upgrade to COCO-SSD / YOLOv8 (80-class detection)

### Option A — YOLOv8 (recommended, best accuracy)
```bash
pip install ultralytics
# Download model (6 MB) from https://github.com/ultralytics/assets/releases
# Place yolov8n.pt in the same folder as object_detection.py
python object_detection.py
```

### Option B — MobileNet-SSD (no extra pip install)
Download these two files into the project folder:
- `deploy.prototxt` → https://github.com/chuanqi305/MobileNet-SSD
- `mobilenet_iter_73000.caffemodel` → same repo

```bash
python object_detection.py
```

---

## Command-Line Options

```bash
python object_detection.py --help

  --cam        INT     Camera index (default: 0)
  --conf       FLOAT   Confidence threshold 0.0–1.0 (default: 0.45)
  --engine     STR     Force engine: "YOLOv8" | "MobileNet-SSD" | "Haar Cascades"
  --model-dir  PATH    Folder containing .pt / .caffemodel files
```

### Examples
```bash
# Use camera 1 instead of default
python object_detection.py --cam 1

# Lower confidence to detect more objects
python object_detection.py --conf 0.30

# Force Haar Cascades even if YOLO is available
python object_detection.py --engine "Haar Cascades"

# Point to a different model folder
python object_detection.py --model-dir /models
```

---

## Keyboard Controls (in the detection window)

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `S` | Save snapshot (saved to `snapshots/` folder) |
| `P` | Pause / Resume detection |
| `+` | Raise confidence threshold by 5% |
| `-` | Lower confidence threshold by 5% |
| `F` | Toggle FPS / HUD display |
| `H` | Toggle help overlay |
| `1` | Switch to YOLOv8 (if available) |
| `2` | Switch to MobileNet-SSD (if available) |
| `3` | Switch to Haar Cascades |

---

## System Modules (per project report Chapter 6)

| Module | File / Class | Description |
|--------|-------------|-------------|
| 1. Camera Module | `CameraModule` | WebRTC-style webcam via OpenCV |
| 2. Video Display Module | `cv2.imshow` | Real-time mirrored display |
| 3. Model Loader Module | `ModelLoader` | Auto-selects best engine |
| 4. Pre-processing Module | `CameraModule.preprocess()` | Resize, normalize, grayscale |
| 5. Object Detection Module | `DetectionModule` | Multi-engine detection |
| 6. Rendering Module | `RenderingModule` | Boxes, labels, corner ticks |
| 7. Control Panel Module | keyboard handler in `run()` | Start/Stop/Snapshot/Conf |
| 8. Performance Monitor | `PerformanceMonitor` | FPS, latency, frame count |

---

## Detection Engines

| Engine | Classes | Accuracy | Speed | Requires |
|--------|---------|----------|-------|----------|
| YOLOv8n | 80 COCO | ★★★★★ | ★★★★ | `yolov8n.pt` |
| MobileNet-SSD | 21 COCO | ★★★★ | ★★★★★ | `.caffemodel` |
| Haar Cascades | 8 types | ★★★ | ★★★★★ | **Built-in ✔** |

---

## Project Report Reference

This implementation covers all chapters of the project report:
- **Ch 1** – Intro & objectives: browser-free, no server, webcam-based
- **Ch 3** – Architecture: modular pipeline Camera→Pre-process→Detect→Render
- **Ch 4** – Implementation: all code in `object_detection.py`
- **Ch 5** – Testing: run with `--conf` flag to test different thresholds
- **Ch 6** – All 8 modules implemented as Python classes
- **Ch 7** – Tech stack: Python, OpenCV, NumPy, (optional) Ultralytics/TF
