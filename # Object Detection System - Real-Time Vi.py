# Object Detection System - Real-Time Video
# Project by: Aditya Mall
# Stack: Python + OpenCV + YOLOv8 / MobileNet-SSD / Haar Cascades
#
# Modules (Chapter 6):
#   1. Camera Module           - webcam access via OpenCV
#   2. Video Display Module    - real-time frame rendering
#   3. Model Loader Module     - auto-selects best available engine
#   4. Pre-processing Module   - resize, normalize, grayscale
#   5. Object Detection Module - multi-class detection
#   6. Rendering Module        - bounding boxes, labels, confidence
#   7. Control Panel Module    - keyboard shortcuts
#   8. Performance Monitor     - FPS, latency, object count
#
# Detection Engines (auto-selected best to fallback):
#   [1] YOLOv8        - place yolov8n.pt in same folder (pip install ultralytics)
#   [2] MobileNet-SSD - place .caffemodel + .prototxt in same folder
#   [3] Haar Cascades - built-in, works with ZERO extra files
#
# Keyboard Controls:
#   Q / ESC   - Quit
#   S         - Save snapshot
#   P         - Pause / Resume
#   + / -     - Raise / Lower confidence threshold
#   F         - Toggle FPS display
#   H         - Toggle help overlay
#   1 / 2 / 3 - Switch detection engine

import cv2
import numpy as np
import time
import os
import sys
import argparse
import datetime

# ── Optional engine imports ───────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── COCO 80-class labels ──────────────────────────────────────────────────────
COCO_LABELS = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

np.random.seed(42)
CLASS_COLORS = np.random.randint(80, 255, size=(len(COCO_LABELS) + 10, 3)).tolist()

HAAR_COLORS = {
    "face":       (0,  229, 176),
    "profile":    (79, 110, 247),
    "body":       (255, 160,  64),
    "upper body": (196, 127, 255),
    "lower body": (64,  200, 255),
    "eye":        (255,  80, 160),
    "smile":      (128, 224,  64),
    "cat":        (255, 128,  64),
}


# =============================================================================
#  MODULE 3 - Model Loader
# =============================================================================
class ModelLoader:
    ENGINE_YOLO = "YOLOv8"
    ENGINE_SSD  = "MobileNet-SSD"
    ENGINE_HAAR = "Haar Cascades"

    def __init__(self, engine_hint=None, model_dir="."):
        self.engine    = None
        self.model     = None
        self.net       = None
        self.cascades  = {}
        self.model_dir = model_dir
        self._load(engine_hint)

    def _load(self, hint):
        order = [hint] if hint else [self.ENGINE_YOLO, self.ENGINE_SSD, self.ENGINE_HAAR]
        for eng in order:
            if eng == self.ENGINE_YOLO and self._try_yolo():
                return
            elif eng == self.ENGINE_SSD and self._try_ssd():
                return
            elif eng == self.ENGINE_HAAR:
                self._load_haar()
                return

    def _try_yolo(self):
        if not YOLO_AVAILABLE:
            return False
        for path in [os.path.join(self.model_dir, "yolov8n.pt"),
                     os.path.join(self.model_dir, "yolov8s.pt"),
                     "yolov8n.pt"]:
            if os.path.exists(path):
                try:
                    self.model  = _YOLO(path)
                    self.engine = self.ENGINE_YOLO
                    print("[ModelLoader] YOLOv8 loaded from", path)
                    return True
                except Exception as e:
                    print("[ModelLoader] YOLO failed:", e)
        return False

    def _try_ssd(self):
        proto = next((p for p in [
            os.path.join(self.model_dir, "deploy.prototxt"), "deploy.prototxt"
        ] if os.path.exists(p)), None)
        mdl = next((p for p in [
            os.path.join(self.model_dir, "mobilenet_iter_73000.caffemodel"),
            os.path.join(self.model_dir, "MobileNetSSD_deploy.caffemodel"),
            "mobilenet_iter_73000.caffemodel",
        ] if os.path.exists(p)), None)
        if proto and mdl:
            try:
                self.net    = cv2.dnn.readNetFromCaffe(proto, mdl)
                self.engine = self.ENGINE_SSD
                print("[ModelLoader] MobileNet-SSD loaded")
                return True
            except Exception as e:
                print("[ModelLoader] SSD failed:", e)
        return False

    def _load_haar(self):
        data = cv2.data.haarcascades
        specs = {
            "face":       "haarcascade_frontalface_default.xml",
            "profile":    "haarcascade_profileface.xml",
            "body":       "haarcascade_fullbody.xml",
            "upper body": "haarcascade_upperbody.xml",
            "lower body": "haarcascade_lowerbody.xml",
            "eye":        "haarcascade_eye.xml",
            "smile":      "haarcascade_smile.xml",
            "cat":        "haarcascade_frontalcatface.xml",
        }
        for name, fname in specs.items():
            path = os.path.join(data, fname)
            if os.path.exists(path):
                self.cascades[name] = cv2.CascadeClassifier(path)
        self.engine = self.ENGINE_HAAR
        print(f"[ModelLoader] Haar Cascades loaded ({len(self.cascades)} classifiers)")

    def switch_engine(self, name):
        self._load(name)


# =============================================================================
#  MODULES 1 + 4 - Camera & Pre-processing
# =============================================================================
class CameraModule:
    def __init__(self, cam_index=0, width=1280, height=720):
        # Try DirectShow backend first on Windows for faster, more reliable init
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Cannot open camera index {}. "
                "Make sure your webcam is connected and not in use by another app.".format(cam_index)
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm up: discard early frames so the first displayed frame is valid
        for _ in range(5):
            self.cap.read()

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened camera {cam_index} @ {actual_w}x{actual_h}")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return None
        return frame

    def preprocess(self, frame, target_size=None):
        # Module 4: optional resize for model input
        if target_size:
            frame = cv2.resize(frame, target_size)
        return frame

    def release(self):
        self.cap.release()
        print("[Camera] Released.")


# =============================================================================
#  MODULE 5 - Object Detection
# =============================================================================
class DetectionModule:
    def __init__(self, loader: ModelLoader, conf_threshold=0.45):
        self.loader    = loader
        self.threshold = conf_threshold

    def detect(self, frame):
        """Returns list of dicts: {class, confidence, bbox:(x,y,w,h)}"""
        engine = self.loader.engine
        if engine == ModelLoader.ENGINE_YOLO:
            return self._detect_yolo(frame)
        elif engine == ModelLoader.ENGINE_SSD:
            return self._detect_ssd(frame)
        else:
            return self._detect_haar(frame)

    def _detect_yolo(self, frame):
        results = self.loader.model(frame, verbose=False, conf=self.threshold)[0]
        dets = []
        for box in results.boxes:
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = results.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append({"class": label, "confidence": conf,
                         "bbox": (x1, y1, x2 - x1, y2 - y1)})
        return dets

    def _detect_ssd(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.loader.net.setInput(blob)
        raw = self.loader.net.forward()
        dets = []
        for i in range(raw.shape[2]):
            conf = float(raw[0, 0, i, 2])
            if conf < self.threshold:
                continue
            cid   = int(raw[0, 0, i, 1])
            label = COCO_LABELS[cid] if cid < len(COCO_LABELS) else f"cls_{cid}"
            box   = raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            dets.append({"class": label, "confidence": conf,
                         "bbox": (x1, y1, x2 - x1, y2 - y1)})
        return dets

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        dets = []
        params = {
            "face":       (1.1, 5),
            "profile":    (1.1, 4),
            "body":       (1.1, 3),
            "upper body": (1.1, 3),
            "lower body": (1.1, 3),
            "eye":        (1.1, 6),
            "smile":      (1.8, 20),
            "cat":        (1.1, 5),
        }
        for name, cascade in self.loader.cascades.items():
            sf, mn = params.get(name, (1.1, 4))
            rects  = cascade.detectMultiScale(gray, scaleFactor=sf,
                                              minNeighbors=mn, minSize=(40, 40))
            for (x, y, w, h) in (rects if len(rects) else []):
                dets.append({"class": name, "confidence": 0.90,
                             "bbox": (int(x), int(y), int(w), int(h))})
        return dets


# =============================================================================
#  MODULE 6 - Rendering
# =============================================================================
class RenderingModule:
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.55
    THICK      = 2

    def draw_detections(self, frame, detections, engine_name):
        for det in detections:
            label = det["class"]
            conf  = det["confidence"]
            x, y, w, h = det["bbox"]

            # Clamp coordinates to frame bounds
            fh, fw = frame.shape[:2]
            x = max(0, min(x, fw - 1))
            y = max(0, min(y, fh - 1))
            w = max(1, min(w, fw - x))
            h = max(1, min(h, fh - y))

            if engine_name == ModelLoader.ENGINE_HAAR:
                color = HAAR_COLORS.get(label, (200, 200, 200))
            else:
                idx   = COCO_LABELS.index(label) if label in COCO_LABELS else 0
                color = tuple(CLASS_COLORS[idx])

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.THICK)

            # Corner ticks
            t = 16
            for (px, py, dx, dy) in [
                (x,     y,     1,  1), (x + w, y,     -1,  1),
                (x,     y + h, 1, -1), (x + w, y + h, -1, -1),
            ]:
                cv2.line(frame, (px, py), (px + dx * t, py), color, 3)
                cv2.line(frame, (px, py), (px, py + dy * t), color, 3)

            # Label
            caption = f"{label.upper()}  {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(caption, self.FONT, self.FONT_SCALE, self.THICK)
            ly = y - 8 if y > th + 12 else y + h + th + 8
            cv2.rectangle(frame, (x, ly - th - 6), (x + tw + 10, ly + 4),
                          color, cv2.FILLED)
            cv2.putText(frame, caption, (x + 5, ly),
                        self.FONT, self.FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    def draw_hud(self, frame, fps, obj_count, frame_idx,
                 engine, threshold, paused, show_help):
        fh, fw = frame.shape[:2]

        # Top bar - draw on a copy then blend so video is still visible
        top_overlay = frame.copy()
        cv2.rectangle(top_overlay, (0, 0), (fw, 42), (15, 15, 20), cv2.FILLED)
        cv2.addWeighted(top_overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, "DETECTRON / REAL-TIME OBJECT DETECTION",
                    (10, 27), self.FONT, 0.56, (0, 229, 176), 1, cv2.LINE_AA)
        badge = f"[ {engine} ]"
        (bw, _), _ = cv2.getTextSize(badge, self.FONT, 0.5, 1)
        cv2.putText(frame, badge, (fw - bw - 12, 27),
                    self.FONT, 0.5, (79, 110, 247), 1, cv2.LINE_AA)

        # Bottom bar
        bot_overlay = frame.copy()
        cv2.rectangle(bot_overlay, (0, fh - 38), (fw, fh), (15, 15, 20), cv2.FILLED)
        cv2.addWeighted(bot_overlay, 0.65, frame, 0.35, 0, frame)

        paused_str = "  [PAUSED]" if paused else ""
        stats = (f"FPS: {fps:>5.1f}   OBJECTS: {obj_count:>3d}   "
                 f"FRAMES: {frame_idx:>6d}   CONF >= {threshold*100:.0f}%{paused_str}")
        cv2.putText(frame, stats, (12, fh - 14),
                    self.FONT, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        # Corner markers
        L, c = 28, (0, 229, 176)
        for (px, py, dx, dy) in [(0, 0, 1, 1), (fw-1, 0, -1, 1),
                                  (0, fh-1, 1, -1), (fw-1, fh-1, -1, -1)]:
            cv2.line(frame, (px, py), (px + dx * L, py), c, 2)
            cv2.line(frame, (px, py), (px, py + dy * L), c, 2)

        if show_help:
            self._draw_help(frame)

        return frame

    def _draw_help(self, frame):
        lines = [
            "KEYBOARD CONTROLS",
            "-------------------",
            "Q / ESC  - Quit",
            "S        - Save snapshot",
            "P        - Pause / Resume",
            "+  / -   - Raise / Lower confidence",
            "F        - Toggle FPS display",
            "H        - Toggle this help",
            "1        - Switch to YOLOv8",
            "2        - Switch to MobileNet-SSD",
            "3        - Switch to Haar Cascades",
        ]
        x0, y0  = 12, 50
        pad     = 10
        line_h  = 22
        box_w   = 320
        box_h   = len(lines) * line_h + pad * 2

        # Semi-transparent dark panel (copy region first, then blend)
        y1_clip = max(0, y0 - pad)
        y2_clip = min(frame.shape[0], y0 + box_h)
        x1_clip = max(0, x0 - pad)
        x2_clip = min(frame.shape[1], x0 + box_w)
        region  = frame[y1_clip:y2_clip, x1_clip:x2_clip].copy()
        cv2.rectangle(frame, (x1_clip, y1_clip), (x2_clip, y2_clip),
                      (12, 12, 20), cv2.FILLED)
        cv2.addWeighted(region, 0.2,
                        frame[y1_clip:y2_clip, x1_clip:x2_clip], 0.8,
                        0, frame[y1_clip:y2_clip, x1_clip:x2_clip])

        for i, line in enumerate(lines):
            if i == 0:
                color = (0, 229, 176)
            elif "---" in line:
                color = (55, 55, 65)
            else:
                color = (180, 180, 190)
            cv2.putText(frame, line, (x0, y0 + i * line_h + 14),
                        self.FONT, 0.46, color, 1, cv2.LINE_AA)


# =============================================================================
#  MODULE 8 - Performance Monitor
# =============================================================================
class PerformanceMonitor:
    def __init__(self, window=30):
        self.times  = []
        self.window = window
        self.last_t = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self.times.append(now - self.last_t)
        self.last_t = now
        if len(self.times) > self.window:
            self.times.pop(0)

    @property
    def fps(self):
        if not self.times:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))

    @property
    def latency_ms(self):
        return (self.times[-1] * 1000) if self.times else 0.0


# =============================================================================
#  Snapshot helper
# =============================================================================
def save_snapshot(frame, out_dir="snapshots"):
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"detection_{ts}.png")
    cv2.imwrite(path, frame)
    print(f"[Snapshot] Saved -> {path}")
    return path


# =============================================================================
#  MODULE 7 - Control Panel + Main Loop
# =============================================================================
def run(cam_index=0, model_dir=".", start_engine=None,
        conf=0.45, window_name="DETECTRON"):

    print("\n" + "=" * 62)
    print("  OBJECT DETECTION SYSTEM  -  Real-Time Video")
    print("  By: Aditya Mall |  Python + OpenCV")
    print("=" * 62)

    loader   = ModelLoader(engine_hint=start_engine, model_dir=model_dir)
    camera   = CameraModule(cam_index)
    detector = DetectionModule(loader, conf_threshold=conf)
    renderer = RenderingModule()
    monitor  = PerformanceMonitor()

    paused    = False
    show_help = True
    show_fps  = True
    frame_idx = 0
    last_dets = []
    consecutive_fails = 0

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    print("\n[System] Press H to toggle help. Q to quit.\n")

    while True:
        frame = camera.read_frame()

        # Handle failed reads gracefully
        if frame is None:
            consecutive_fails += 1
            if consecutive_fails > 30:
                print("[Error] Camera stopped providing frames. Exiting.")
                break
            if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                break
            continue
        consecutive_fails = 0

        # Mirror frame (natural selfie view)
        frame = cv2.flip(frame, 1)

        if not paused:
            frame_idx += 1
            dets      = detector.detect(frame)
            last_dets = dets
            monitor.tick()
        else:
            dets = last_dets

        # Draw detections onto the real video frame
        frame = renderer.draw_detections(frame, dets, loader.engine)

        # Draw HUD on top
        if show_fps:
            frame = renderer.draw_hud(
                frame,
                fps       = monitor.fps,
                obj_count = len(dets),
                frame_idx = frame_idx,
                engine    = loader.engine,
                threshold = detector.threshold,
                paused    = paused,
                show_help = show_help,
            )

        cv2.imshow(window_name, frame)

        # Keyboard handler
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            save_snapshot(frame)
        elif key == ord('p'):
            paused = not paused
            print(f"[Control] {'Paused' if paused else 'Resumed'}")
        elif key in (ord('+'), 43):
            detector.threshold = min(0.95, detector.threshold + 0.05)
            print(f"[Control] Confidence: {detector.threshold*100:.0f}%")
        elif key in (ord('-'), 45):
            detector.threshold = max(0.10, detector.threshold - 0.05)
            print(f"[Control] Confidence: {detector.threshold*100:.0f}%")
        elif key == ord('f'):
            show_fps = not show_fps
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('1'):
            loader.switch_engine(ModelLoader.ENGINE_YOLO)
            print(f"[Control] Switched to {loader.engine}")
        elif key == ord('2'):
            loader.switch_engine(ModelLoader.ENGINE_SSD)
            print(f"[Control] Switched to {loader.engine}")
        elif key == ord('3'):
            loader.switch_engine(ModelLoader.ENGINE_HAAR)
            print(f"[Control] Switched to {loader.engine}")

    camera.release()
    cv2.destroyAllWindows()
    print(f"\n[System] Done. Frames processed: {frame_idx}")


# =============================================================================
#  Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection System - Mohd Aadil"
    )
    parser.add_argument("--cam",       type=int,   default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--conf",      type=float, default=0.45,
                        help="Confidence threshold 0.0-1.0 (default: 0.45)")
    parser.add_argument("--engine",    type=str,   default=None,
                        choices=["YOLOv8", "MobileNet-SSD", "Haar Cascades"],
                        help="Force a specific engine")
    parser.add_argument("--model-dir", type=str,   default=".",
                        help="Folder with model files")
    args = parser.parse_args()

    try:
        run(cam_index    = args.cam,
            model_dir    = args.model_dir,
            start_engine = args.engine,
            conf         = args.conf)
    except RuntimeError as e:
        print(f"\n[Fatal] {e}")
        print("\nTroubleshooting:")
        print("  - Try:  python object_detection.py --cam 1")
        print("  - Close any other app using your webcam (Zoom, Teams, OBS, etc.)")
        print("  - On Windows: Settings > Privacy > Camera > allow app access")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[System] Interrupted.")
        cv2.destroyAllWindows()
