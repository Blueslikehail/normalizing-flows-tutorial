"""
YOLO Custom Training — Flask Backend
=====================================
Exposes a REST API consumed by the browser frontend:

  GET  /                    Serve the main HTML UI
  POST /infer               Run YOLOv8 inference on a JPEG frame
  POST /save_annotation     Save annotated frame + YOLO-format labels
  GET  /annotations_info    Current dataset stats
  POST /clear_data          Wipe all saved annotations
  POST /train               Start background fine-tuning
  GET  /train_status        Poll training progress
"""

import json
import base64
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
IMAGES_DIR  = DATA_DIR / "images" / "train"
LABELS_DIR  = DATA_DIR / "labels" / "train"
RUNS_DIR    = BASE_DIR / "runs"

for _d in [IMAGES_DIR, LABELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global state (thread-safe enough for a tutorial; Python GIL protects reads)
# ---------------------------------------------------------------------------
base_model   = YOLO("yolov8n.pt")   # downloaded on first run (~6 MB)
custom_model = None                  # set after a successful fine-tune
custom_classes: list[str] = []

training_status: dict = {
    "running":       False,
    "progress":      0,
    "epoch":         0,
    "total_epochs":  0,
    "message":       "Ready",
    "error":         None,
}

# ---------------------------------------------------------------------------
# Routes — static
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ---------------------------------------------------------------------------
# Routes — inference
# ---------------------------------------------------------------------------

@app.post("/infer")
def infer():
    """
    Accepts: { image: "<data-url>" }
    Returns: { detections: [...], model: "yolov8n"|"custom", img_w, img_h }

    Each detection: { x1, y1, x2, y2 (normalised 0-1), confidence, label }
    """
    payload = request.get_json(force=True)
    img = _decode_image(payload.get("image", ""))
    if img is None:
        return jsonify(error="Could not decode image"), 400

    h, w = img.shape[:2]
    m = custom_model if custom_model is not None else base_model
    model_name = "custom" if custom_model is not None else "yolov8n"

    results = m.predict(img, verbose=False, conf=0.35)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append(
            dict(
                x1=x1 / w, y1=y1 / h, x2=x2 / w, y2=y2 / h,
                confidence=round(float(box.conf[0]), 3),
                label=m.names[int(box.cls[0])],
            )
        )

    return jsonify(detections=detections, model=model_name, img_w=w, img_h=h)


# ---------------------------------------------------------------------------
# Routes — annotation collection
# ---------------------------------------------------------------------------

@app.post("/save_annotation")
def save_annotation():
    """
    Accepts:
      {
        image: "<data-url>",
        annotations: [{ label, x_center, y_center, width, height }]   # normalised
      }
    Returns: { saved, filename, classes, total_annotations }
    """
    global custom_classes

    payload     = request.get_json(force=True)
    annotations = payload.get("annotations", [])
    if not annotations:
        return jsonify(error="No annotations provided"), 400

    img = _decode_image(payload.get("image", ""))
    if img is None:
        return jsonify(error="Could not decode image"), 400

    # Update class list (preserve insertion order)
    for ann in annotations:
        label = ann.get("label", "").strip()
        if label and label not in custom_classes:
            custom_classes.append(label)

    # Persist image
    ts       = int(time.time() * 1000)
    img_path = IMAGES_DIR / f"frame_{ts}.jpg"
    lbl_path = LABELS_DIR / f"frame_{ts}.txt"
    cv2.imwrite(str(img_path), img)

    # Write YOLO-format label file: class_id xc yc w h  (all normalised)
    with open(lbl_path, "w") as f:
        for ann in annotations:
            label = ann.get("label", "").strip()
            if label not in custom_classes:
                continue
            cls_id = custom_classes.index(label)
            f.write(
                f"{cls_id} "
                f"{ann['x_center']:.6f} {ann['y_center']:.6f} "
                f"{ann['width']:.6f} {ann['height']:.6f}\n"
            )

    total = len(list(IMAGES_DIR.glob("*.jpg")))
    return jsonify(saved=True, filename=img_path.name,
                   classes=custom_classes, total_annotations=total)


@app.get("/annotations_info")
def annotations_info():
    total = len(list(IMAGES_DIR.glob("*.jpg")))
    return jsonify(total=total, classes=custom_classes)


@app.post("/clear_data")
def clear_data():
    global custom_classes, custom_model
    for f in IMAGES_DIR.glob("*"):
        f.unlink(missing_ok=True)
    for f in LABELS_DIR.glob("*"):
        f.unlink(missing_ok=True)
    yaml = DATA_DIR / "dataset.yaml"
    if yaml.exists():
        yaml.unlink()
    custom_classes = []
    custom_model   = None
    return jsonify(cleared=True)


# ---------------------------------------------------------------------------
# Routes — training
# ---------------------------------------------------------------------------

@app.post("/train")
def start_training():
    global training_status

    if training_status["running"]:
        return jsonify(error="Training already in progress"), 400

    total = len(list(IMAGES_DIR.glob("*.jpg")))
    if total < 5:
        return jsonify(error=f"Need at least 5 annotated frames (have {total})"), 400
    if not custom_classes:
        return jsonify(error="No custom classes defined yet"), 400

    payload = request.get_json(force=True) or {}
    epochs  = max(1, min(int(payload.get("epochs", 20)), 100))

    training_status = dict(running=True, progress=0, epoch=0,
                           total_epochs=epochs, message="Initialising…", error=None)

    t = threading.Thread(target=_run_training, args=(epochs,), daemon=True)
    t.start()
    return jsonify(started=True, epochs=epochs)


@app.get("/train_status")
def train_status():
    return jsonify(training_status)


# ---------------------------------------------------------------------------
# Background training
# ---------------------------------------------------------------------------

def _run_training(epochs: int):
    """Fine-tune YOLOv8n on the collected custom dataset."""
    global custom_model, training_status, custom_classes

    try:
        # --- Write dataset.yaml -----------------------------------------------
        yaml_path = DATA_DIR / "dataset.yaml"
        names_yaml = "\n".join(f"  - {c}" for c in custom_classes)
        yaml_path.write_text(
            f"path: {DATA_DIR.absolute()}\n"
            f"train: images/train\n"
            f"val:   images/train\n\n"
            f"nc: {len(custom_classes)}\n"
            f"names:\n{names_yaml}\n"
        )

        training_status["message"] = "Loading base model…"
        m = YOLO("yolov8n.pt")

        # --- Epoch progress callback ------------------------------------------
        def _on_epoch_end(trainer):
            ep = trainer.epoch + 1
            training_status.update(
                progress=int(ep / epochs * 100),
                epoch=ep,
                message=f"Epoch {ep}/{epochs}",
            )

        m.add_callback("on_train_epoch_end", _on_epoch_end)
        training_status["message"] = "Training…"

        # Clean up previous run so ultralytics doesn't append index suffixes
        run_dir = RUNS_DIR / "train" / "custom_yolo"
        if run_dir.exists():
            shutil.rmtree(run_dir)

        m.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=416,          # smaller than default 640 → faster on CPU
            batch=4,
            lr0=1e-3,
            project=str(RUNS_DIR / "train"),
            name="custom_yolo",
            exist_ok=True,
            verbose=False,
            plots=False,
        )

        best_weights = RUNS_DIR / "train" / "custom_yolo" / "weights" / "best.pt"
        if best_weights.exists():
            custom_model = YOLO(str(best_weights))
            training_status.update(
                running=False, progress=100,
                message=(
                    f"Done! Custom model active. "
                    f"Classes: {', '.join(custom_classes)}"
                ),
            )
        else:
            training_status.update(
                running=False,
                error="Training finished but best.pt not found.",
                message="Error: weights file missing.",
            )

    except Exception as exc:  # noqa: BLE001
        training_status.update(
            running=False,
            error=str(exc),
            message=f"Training failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(data_url: str):
    """Decode a base64 data-URL into an OpenCV BGR image."""
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw   = base64.b64decode(data_url)
        nparr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 60)
    print("  YOLO Custom Object Detector")
    print("  Open http://localhost:5000 in your browser")
    print("═" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
