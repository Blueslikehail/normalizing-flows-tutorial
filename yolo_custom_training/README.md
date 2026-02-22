# YOLO Custom Object Detector

An end-to-end interactive demo that lets you:

1. **Run real-time inference** — YOLOv8n detects the 80 COCO classes on your camera feed.
2. **Collect your own data** — draw bounding boxes on captured frames and label them.
3. **Fine-tune YOLOv8** — train the model on your custom labels directly from the browser UI.

---

## Architecture

```
Browser (HTML / Canvas / JS)              Flask server (Python)
────────────────────────────              ─────────────────────
Camera via getUserMedia()  ──POST /infer────►  YOLOv8 predict()
Draw bounding boxes        ──POST /save_annotation──► write YOLO .txt labels
Click "Start Fine-Tuning"  ──POST /train───►  ultralytics YOLO.train() [thread]
Poll progress              ──GET  /train_status──►  epoch / progress
```

---

## Quick Start

### 1. Install dependencies

```bash
cd yolo_custom_training
pip install -r requirements.txt
```

The first run also downloads `yolov8n.pt` (~6 MB) automatically.

### 2. Start the server

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Workflow

### Step A — Inference (default mode)

- The camera feed runs live at ~2 FPS (throttled to keep the UI responsive).
- YOLO draws labelled bounding boxes over detected objects.
- The **model badge** in the top-right corner shows which model is active.

### Step B — Annotate

1. Click **Annotate** tab.
2. Click **Capture Frame** to freeze the current camera frame.
3. Type a label in the input box (`"hand"`, `"mug"`, …) and click **Set**.
4. Click and drag on the image to draw a bounding box — repeat for every object.
5. Click **Save Frame**. The app immediately captures the next frame.
6. Aim for **≥ 10 frames per class** with varied backgrounds and angles.

### Step C — Fine-Tune

1. After collecting frames, switch back to the **Dataset** card to verify counts.
2. Adjust the **Epochs** slider (5–50). More epochs = more training time.
3. Click **Start Fine-Tuning**.
4. A progress bar tracks each epoch. Training runs in a background thread.
5. When complete, the **Custom Model** badge turns green and inference switches
   automatically to your new model.

---

## Data Format

Labels are stored in standard **YOLO format**:

```
<class_id>  <x_center>  <y_center>  <width>  <height>
```

All coordinates are **normalised to [0, 1]** relative to the image dimensions.

```
data/
├── images/train/      # JPEG frames
├── labels/train/      # .txt label files (one per image)
└── dataset.yaml       # Generated automatically before training
```

Example label line for a 200×150 px box in a 640×480 frame:

```
0  0.4688  0.3125  0.3125  0.3125
```

---

## Key Concepts

### Transfer Learning / Fine-Tuning

We start from `yolov8n.pt`, which was pre-trained on the 80-class COCO dataset
(~118k images). Fine-tuning re-trains the detection head on your small custom
dataset. This works well because the backbone already knows how to extract
low-level features (edges, textures); it only needs to learn *which* feature
combinations correspond to your new classes.

> **Note:** After fine-tuning, the model detects **only** your custom classes —
> COCO classes are replaced. This is intentional for clarity; in production you
> would merge datasets to keep both.

### Hyperparameter Choices

| Parameter | Value  | Reason |
|-----------|--------|--------|
| `imgsz`   | 416    | Smaller than default 640 → faster on CPU |
| `batch`   | 4      | Low memory usage for consumer GPUs/CPUs |
| `lr0`     | 1e-3   | Standard Adam learning rate for fine-tuning |
| `epochs`  | 5–50   | User-controlled via slider |

### Why YOLOv8n?

The `n` (nano) variant has ~3.2M parameters and runs on CPU in real-time.
It's the right choice for an interactive tutorial; switch to `yolov8s` or
`yolov8m` for better accuracy on a GPU.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/` | Serve the web UI |
| POST | `/infer` | `{image: <data-url>}` → detections list |
| POST | `/save_annotation` | Save frame + YOLO labels |
| GET  | `/annotations_info` | Dataset stats |
| POST | `/clear_data` | Delete all saved data |
| POST | `/train` | Start fine-tuning `{epochs: N}` |
| GET  | `/train_status` | Poll epoch / progress / message |

---

## Requirements

- Python 3.8+
- A webcam (or built-in camera)
- `flask`, `ultralytics`, `opencv-python-headless`, `numpy`
- GPU optional — CPU is sufficient for `yolov8n` inference and short training runs
