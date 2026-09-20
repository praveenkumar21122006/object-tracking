# Real-Time Object Tracking in Video Streams

High-performance multi-object tracking: **YOLOv8 + ByteTrack / SORT / Centroid + Kalman Filter + OpenCV** — webcam, video file, RTSP, browser UI.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org) [![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)

## Features
- **Detector**: YOLOv8n/s/m (auto-download, COCO 80 classes)
- **Trackers**: ByteTrack (default), SORT, Centroid + Hungarian + Kalman
- **Inputs**: webcam `0`, `video.mp4`, `rtsp://`, YouTube/HTTP
- **Browser**: Flask `web_app.py` — upload / sample / RTSP → tracked mp4 + live MJPEG at `http://127.0.0.1:5000`
- **CLI**: `app.py` with `--show`/`--no-show`/`--save`, trails, FPS HUD, mp4 export

## Quick Start (CLI)
```bash
pip install -r requirements.txt
python app.py --source 0 --tracker bytetrack --conf 0.35          # webcam
python app.py --source "rtsp://user:pass@ip/stream" --save --no-show
python app.py --source video.mp4 --tracker sort --save --output output/tracked.mp4
python demo.py   # synthetic tracker sanity check (60 frames)
```

## Browser
```bash
pip install -r requirements.txt
python web_app.py
# open http://127.0.0.1:5000 → tracker/conf/model → Upload or Sample (bus_pan.mp4) → Run → Download
# APIs: POST /api/track (multipart), POST /api/track_sample, POST /api/track_url, GET /stream
```

## Config
Edit `config.yaml`:
```yaml
detection: {model: "yolov8n.pt", conf_threshold: 0.35, classes: null}
tracking: {tracker_type: "bytetrack", track_buffer: 30, match_thresh: 0.3}
```

## Structure
```
app.py                    # CLI: capture → detect → track → visualize
web_app.py                # Flask browser UI + /api/track + MJPEG /stream
tracker/detector.py       # YOLODetector (dummy torchvision fallback)
tracker/object_tracker.py # ByteTracker/SORT/Centroid
utils/visualization.py    # boxes, trails, HUD
config.yaml
```

## Deploy
- Docker: `docker build -t tracking . && docker run -p 5000:5000 tracking` (see Dockerfile)
- Render/HF: set start command `python web_app.py`, port 5000
- Gunicorn: `gunicorn -w 2 -b 0.0.0.0:$PORT web_app:app`

## License MIT
