#!/usr/bin/env python3
"""
Real-Time Object Tracking in Video Streams
Usage:
  python app.py --source 0                          # webcam
  python app.py --source video.mp4 --save           # file
  python app.py --source rtsp://...                 # RTSP stream
  python app.py --source video.mp4 --tracker sort --conf 0.4
"""
import argparse
import cv2
import time
import yaml
import os
from pathlib import Path

from tracker.detector import YOLODetector
from tracker.object_tracker import build_tracker
from utils.visualization import Visualizer

def load_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}

def parse_args():
    cfg = load_config()
    det_cfg = cfg.get("detection",{})
    trk_cfg = cfg.get("tracking",{})
    vid_cfg = cfg.get("video",{})

    p = argparse.ArgumentParser(description="Real-Time Object Tracking")
    p.add_argument("--source", type=str, default=str(vid_cfg.get("source",0)), help="0, video path, rtsp, youtube")
    p.add_argument("--model", type=str, default=det_cfg.get("model","yolov8n.pt"))
    p.add_argument("--conf", type=float, default=det_cfg.get("conf_threshold",0.35))
    p.add_argument("--iou", type=float, default=det_cfg.get("iou_threshold",0.45))
    p.add_argument("--tracker", type=str, default=trk_cfg.get("tracker_type","bytetrack"), choices=["bytetrack","sort","centroid"])
    p.add_argument("--classes", type=int, nargs="*", default=det_cfg.get("classes",None), help="COCO class ids to keep, e.g. --classes 0 2")
    p.add_argument("--show", action="store_true", default=vid_cfg.get("show",True))
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--save", action="store_true", default=vid_cfg.get("save",False))
    p.add_argument("--output", type=str, default=vid_cfg.get("output","output/tracked_output.mp4"))
    p.add_argument("--no-trails", action="store_true", help="disable trail drawing")
    p.add_argument("--imgsz", type=int, default=det_cfg.get("imgsz",640))
    return p.parse_args()

def resolve_source(src):
    if src.isdigit():
        return int(src)
    return src

def main():
    args = parse_args()
    source = resolve_source(args.source)
    print(f"[Config] source={source} model={args.model} tracker={args.tracker} conf={args.conf}")

    detector = YOLODetector(model_path=args.model, conf=args.conf, iou=args.iou, classes=args.classes, imgsz=args.imgsz)

    tracker_cfg = dict(tracker_type=args.tracker, track_thresh=args.conf, track_buffer=30, match_thresh=0.8, min_hits=3)
    # tune per tracker
    if args.tracker=="sort":
        tracker_cfg.update(match_thresh=0.3, min_hits=3)
    elif args.tracker=="centroid":
        tracker_cfg.update(track_buffer=30)

    tracker = build_tracker(tracker_cfg)
    viz = Visualizer(draw_trails=not args.no_trails)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Cannot open source: {source}")
        return

    # output writer (lazy init after first frame size known)
    writer = None
    if args.save:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    fps_counter = 0
    t0 = time.time()
    fps = 0
    frame_idx = 0

    print("[Info] Started. Press 'q' to quit, 's' to toggle save screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Info] End of stream / no frame")
            break
        frame_idx += 1

        # detect + track
        start = time.time()
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        infer_ms = (time.time() - start)*1000

        # fps calc (EMA)
        fps_counter += 1
        if fps_counter % 10 == 0:
            fps = 10 / (time.time() - t0 + 1e-6)
            t0 = time.time()

        # draw
        frame = viz.draw(frame, tracks, detector, fps=fps)
        cv2.putText(frame, f"Infer {infer_ms:.1f}ms | Dets {len(detections)}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if args.save:
            if writer is None:
                h,w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # try to get source fps
                src_fps = cap.get(cv2.CAP_PROP_FPS)
                if src_fps <=1 or src_fps>60:
                    src_fps = 30
                writer = cv2.VideoWriter(args.output, fourcc, src_fps, (w,h))
                print(f"[Record] Saving to {args.output} @ {src_fps:.1f} fps {w}x{h}")
            writer.write(frame)

        if args.show:
            cv2.imshow("Real-Time Object Tracking - YOLO + "+args.tracker.upper(), frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                name = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(name, frame)
                print(f"[Saved] {name}")

        # print every 30 frames
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx}: {len(detections)} detections, {len(tracks)} tracks, {fps:.1f} FPS, {infer_ms:.1f} ms")

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"[Done] Processed {frame_idx} frames.")

if __name__ == "__main__":
    main()
