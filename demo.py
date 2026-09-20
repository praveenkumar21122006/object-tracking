"""Quick sanity test: synthetic moving boxes to verify tracker logic without camera"""
import numpy as np
import cv2
from tracker.object_tracker import build_tracker

# fake detections: two objects moving diagonally
tracker = build_tracker(dict(tracker_type="bytetrack", track_thresh=0.5, match_thresh=0.8, track_buffer=30))

for f in range(60):
    dets = [
        [100+f*2, 100+f*1, 150+f*2, 180+f*1, 0.9, 0],  # person moving
        [300-f*1, 200+f*1, 360-f*1, 260+f*1, 0.85, 2], # car
    ]
    if f > 30 and f < 35:
        dets = []  # occlusion
    tracks = tracker.update(dets)
    print(f"Frame {f}: dets {len(dets)} -> tracks {len(tracks)} IDs {[int(t[4]) for t in tracks]}")
    assert len(tracks) <= 2

print("Tracker sanity check PASSED")
