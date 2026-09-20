import cv2
import numpy as np
from collections import defaultdict, deque

# distinct colors for up to 80 ids
PALETTE = [(31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),
           (140,86,75),(227,119,194),(127,127,127),(188,189,34),(23,190,207)]

def get_color(idx):
    if idx < len(PALETTE):
        return PALETTE[idx % len(PALETTE)]
    # deterministic hash color
    np.random.seed(idx)
    return tuple(int(c) for c in np.random.randint(0,255,3))

class Visualizer:
    def __init__(self, draw_trails=True, trail_length=30):
        self.draw_trails = draw_trails
        self.trail_length = trail_length
        self.trails = defaultdict(lambda: deque(maxlen=trail_length))

    def draw(self, frame, tracks, detector=None, fps=0):
        """
        tracks: [x1,y1,x2,y2, track_id, cls, conf]
        """
        for x1,y1,x2,y2, tid, cls, conf in tracks:
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            color = get_color(int(tid))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"ID {int(tid)}"
            if detector is not None and cls is not None and cls>=0:
                label += f" {detector.get_class_name(int(cls))}"
            if conf is not None:
                label += f" {conf:.2f}"
            (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            if self.draw_trails:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                self.trails[int(tid)].append((cx,cy))
                pts = list(self.trails[int(tid)])
                for i in range(1, len(pts)):
                    thickness = int(1 + i/len(pts)*3)
                    cv2.line(frame, pts[i-1], pts[i], color, thickness)

        # HUD
        cv2.rectangle(frame, (0,0), (260, 70), (0,0,0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,150), 2)
        cv2.putText(frame, f"Press Q to quit", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        return frame
