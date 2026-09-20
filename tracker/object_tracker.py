"""
Unified Multi-Object Tracker
Supports: ByteTrack (simple IoU + Kalman), SORT, Centroid
No external tracker dependency - pure numpy/scipy.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from collections import deque

def iou(bb1, bb2):
    xx1 = max(bb1[0], bb2[0])
    yy1 = max(bb1[1], bb2[1])
    xx2 = min(bb1[2], bb2[2])
    yy2 = min(bb1[3], bb2[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w*h
    area1 = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    union = area1 + area2 - inter
    if union == 0: return 0
    return inter / union

def iou_batch(dets, trks):
    ious = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            ious[d,t] = iou(det, trk)
    return ious

def convert_bbox_to_z(bbox):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h
    r = w/float(h) if h!=0 else 0
    return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2]*x[3]) if x[2]*x[3] >0 else 0
    h = x[2]/w if w!=0 else 0
    x1 = x[0]-w/2.
    y1 = x[1]-h/2.
    x2 = x[0]+w/2.
    y2 = x[1]+h/2.
    if score is None:
        return np.array([x1,y1,x2,y2]).reshape((1,4))
    return np.array([x1,y1,x2,y2,score]).reshape((1,5))

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # bbox: [x1,y1,x2,y2,conf,cls]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls = int(bbox[5]) if len(bbox)>5 else -1
        self.conf = bbox[4]
        self.history = deque(maxlen=30)
        self.bbox = bbox[:4]

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.bbox = bbox[:4]
        self.cls = int(bbox[5]) if len(bbox)>5 else self.cls
        self.conf = bbox[4]

    def predict(self):
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x)[0])
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)[0]

# ---------- SORT ----------
class SORTTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections):
        """
        detections: list of [x1,y1,x2,y2,conf,cls]
        returns: list of [x1,y1,x2,y2, track_id, cls, conf]
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.delete(trks, to_del, axis=0)
        for t in reversed(to_del):
            self.trackers.pop(t)

        if len(detections)>0 and len(trks)>0:
            dets = np.array([d[:4] for d in detections])
            iou_mat = iou_batch(dets, trks)
            matched_indices = linear_sum_assignment(-iou_mat)
            matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
            unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:,0]]
            unmatched_trks = [t for t in range(len(trks)) if t not in matched_indices[:,1]]
            matches = []
            for m in matched_indices:
                if iou_mat[m[0], m[1]] < self.iou_threshold:
                    unmatched_dets.append(m[0])
                    unmatched_trks.append(m[1])
                else:
                    matches.append(m.reshape(1,2))
            if len(matches)==0:
                matches = np.empty((0,2), dtype=int)
            else:
                matches = np.concatenate(matches, axis=0)
        else:
            matches = np.empty((0,2), dtype=int)
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(trks)))

        for m in matches:
            self.trackers[m[1]].update(detections[m[0]])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append([d[0],d[1],d[2],d[3], trk.id+1, trk.cls, trk.conf])
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return ret

# ---------- ByteTrack (simplified) ----------
class ByteTracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, min_hits=1):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections):
        self.frame_count += 1
        # split high vs low conf
        high_dets = [d for d in detections if d[4] >= self.track_thresh]
        low_dets  = [d for d in detections if d[4] < self.track_thresh]

        # predict
        for trk in self.trackers:
            trk.predict()

        # first association with high dets
        if len(high_dets)>0 and len(self.trackers)>0:
            dets = np.array([d[:4] for d in high_dets])
            trks = np.array([t.get_state()[:4] for t in self.trackers])
            iou_mat = iou_batch(dets, trks)
            matched, u_dets, u_trks = self._associate(iou_mat, self.match_thresh)
            for m in matched:
                self.trackers[m[1]].update(high_dets[m[0]])
            # second association with low dets for unmatched tracks
            if len(low_dets)>0 and len(u_trks)>0:
                dets2 = np.array([d[:4] for d in low_dets])
                trks2 = np.array([self.trackers[i].get_state()[:4] for i in u_trks])
                iou_mat2 = iou_batch(dets2, trks2)
                matched2, _, _ = self._associate(iou_mat2, 0.5)
                for m in matched2:
                    tidx = u_trks[m[1]]
                    self.trackers[tidx].update(low_dets[m[0]])
                    # remove from unmatched
                # keep unmatched high det indices for new tracks
                unmatched_high = [high_dets[i] for i in u_dets]
            else:
                unmatched_high = [high_dets[i] for i in u_dets]
        else:
            unmatched_high = high_dets

        # create new trackers
        for d in unmatched_high:
            self.trackers.append(KalmanBoxTracker(d))

        # remove aged
        for i in reversed(range(len(self.trackers))):
            if self.trackers[i].time_since_update > self.track_buffer:
                self.trackers.pop(i)

        # output
        ret=[]
        for trk in self.trackers:
            if trk.time_since_update <1 and trk.hits >= self.min_hits:
                d = trk.get_state()
                ret.append([d[0],d[1],d[2],d[3], trk.id+1, trk.cls, trk.conf])
            elif trk.time_since_update <1 and self.frame_count <5:
                # show early tracks too
                d = trk.get_state()
                ret.append([d[0],d[1],d[2],d[3], trk.id+1, trk.cls, trk.conf])
        return ret

    def _associate(self, iou_mat, thresh):
        if iou_mat.size==0:
            return np.empty((0,2),dtype=int), list(range(iou_mat.shape[0])), list(range(iou_mat.shape[1])) if iou_mat.ndim==2 else []
        row, col = linear_sum_assignment(-iou_mat)
        matches=[]
        unmatched_dets = set(range(iou_mat.shape[0]))
        unmatched_trks = set(range(iou_mat.shape[1]))
        for r,c in zip(row,col):
            if iou_mat[r,c] >= thresh:
                matches.append([r,c])
                unmatched_dets.discard(r)
                unmatched_trks.discard(c)
            else:
                pass
        return np.array(matches) if matches else np.empty((0,2),dtype=int), list(unmatched_dets), list(unmatched_trks)

# ---------- Centroid (fallback, no kalman) ----------
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_id=1
        self.objects={} # id -> bbox
        self.disappeared={}
        self.trails={} # id -> deque
        self.max_disappeared=max_disappeared
        self.max_distance=max_distance

    def update(self, detections):
        if len(detections)==0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid]+=1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
            return [[*v, k, -1, 0.9] for k,v in self.objects.items()]

        input_centroids = np.array([[(d[0]+d[2])/2, (d[1]+d[3])/2] for d in detections])
        input_bboxes = detections

        if len(self.objects)==0:
            for bbox in input_bboxes:
                self.objects[self.next_id]=bbox[:4]
                self.disappeared[self.next_id]=0
                self.trails[self.next_id]=deque(maxlen=30)
                self.next_id+=1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in self.objects.values()])
            D = np.linalg.norm(object_centroids[:,None]-input_centroids[None,:], axis=2)
            rows, cols = linear_sum_assignment(D)
            used_rows=set(); used_cols=set()
            for r,c in zip(rows,cols):
                if D[r,c] > self.max_distance:
                    continue
                oid = object_ids[r]
                self.objects[oid]=input_bboxes[c][:4]
                self.disappeared[oid]=0
                self.trails[oid].append(input_centroids[c])
                used_rows.add(r); used_cols.add(c)
            unused_rows = set(range(len(object_ids))) - used_rows
            unused_cols = set(range(len(input_bboxes))) - used_cols
            for r in unused_rows:
                oid = object_ids[r]
                self.disappeared[oid]+=1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
            for c in unused_cols:
                self.objects[self.next_id]=input_bboxes[c][:4]
                self.disappeared[self.next_id]=0
                self.trails[self.next_id]=deque(maxlen=30)
                self.next_id+=1
        return [[*v, k, -1, 0.9] for k,v in self.objects.items()]

def build_tracker(cfg):
    t = cfg.get("tracker_type","bytetrack").lower()
    if t=="sort":
        return SORTTracker(max_age=cfg.get("track_buffer",30), min_hits=cfg.get("min_hits",3), iou_threshold=cfg.get("match_thresh",0.3))
    elif t=="centroid":
        return CentroidTracker(max_disappeared=cfg.get("track_buffer",30))
    else:
        return ByteTracker(track_thresh=cfg.get("track_thresh",0.5), match_thresh=cfg.get("match_thresh",0.8),
                           track_buffer=cfg.get("track_buffer",30), min_hits=cfg.get("min_hits",1))
