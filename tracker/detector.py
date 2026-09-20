import sys, types
# Workaround for torchvision 0.29 + torch 2.14 on Python 3.14 (torchvision::nms kernel missing)
# Inject dummy torchvision before ultralytics imports it, so it falls back to pure-Python NMS
try:
    import torch
    need_dummy = False
    try:
        import torchvision  # noqa
        if not torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "CompositeImplicitAutograd"):
            need_dummy = True
    except Exception:
        need_dummy = True
    if need_dummy:
        dummy = types.ModuleType('torchvision')
        dummy_ops = types.ModuleType('torchvision.ops')
        def _dummy_nms(boxes, scores, iou_threshold):
            # Pure-Python NMS fallback (boxes: Tensor[N,4] x1y1x2y2)
            if boxes.numel() == 0:
                return torch.empty((0,), dtype=torch.int64, device=boxes.device)
            # use Python loop - acceptable for warmup/NMS of few hundred boxes
            idxs = scores.argsort(descending=True)
            keep = []
            while idxs.numel() > 0:
                i = idxs[0].item()
                keep.append(i)
                if idxs.numel() == 1:
                    break
                # IoU of remaining vs i
                xx1 = torch.maximum(boxes[i, 0], boxes[idxs[1:], 0])
                yy1 = torch.maximum(boxes[i, 1], boxes[idxs[1:], 1])
                xx2 = torch.minimum(boxes[i, 2], boxes[idxs[1:], 2])
                yy2 = torch.minimum(boxes[i, 3], boxes[idxs[1:], 3])
                w = (xx2 - xx1).clamp(min=0)
                h = (yy2 - yy1).clamp(min=0)
                inter = w * h
                area_i = (boxes[i, 2]-boxes[i, 0])*(boxes[i, 3]-boxes[i, 1])
                area_rest = (boxes[idxs[1:], 2]-boxes[idxs[1:], 0])*(boxes[idxs[1:], 3]-boxes[idxs[1:], 1])
                iou = inter / (area_i + area_rest - inter + 1e-6)
                idxs = idxs[1:][iou <= iou_threshold]
            return torch.tensor(keep, dtype=torch.int64, device=boxes.device)
        dummy_ops.nms = _dummy_nms
        dummy.ops = dummy_ops
        sys.modules['torchvision'] = dummy
        sys.modules['torchvision.ops'] = dummy_ops
        print("[Detector] Injected dummy torchvision (fallback NMS)")
except Exception as e:
    print(f"[Detector] torchvision workaround failed: {e}")

from ultralytics import YOLO
import cv2
import numpy as np

COCO_NAMES = [
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
 "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
 "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
 "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
 "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
 "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
 "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
 "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.35, iou=0.45, classes=None, imgsz=640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.imgsz = imgsz
        # Warmup
        print(f"[Detector] Loaded {model_path} | conf={conf} iou={iou}")

    def detect(self, frame):
        """
        Returns: list of [x1,y1,x2,y2, conf, cls]
        """
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, classes=self.classes, imgsz=self.imgsz, verbose=False)
        detections = []
        if len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()
                for (x1,y1,x2,y2), c, cls in zip(boxes, confs, clss):
                    detections.append([float(x1),float(y1),float(x2),float(y2),float(c),int(cls)])
        return detections

    def get_class_name(self, cls_id):
        if 0 <= cls_id < len(COCO_NAMES):
            return COCO_NAMES[cls_id]
        return str(cls_id)
