"""Pretrained YOLOX (Apache-2.0, COCO-80) via ONNX Runtime. Detects generic COCO classes, NOT grocery SKUs."""
import numpy as np, cv2, onnxruntime as ort
from src.common import Detection

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter bench "
        "bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard "
        "sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon "
        "bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table "
        "toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors "
        "teddy_bear hair_drier toothbrush").split()
COCO = [c.replace("_", " ") for c in COCO]

class YoloxOnnxDetector:
    model_id = "yolox-onnx"
    def __init__(self, model_path, input_size=416, conf_threshold=0.3, nms_threshold=0.45, providers=None):
        avail = ort.get_available_providers()
        prov = [p for p in (providers or ["CPUExecutionProvider"]) if p in avail] or ["CPUExecutionProvider"]
        try:
            self.sess = ort.InferenceSession(model_path, providers=prov)
        except Exception:
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.providers = self.sess.get_providers()
        self.size, self.conf, self.nms = input_size, conf_threshold, nms_threshold
        self.inp = self.sess.get_inputs()[0].name
        g, s = [], []
        for st in (8, 16, 32):
            n = input_size // st
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grid = np.stack((xv, yv), 2).reshape(-1, 2)
            g.append(grid); s.append(np.full((len(grid), 1), st))
        self.grids, self.strides = np.concatenate(g), np.concatenate(s)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(h * r), int(w * r)
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        pad[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = pad.transpose(2, 0, 1)[None].astype(np.float32)        # YOLOX: BGR, 0-255, no mean/std
        out = self.sess.run(None, {self.inp: x})[0][0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        scores = out[:, 4:5] * out[:, 5:]
        cls, conf = scores.argmax(1), scores.max(1)
        m = conf >= self.conf
        if not m.any():
            return []
        tl = (xy[m] - wh[m] / 2) / r
        boxes = np.concatenate([tl, wh[m] / r], 1)                  # x,y,w,h in original pixels
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf[m].tolist(), cls[m].tolist(), self.conf, self.nms)
        res = []
        for i in np.array(idx).flatten():
            x0, y0, bw, bh = boxes[i]
            x1, y1 = max(0, x0), max(0, y0)
            res.append(Detection((float(x1), float(y1), float(min(w, x0 + bw)), float(min(h, y0 + bh))),
                                 int(cls[m][i]), COCO[int(cls[m][i])], float(conf[m][i]), model_id=self.model_id))
        return res


class YoloDetectorAdapter:
    """Optional Ultralytics ONNX adapter. Requires caller-supplied compatible weights/runtime."""
    model_id = "yolo-onnx"

    def __init__(self, model_path, confidence=0.3):
        from pathlib import Path
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"YOLO model unavailable: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO adapter requires the optional ultralytics package") from exc
        self.model = YOLO(str(model_path), task="detect")
        self.confidence = float(confidence)

    def detect(self, bgr):
        result = self.model.predict(source=bgr, conf=self.confidence, verbose=False)[0]
        names = result.names
        out = []
        for box in result.boxes:
            cid = int(box.cls.item()); score = float(box.conf.item())
            out.append(Detection(tuple(map(float, box.xyxy[0].tolist())), cid,
                                 str(names[cid]), score, model_id=self.model_id))
        return out


def make_detector(cfg):
    """Select an independently loadable detector; never downloads weights."""
    kind = cfg.get("backend", "yolox")
    if kind == "yolo":
        return YoloDetectorAdapter(cfg["model_path"], cfg.get("conf_threshold", 0.3))
    if kind == "yolox":
        return YoloxOnnxDetector(cfg["model_path"], cfg["input_size"], cfg["conf_threshold"],
                                 cfg["nms_threshold"], cfg.get("providers"))
    raise ValueError(f"unknown detector backend: {kind}")
