"""ByteTrack from the `supervision` library (reused, not reimplemented)."""
import numpy as np, supervision as sv
from src.common import Detection

class ByteTrackTracker:
    def __init__(self, frame_rate=30):
        self.t = sv.ByteTrack(frame_rate=int(max(1, frame_rate)))

    def update(self, dets):
        if not dets:
            self.t.update_with_detections(sv.Detections.empty())
            return []
        d = sv.Detections(xyxy=np.array([x.bbox for x in dets], dtype=float),
                          confidence=np.array([x.confidence for x in dets]),
                          class_id=np.array([x.class_id for x in dets]))
        r = self.t.update_with_detections(d)
        names = {x.class_id: x.label for x in dets}
        return [Detection(tuple(map(float, b)), int(c), names.get(int(c), str(c)), float(s), int(i))
                for b, c, s, i in zip(r.xyxy, r.class_id, r.confidence, r.tracker_id)]
