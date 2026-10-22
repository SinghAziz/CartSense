"""Orchestration: input -> detect -> (track) -> barcode -> identify -> events -> cart -> bill."""
import time, cv2, numpy as np
from src.barcode.decoder import make_decoder
from src.tracking.tracker import ByteTrackTracker
from src.identification.identifier import Identifier
from src.events.engine import EventEngine, TrackObs
from src.input.sources import open_video, iter_frames

def _ms(t0): return (time.perf_counter() - t0) * 1000

def _contains(b, p, m=0.05):
    w, h = b[2] - b[0], b[3] - b[1]
    return b[0] - m * w <= p[0] <= b[2] + m * w and b[1] - m * h <= p[1] <= b[3] + m * h

def associate(dets, barcodes):
    """Barcode belongs to a detection if its centre lies inside the box. Returns (per-det lists, orphans)."""
    per, orphans = [[] for _ in dets], []
    for bc in barcodes:
        c = ((bc.bbox[0] + bc.bbox[2]) / 2, (bc.bbox[1] + bc.bbox[3]) / 2)
        hit = [i for i, d in enumerate(dets) if _contains(d.bbox, c)]
        if hit:
            i = min(hit, key=lambda i: (dets[i].bbox[2] - dets[i].bbox[0]) * (dets[i].bbox[3] - dets[i].bbox[1]))
            per[i].append(bc.value)
        else:
            orphans.append(bc)
    return per, orphans

def annotate(img, dets, idents, zone=None, orphans=()):
    out = img.copy(); H, W = out.shape[:2]
    if zone:
        cv2.rectangle(out, (int(zone[0] * W), int(zone[1] * H)), (int(zone[2] * W) - 1, int(zone[3] * H) - 1), (255, 160, 0), 2)
    for d, idn in zip(dets, idents):
        x1, y1, x2, y2 = map(int, d.bbox)
        col = (0, 170, 0) if idn.status == "RESOLVED" else (0, 140, 255) if idn.status == "UNRESOLVED" else (0, 0, 220)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        txt = (f"#{d.track_id} " if d.track_id is not None else "") + f"{d.label} {d.confidence:.2f}" + (f" {idn.sku_id}" if idn.sku_id else " ?")
        cv2.putText(out, txt, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    for bc in orphans:
        cv2.rectangle(out, tuple(map(int, bc.bbox[:2])), tuple(map(int, bc.bbox[2:])), (200, 0, 200), 2)
    return out

class Pipeline:
    def __init__(self, cfg, detector, catalog, session, decoder=None):
        self.cfg, self.det, self.cat, self.session = cfg, detector, catalog, session
        self.decoder = decoder or make_decoder(cfg["barcode"].get("backend", "zxing") if cfg["barcode"]["enabled"] else "none")
        cc = cfg["catalog"]
        self.idf = Identifier(catalog, cc["allow_class_only_sku"], cc["conflict_on_class_mismatch"])
        self.ignore = set(cfg.get("ignore_classes", []))

    def process_image(self, img):
        """Single image: detection + barcode + identification only. No cart events (no temporal evidence)."""
        t0 = time.perf_counter(); dets = self.det.detect(img); t_det = _ms(t0)
        t0 = time.perf_counter(); bcs = self.decoder.decode(img); t_bc = _ms(t0)
        per, orphans = associate(dets, bcs)
        idents = [self.idf.identify(d.label, d.confidence, b) for d, b in zip(dets, per)]
        orphan_ids = [self.idf.identify(None, None, [o.value]) for o in orphans]
        return dict(detections=dets, identifications=idents, orphan_barcodes=list(zip(orphans, orphan_ids)),
                    annotated=annotate(img, dets, idents, orphans=orphans), latency_ms=dict(detect=t_det, barcode=t_bc))

    def process_video(self, source, out_path=None, max_frames=None, on_frame=None):
        cap, fps = open_video(source); stride = self.cfg["video"]["frame_stride"]
        tracker = ByteTrackTracker(fps / stride)
        ev_cfg = self.cfg["events"]; engine = EventEngine(self.idf, self.session.session_id, ev_cfg)
        lat = dict(detect=[], barcode=[], track=[], total=[]); new_events, writer, n, wh, last = [], None, 0, None, 0
        for idx, t, frame in iter_frames(cap, fps, stride, max_frames):
            T0 = time.perf_counter(); H, W = frame.shape[:2]; wh = (W, H); last = idx
            t0 = time.perf_counter(); dets = [d for d in self.det.detect(frame) if d.label not in self.ignore]; lat["detect"].append(_ms(t0))
            t0 = time.perf_counter(); tr = tracker.update(dets); lat["track"].append(_ms(t0))
            t0 = time.perf_counter(); bcs = self.decoder.decode(frame); lat["barcode"].append(_ms(t0))
            per, orphans = associate(tr, bcs)
            obs = [TrackObs(d.track_id, d.bbox, d.label, d.confidence, b) for d, b in zip(tr, per)]
            evs = engine.update(idx, t, wh, obs)
            for e in evs:
                self.session.submit(e); new_events.append(e)
            idents = [self.idf.identify(d.label, d.confidence, b) for d, b in zip(tr, per)]
            if out_path:
                if writer is None:
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, wh)
                writer.write(annotate(frame, tr, idents, ev_cfg["zone"], orphans))
            lat["total"].append(_ms(T0)); n += 1
            if on_frame: on_frame(idx, tr, evs)
        for e in engine.flush(last, last / fps):
            self.session.submit(e); new_events.append(e)
        if writer: writer.release()
        return dict(frames=n, events=new_events, latency_ms=lat, fps_source=fps, out_path=str(out_path) if out_path else None)
