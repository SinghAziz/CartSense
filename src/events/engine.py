"""Cart-event engine: zone-crossing evidence from tracked objects -> candidate ADD/REMOVE/UNKNOWN events.
Assumes a fixed camera and a configurable 'cart zone'. Tracks that START inside the zone produce no ADD
(no transfer evidence). Thresholds come from config."""
import uuid, datetime as dt
from collections import Counter
from dataclasses import dataclass, field
from src.common import Event

@dataclass
class TrackObs:
    track_id: int
    bbox: tuple
    label: str
    conf: float
    barcodes: list = field(default_factory=list)   # barcodes read THIS frame on/in this object

class _TS:
    def __init__(self):
        self.hits = 0; self.phase = None; self.in_run = 0; self.out_run = 0; self.last_frame = 0
        self.labels = Counter(); self.confs = []; self.barcodes = []; self.transitions = 0

class EventEngine:
    def __init__(self, identifier, session_id, cfg, frame_wh=None):
        self.idf, self.sid, self.c = identifier, session_id, cfg
        self.tracks, self.emitted = {}, set()
        self.states = {}      # track_id -> last lifecycle state (OBSERVED/TRACKED) for display

    def _inside(self, bbox, wh):
        x1, y1, x2, y2 = self.c["zone"]; W, H = wh
        cx, cy = (bbox[0] + bbox[2]) / 2 / W, (bbox[1] + bbox[3]) / 2 / H
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def update(self, frame_idx, t, wh, obs):
        n, evs = self.c["min_frames_confirm"], []
        for o in obs:
            ts = self.tracks.setdefault(o.track_id, _TS())
            ts.hits += 1; ts.last_frame = frame_idx
            ts.labels[o.label] += 1; ts.confs.append(o.conf); ts.barcodes += list(o.barcodes)
            self.states[o.track_id] = "TRACKED" if ts.hits >= self.c["min_track_hits"] else "OBSERVED"
            if self._inside(o.bbox, wh): ts.in_run += 1; ts.out_run = 0
            else: ts.out_run += 1; ts.in_run = 0
            ready = ts.hits >= self.c["min_track_hits"]
            if ts.in_run >= n:
                if ts.phase == "outside" and ready:
                    evs.append(self._emit("ADD", o.track_id, ts, frame_idx, t, "entered cart zone"))
                ts.phase = "inside"
            elif ts.out_run >= n:
                if ts.phase == "inside" and ready:
                    evs.append(self._emit("REMOVE", o.track_id, ts, frame_idx, t, "left cart zone"))
                ts.phase = "outside"
        for tid, ts in list(self.tracks.items()):                # lost tracks
            if frame_idx - ts.last_frame > self.c["max_lost_frames"]:
                e = self._lost(tid, ts, frame_idx, t)
                if e: evs.append(e)
                del self.tracks[tid]
        return evs

    def flush(self, frame_idx, t):
        evs = []
        for tid, ts in list(self.tracks.items()):
            e = self._lost(tid, ts, frame_idx, t)
            if e: evs.append(e)
            del self.tracks[tid]
        return evs

    def _lost(self, tid, ts, frame_idx, t):
        mid = (ts.phase == "outside" and ts.in_run > 0) or (ts.phase == "inside" and ts.out_run > 0)
        if mid and ts.hits >= self.c["min_track_hits"]:
            return self._emit("UNKNOWN", tid, ts, frame_idx, t, "track lost mid-transfer; direction not confirmed")
        return None

    def _emit(self, typ, tid, ts, frame_idx, t, why):
        label = ts.labels.most_common(1)[0][0]
        mean_conf = sum(ts.confs) / len(ts.confs)
        ident = self.idf.identify(label, mean_conf, ts.barcodes)
        key = f"{self.sid}:{tid}:{typ}:{ts.transitions}"; ts.transitions += 1
        ev = Event(event_id="E-" + uuid.uuid4().hex[:8], session_id=self.sid,
                   timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), media_time_s=t,
                   event_type=typ, state="OBSERVED", sku_id=ident.sku_id, track_id=tid,
                   verification="BARCODE_VERIFIED" if ident.method == "barcode" and ident.status == "RESOLVED"
                   else ("CLASS_ONLY" if ident.method == "class_only" else "UNRESOLVED"),
                   detector_confidence=round(mean_conf, 4),
                   evidence=f"{why}; track hits={ts.hits}; class={label}; id={ident.status}/{ident.method}: {ident.reason}",
                   dedupe_key=key, frame_idx=frame_idx)
        ev.state_history = ["OBSERVED", "TRACKED"]; ev.state = "TRACKED"
        if typ in ("ADD", "REMOVE"):
            ev.set_state("CANDIDATE_" + typ)
        # --- decision policy (all thresholds configurable) ---
        if mean_conf < self.c["reject_below_detector_conf"]:
            ev.set_state("REJECTED"); ev.evidence += " | rejected: low detector confidence"
        elif typ == "UNKNOWN" or ident.status != "RESOLVED":
            ev.set_state("NEEDS_REVIEW")
        elif ident.method == "class_only" and self.c["auto_accept_requires_barcode"]:
            ev.set_state("NEEDS_REVIEW"); ev.evidence += " | class-only identification needs barcode/manual check"
        else:
            ev.set_state("ACCEPTED")
        return ev
