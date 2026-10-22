"""Authoritative digital cart. Only EVENTS change it; raw detections never do. Idempotent + audited."""
import uuid, datetime as dt
from src.common import Event

def _now(): return dt.datetime.now(dt.timezone.utc).isoformat()

class CartSession:
    def __init__(self, catalog=None, session_id=None):
        self.catalog = catalog
        self.history = []          # archived previous sessions (snapshots)
        self._init(session_id)

    def _init(self, session_id=None):
        self.session_id = session_id or "S-" + uuid.uuid4().hex[:8]
        self.items, self.events, self.audit = {}, {}, []
        self.applied_ids, self.applied_keys = set(), set()

    def _log(self, action, event_id=None, **detail):
        self.audit.append(dict(ts=_now(), action=action, event_id=event_id, **detail))

    def submit(self, ev: Event):
        seen_keys = {e.dedupe_key for e in self.events.values() if e.dedupe_key}
        if ev.event_id in self.events or (ev.dedupe_key and ev.dedupe_key in seen_keys):
            self._log("DUPLICATE_IGNORED", ev.event_id, dedupe_key=ev.dedupe_key)
            return False
        self.events[ev.event_id] = ev
        self._log("EVENT_RECEIVED", ev.event_id, type=ev.event_type, state=ev.state, sku=ev.sku_id)
        if ev.state == "ACCEPTED":
            return self._apply(ev)
        ev.processing_status = "DISCARDED" if ev.state == "REJECTED" else "HELD"
        return False

    def _apply(self, ev):
        if ev.event_id in self.applied_ids:                 # idempotent replay
            self._log("REPLAY_IGNORED", ev.event_id); return False
        if ev.state != "ACCEPTED" or not ev.sku_id or ev.event_type not in ("ADD", "REMOVE"):
            raise ValueError("only ACCEPTED ADD/REMOVE events with a SKU can modify the cart")
        if ev.event_type == "ADD":
            self.items[ev.sku_id] = self.items.get(ev.sku_id, 0) + 1
        else:
            if self.items.get(ev.sku_id, 0) < 1:
                ev.set_state("NEEDS_REVIEW"); ev.processing_status = "HELD"
                ev.evidence += " | removal exceeds cart quantity"
                self._log("REMOVE_REFUSED", ev.event_id, sku=ev.sku_id)
                return False
            self.items[ev.sku_id] -= 1
            if self.items[ev.sku_id] == 0: del self.items[ev.sku_id]
        self.applied_ids.add(ev.event_id); self.applied_keys.add(ev.dedupe_key)
        ev.processing_status = "APPLIED"
        self._log("APPLIED", ev.event_id, type=ev.event_type, sku=ev.sku_id, resolution=ev.resolution)
        return True

    def pending(self):
        return [e for e in self.events.values() if e.state == "NEEDS_REVIEW"]

    def confirm(self, event_id, reason, sku_id=None, event_type=None):
        ev = self.events[event_id]
        if ev.state != "NEEDS_REVIEW": raise ValueError("event is not awaiting review")
        sku_id = sku_id or ev.sku_id
        if not sku_id or (self.catalog and self.catalog.get(sku_id) is None):
            raise ValueError(f"unknown or missing SKU: {sku_id}")
        et = event_type or ev.event_type
        if et not in ("ADD", "REMOVE"): raise ValueError("choose ADD or REMOVE to confirm")
        self._log("MANUAL_CONFIRM", event_id, reason=reason, old_sku=ev.sku_id, new_sku=sku_id,
                  old_type=ev.event_type, new_type=et)
        ev.sku_id, ev.event_type, ev.resolution, ev.verification = sku_id, et, "MANUAL", "MANUAL"
        ev.set_state("ACCEPTED")
        return self._apply(ev)

    def reject(self, event_id, reason):
        ev = self.events[event_id]
        if ev.event_id in self.applied_ids: raise ValueError("already applied; use a corrective event")
        self._log("MANUAL_REJECT", event_id, reason=reason)
        ev.resolution = "MANUAL"; ev.set_state("REJECTED"); ev.processing_status = "DISCARDED"

    def manual_event(self, event_type, sku_id, reason):
        """Operator-created event (e.g. from a single image, which has no temporal evidence)."""
        ev = Event(event_id="E-" + uuid.uuid4().hex[:8], session_id=self.session_id, timestamp=_now(),
                   media_time_s=None, event_type=event_type, state="NEEDS_REVIEW", sku_id=sku_id,
                   track_id=None, verification="MANUAL", detector_confidence=None,
                   evidence="manually created: " + reason, dedupe_key="manual:" + uuid.uuid4().hex)
        ev.state_history = ["NEEDS_REVIEW"]
        self.events[ev.event_id] = ev
        return self.confirm(ev.event_id, reason)

    def snapshot(self):
        return dict(session_id=self.session_id, items=dict(self.items), audit=list(self.audit),
                    events=[e.to_dict() for e in self.events.values()])

    def reset(self):
        self.history.append(self.snapshot())     # previous records preserved
        self._init()
