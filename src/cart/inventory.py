"""Session-local item instances; visibility loss never directly edits the authoritative cart."""
from dataclasses import dataclass, field
from enum import Enum
import time


class Lifecycle(str, Enum):
    DETECTED = "detected"
    CONFIRMED_PRESENT = "confirmed_present"
    OCCLUDED = "occluded"
    REMOVAL_CANDIDATE = "removal_candidate"
    REMOVED = "removed"
    UNRESOLVED = "unresolved"


@dataclass
class ItemInstance:
    track_id: int
    bbox: tuple
    class_label: str
    detection_confidence: float
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    consecutive_hits: int = 1
    misses: int = 0
    lifecycle: Lifecycle = Lifecycle.DETECTED
    sku_id: str | None = None
    identity_confidence: str = "unknown"
    last_verified: float | None = None
    history: list = field(default_factory=list)


class CartInventory:
    """Tracks per-instance presence; accepted cart changes remain event/session-owned."""
    def __init__(self, confirm_hits=3, removal_misses=15):
        self.confirm_hits = int(confirm_hits)
        self.removal_misses = int(removal_misses)
        self.instances = {}
        self.needs_verification = False

    def update(self, observations, now=None):
        now = time.time() if now is None else now
        seen = set()
        for obs in observations:
            tid = int(obs.track_id); seen.add(tid)
            item = self.instances.get(tid)
            if item is None or item.lifecycle == Lifecycle.REMOVED:
                confidence = float(getattr(obs, "conf", getattr(obs, "confidence", 0.0)))
                item = ItemInstance(tid, tuple(obs.bbox), obs.label, confidence, first_seen=now, last_seen=now)
                self.instances[tid] = item
                self._record(item, "detected", "new track")
                if self.confirm_hits <= 1:
                    item.lifecycle = Lifecycle.CONFIRMED_PRESENT
                    self._record(item, "confirmed_present", "temporal hit threshold met")
            else:
                item.bbox = tuple(obs.bbox); item.class_label = obs.label
                item.detection_confidence = float(getattr(obs, "conf", getattr(obs, "confidence", 0.0))); item.last_seen = now
                item.consecutive_hits += 1; item.misses = 0
                if item.lifecycle in (Lifecycle.OCCLUDED, Lifecycle.REMOVAL_CANDIDATE):
                    item.lifecycle = Lifecycle.CONFIRMED_PRESENT
                    self._record(item, "confirmed_present", "track observed again")
                elif item.lifecycle == Lifecycle.DETECTED and item.consecutive_hits >= self.confirm_hits:
                    item.lifecycle = Lifecycle.CONFIRMED_PRESENT
                    self._record(item, "confirmed_present", "temporal hit threshold met")
        for tid, item in self.instances.items():
            if tid in seen or item.lifecycle in (Lifecycle.REMOVED, Lifecycle.UNRESOLVED):
                continue
            item.misses += 1
            if item.misses >= self.removal_misses and item.lifecycle != Lifecycle.REMOVAL_CANDIDATE:
                item.lifecycle = Lifecycle.REMOVAL_CANDIDATE
                self._record(item, "removal_candidate", "track absent beyond provisional miss tolerance; confirm removal with transfer evidence")
            elif item.lifecycle != Lifecycle.REMOVAL_CANDIDATE:
                item.lifecycle = Lifecycle.OCCLUDED
                self._record(item, "occluded", "temporarily absent; cart quantity preserved")

    def mark_unresolved(self, track_id, reason):
        item = self.instances[track_id]; item.lifecycle = Lifecycle.UNRESOLVED
        self.needs_verification = True; self._record(item, "unresolved", reason)

    def confirm_removed(self, track_id, reason):
        item = self.instances[track_id]; item.lifecycle = Lifecycle.REMOVED
        self._record(item, "removed", reason)

    def interruption(self):
        self.needs_verification = True

    def recover(self):
        self.needs_verification = True  # lost IDs cannot be physically reconciled from absence alone

    def set_identity(self, track_id, identification, verified_at=None):
        item = self.instances.get(int(track_id))
        if item is None:
            return
        item.sku_id = identification.sku_id
        item.identity_confidence = identification.confidence_category
        if identification.status == "RESOLVED":
            item.last_verified = time.time() if verified_at is None else verified_at
        if identification.status == "CONFLICT":
            item.lifecycle = Lifecycle.UNRESOLVED
            self.needs_verification = True
            self._record(item, "unresolved", identification.reason)

    def snapshot(self):
        return {"needs_verification": self.needs_verification,
                "items": [dict(track_id=x.track_id, sku_id=x.sku_id, class_label=x.class_label,
                               lifecycle=x.lifecycle.value, detection_confidence=x.detection_confidence,
                               identity_confidence=x.identity_confidence, misses=x.misses,
                               consecutive_hits=x.consecutive_hits, first_seen=x.first_seen, last_seen=x.last_seen,
                               last_verified=x.last_verified, history=list(x.history))
                          for x in self.instances.values()]}

    @staticmethod
    def _record(item, state, reason):
        item.history.append({"state": state, "reason": reason, "timestamp": time.time()})
