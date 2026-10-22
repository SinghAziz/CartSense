"""Shared dataclasses (CartSense-specific)."""
from dataclasses import dataclass, field, asdict
from typing import Optional

EVENT_STATES = ["OBSERVED", "TRACKED", "CANDIDATE_ADD", "CANDIDATE_REMOVE",
                "ACCEPTED", "NEEDS_REVIEW", "REJECTED"]

@dataclass
class Detection:
    bbox: tuple            # x1,y1,x2,y2 pixels
    class_id: int
    label: str
    confidence: float      # raw detector confidence
    track_id: Optional[int] = None

@dataclass
class BarcodeRead:
    value: str
    symbology: str
    bbox: tuple

@dataclass
class Identification:
    status: str            # RESOLVED | UNRESOLVED | CONFLICT
    sku_id: Optional[str]
    method: str            # barcode | class_only | none
    reason: str
    barcode: Optional[str] = None
    detector_label: Optional[str] = None
    detector_confidence: Optional[float] = None   # raw, never fused

@dataclass
class Event:
    event_id: str
    session_id: str
    timestamp: str                 # wall-clock UTC ISO
    media_time_s: Optional[float]
    event_type: str                # ADD | REMOVE | UNKNOWN
    state: str                     # one of EVENT_STATES
    sku_id: Optional[str]
    track_id: Optional[int]
    verification: str              # BARCODE_VERIFIED | CLASS_ONLY | UNRESOLVED | MANUAL
    detector_confidence: Optional[float]   # raw mean detector conf (NOT a calibrated event confidence)
    evidence: str
    processing_status: str = "PENDING"     # PENDING | APPLIED | HELD | DISCARDED
    dedupe_key: str = ""
    frame_idx: Optional[int] = None
    resolution: str = "AUTO"       # AUTO | MANUAL
    state_history: list = field(default_factory=list)

    def set_state(self, s):
        assert s in EVENT_STATES, s
        self.state = s
        self.state_history.append(s)

    def to_dict(self):
        return asdict(self)
