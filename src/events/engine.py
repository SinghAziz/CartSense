"""Build cart events from tracked objects crossing a configured zone.

The camera must remain fixed. A track that starts inside the cart zone does not
produce an ADD event because the pipeline has no evidence that it entered.
"""

import datetime as dt
import uuid
from collections import Counter
from dataclasses import dataclass, field

from src.common import Event


@dataclass
class TrackObs:
    track_id: int
    bbox: tuple
    label: str
    confidence: float
    barcodes: list = field(default_factory=list)


@dataclass
class _TrackState:
    observation_count: int = 0
    zone_phase: str | None = None
    consecutive_inside_frames: int = 0
    consecutive_outside_frames: int = 0
    last_frame_index: int = 0
    label_counts: Counter = field(default_factory=Counter)
    confidence_scores: list = field(default_factory=list)
    barcodes: list = field(default_factory=list)
    transition_count: int = 0


class EventEngine:
    def __init__(self, identifier, session_id, config):
        self.identifier = identifier
        self.session_id = session_id
        self.config = config
        self.run_id = uuid.uuid4().hex[:10]
        self.tracks = {}

    def _inside(self, bbox, frame_size):
        zone_x1, zone_y1, zone_x2, zone_y2 = self.config["zone"]
        width, height = frame_size
        center_x = (bbox[0] + bbox[2]) / 2 / width
        center_y = (bbox[1] + bbox[3]) / 2 / height
        return zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2

    def update(self, frame_index, media_time_seconds, frame_size, observations):
        confirmation_frames = self.config["min_frames_confirm"]
        events = []

        for observation in observations:
            track_state = self.tracks.setdefault(observation.track_id, _TrackState())
            track_state.observation_count += 1
            track_state.last_frame_index = frame_index
            track_state.label_counts[observation.label] += 1
            track_state.confidence_scores.append(observation.confidence)
            track_state.barcodes.extend(observation.barcodes)

            is_inside = self._inside(observation.bbox, frame_size)
            if is_inside:
                track_state.consecutive_inside_frames += 1
                track_state.consecutive_outside_frames = 0
            else:
                track_state.consecutive_outside_frames += 1
                track_state.consecutive_inside_frames = 0

            track_is_ready = (
                track_state.observation_count >= self.config["min_track_hits"]
            )
            if track_state.consecutive_inside_frames >= confirmation_frames:
                if track_state.zone_phase == "outside" and track_is_ready:
                    events.append(
                        self._emit(
                            "ADD",
                            observation.track_id,
                            track_state,
                            frame_index,
                            media_time_seconds,
                            "entered cart zone",
                        )
                    )
                track_state.zone_phase = "inside"
            elif track_state.consecutive_outside_frames >= confirmation_frames:
                if track_state.zone_phase == "inside" and track_is_ready:
                    events.append(
                        self._emit(
                            "REMOVE",
                            observation.track_id,
                            track_state,
                            frame_index,
                            media_time_seconds,
                            "left cart zone",
                        )
                    )
                track_state.zone_phase = "outside"

        for track_id, track_state in list(self.tracks.items()):
            frames_since_last_seen = frame_index - track_state.last_frame_index
            if frames_since_last_seen > self.config["max_lost_frames"]:
                event = self._lost(
                    track_id, track_state, frame_index, media_time_seconds
                )
                if event:
                    events.append(event)
                del self.tracks[track_id]
        return events

    def flush(self, frame_index, media_time_seconds):
        events = []
        for track_id, track_state in list(self.tracks.items()):
            event = self._lost(track_id, track_state, frame_index, media_time_seconds)
            if event:
                events.append(event)
            del self.tracks[track_id]
        return events

    def _lost(self, track_id, track_state, frame_index, media_time_seconds):
        mid_transfer = (
            track_state.zone_phase == "outside"
            and track_state.consecutive_inside_frames > 0
        ) or (
            track_state.zone_phase == "inside"
            and track_state.consecutive_outside_frames > 0
        )
        track_is_ready = track_state.observation_count >= self.config["min_track_hits"]
        if mid_transfer and track_is_ready:
            return self._emit(
                "UNKNOWN",
                track_id,
                track_state,
                frame_index,
                media_time_seconds,
                "track lost mid-transfer; direction not confirmed",
            )
        return None

    def _emit(
        self,
        event_type,
        track_id,
        track_state,
        frame_index,
        media_time_seconds,
        reason,
    ):
        label = track_state.label_counts.most_common(1)[0][0]
        mean_confidence = sum(track_state.confidence_scores) / len(
            track_state.confidence_scores
        )
        identification = self.identifier.identify(
            label, mean_confidence, track_state.barcodes
        )
        dedupe_key = (
            f"{self.session_id}:{self.run_id}:{track_id}:{event_type}:"
            f"{track_state.transition_count}"
        )
        track_state.transition_count += 1

        if identification.method == "barcode" and identification.status == "RESOLVED":
            verification = "BARCODE_VERIFIED"
        elif identification.method in (
            "class_only",
            "learned_class",
            "learned_barcode",
        ):
            verification = "CLASS_ONLY"
        else:
            verification = "UNRESOLVED"

        event = Event(
            event_id="E-" + uuid.uuid4().hex[:8],
            session_id=self.session_id,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            media_time_s=media_time_seconds,
            event_type=event_type,
            state="OBSERVED",
            sku_id=identification.sku_id,
            track_id=track_id,
            verification=verification,
            detector_confidence=round(mean_confidence, 4),
            evidence=(
                f"{reason}; track hits={track_state.observation_count}; "
                f"class={label}; id={identification.status}/{identification.method}: "
                f"{identification.reason}"
            ),
            dedupe_key=dedupe_key,
            frame_idx=frame_index,
        )
        event.state_history = ["OBSERVED", "TRACKED"]
        event.state = "TRACKED"
        if event_type in ("ADD", "REMOVE"):
            event.set_state("CANDIDATE_" + event_type)

        if mean_confidence < self.config["reject_below_detector_conf"]:
            event.set_state("REJECTED")
            event.evidence += " | rejected: low detector confidence"
        elif event_type == "UNKNOWN" or identification.status != "RESOLVED":
            event.set_state("NEEDS_REVIEW")
        elif (
            identification.method in ("class_only", "learned_class", "learned_barcode")
            and self.config["auto_accept_requires_barcode"]
        ):
            event.set_state("NEEDS_REVIEW")
            event.evidence += " | class-only identification needs barcode/manual check"
        else:
            event.set_state("ACCEPTED")
        return event
