"""Orchestration: input -> detect -> (track) -> barcode -> identify -> events -> cart -> bill."""

import sys, threading, time, cv2, numpy as np
from pathlib import Path
from src.barcode.decoder import make_decoder
from src.tracking.tracker import ByteTrackTracker
from src.identification.identifier import Identifier
from src.events.engine import EventEngine, TrackObs
from src.input.sources import open_video, iter_frames
from src.pricing.billing import compute_bill
from src.cart.inventory import CartInventory
from src.identification.learned_rules import LearnedRules
from src.tracking.geometry import iou


def _milliseconds_since(start_time):
    return (time.perf_counter() - start_time) * 1000


def _contains(bbox, point, margin_ratio=0.05):
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    return (
        bbox[0] - margin_ratio * box_width
        <= point[0]
        <= bbox[2] + margin_ratio * box_width
        and bbox[1] - margin_ratio * box_height
        <= point[1]
        <= bbox[3] + margin_ratio * box_height
    )


def associate(detections, barcode_reads):
    """Associate barcode values to containing detections; return unmatched reads too."""
    barcodes_by_detection = [[] for _ in detections]
    orphan_barcodes = []
    for barcode in barcode_reads:
        barcode_center = (
            (barcode.bbox[0] + barcode.bbox[2]) / 2,
            (barcode.bbox[1] + barcode.bbox[3]) / 2,
        )
        matching_indexes = [
            index
            for index, detection in enumerate(detections)
            if _contains(detection.bbox, barcode_center)
        ]
        if matching_indexes:
            smallest_detection_index = min(
                matching_indexes,
                key=lambda index: (
                    detections[index].bbox[2] - detections[index].bbox[0]
                )
                * (detections[index].bbox[3] - detections[index].bbox[1]),
            )
            barcodes_by_detection[smallest_detection_index].append(barcode.value)
        else:
            orphan_barcodes.append(barcode)
    return barcodes_by_detection, orphan_barcodes


def annotate(image, detections, identifications, zone=None, orphan_barcodes=()):
    annotated_image = image.copy()
    image_height, image_width = annotated_image.shape[:2]
    if zone:
        zone_x1, zone_y1 = int(zone[0] * image_width), int(zone[1] * image_height)
        zone_x2, zone_y2 = (
            int(zone[2] * image_width) - 1,
            int(zone[3] * image_height) - 1,
        )
        cv2.rectangle(
            annotated_image, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 160, 0), 2
        )
        cv2.putText(
            annotated_image,
            "CART ZONE",
            (zone_x1 + 10, max(24, zone_y1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 160, 0),
            2,
        )
        if zone_x1 > 10:
            cv2.putText(
                annotated_image,
                "OUTSIDE ZONE",
                (10, max(24, zone_y1 + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (120, 120, 120),
                2,
            )
    for detection, identification in zip(detections, identifications):
        x1, y1, x2, y2 = map(int, detection.bbox)
        col = (
            (0, 170, 0)
            if identification.status == "RESOLVED"
            else (0, 140, 255) if identification.status == "UNRESOLVED" else (0, 0, 220)
        )
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), col, 2)
        txt = (
            (f"#{detection.track_id} " if detection.track_id is not None else "")
            + f"{detection.label} {detection.confidence:.2f}"
            + (f" {identification.sku_id}" if identification.sku_id else " ?")
        )
        cv2.putText(
            annotated_image,
            txt,
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            2,
        )
    for barcode in orphan_barcodes:
        cv2.rectangle(
            annotated_image,
            tuple(map(int, barcode.bbox[:2])),
            tuple(map(int, barcode.bbox[2:])),
            (200, 0, 200),
            2,
        )
    return annotated_image


def annotate_cart_status(
    img, session, catalog, events=(), unit_price_cents=None, diagnostics=None
):
    """Show cart state and transparent per-frame component diagnostics on live output."""
    out = img.copy()
    bill = compute_bill(session, catalog, unit_price_cents)
    cart_text = (
        ", ".join(f"{line['sku_id']} x{line['qty']}" for line in bill["lines"])
        or "empty"
    )
    lines = [
        f"CART: {cart_text}",
        f"BILL: {bill['subtotal_cents'] / 100:.2f} {bill['currency']} ({bill['status']})",
        "REVIEW: Y=confirm, N=reject",
    ]
    lines.extend(
        f"{e.event_type}: {e.sku_id or 'unknown'} [{e.state}]" for e in events[-2:]
    )
    if bill["unresolved_events"]:
        lines.append(f"REVIEW: {len(bill['unresolved_events'])} event(s)")
    x, y = 10, 70
    box_h = 28 * len(lines) + 12
    overlay = out.copy()
    cv2.rectangle(
        overlay,
        (x - 6, y - 24),
        (min(out.shape[1] - 10, 520), y + box_h),
        (25, 25, 25),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
    for line in lines:
        color = (
            (0, 220, 0)
            if "ACCEPTED" in line
            else (
                (0, 190, 255)
                if "REVIEW" in line or "NEEDS_REVIEW" in line
                else (255, 255, 255)
            )
        )
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        y += 28
    if diagnostics:
        dx, dy = max(10, out.shape[1] - 440), 22
        rows = [
            f"LIVE DIAGNOSTICS | {diagnostics['model']}",
            f"Detections {diagnostics['detections']} | tracked {diagnostics['tracks']} | FPS {diagnostics['fps']:.1f}",
            f"Classes detected: {diagnostics['classes']}",
            f"Mean detector score {diagnostics['mean_conf']:.2f} (raw, uncalibrated)",
            f"Latency ms: detect {diagnostics['detect_ms']:.0f} | track {diagnostics['track_ms']:.1f}",
            f"Barcode {diagnostics['barcode_ms']:.1f} ms | decoded {diagnostics['barcodes']}",
            f"Identity: barcode verified {diagnostics['verified']} | unresolved {diagnostics['unresolved']}",
            f"Instance lifecycle: present {diagnostics['present']} | occluded {diagnostics['occluded']} | removal review {diagnostics['removal']}",
            f"OCR: {diagnostics['ocr']} | YOLO optional: {diagnostics['yolo']}",
            "Track score is not separately calibrated; ID age/misses shown by lifecycle.",
            "Each box label shows its detector score, track ID, and known SKU or ?.",
        ]
        width, height = 430, 24 * len(rows) + 14
        dx = max(10, out.shape[1] - width - 10)
        dy = max(210, out.shape[0] - height - 8)
        overlay = out.copy()
        cv2.rectangle(
            overlay,
            (dx, dy),
            (min(out.shape[1] - 4, dx + width), min(out.shape[0] - 4, dy + height)),
            (15, 15, 15),
            -1,
        )
        cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)
        for row in rows:
            cv2.putText(
                out,
                row,
                (dx + 10, dy + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            dy += 24
    return out


class Pipeline:
    def __init__(
        self, config, detector, catalog, session, decoder=None, learned_rules=None
    ):
        self.config = config
        self.detector = detector
        self.catalog = catalog
        self.session = session
        self.decoder = decoder or make_decoder(
            config["barcode"].get("backend", "zxing")
            if config["barcode"]["enabled"]
            else "none"
        )
        self.ignored_classes = set(config.get("ignore_classes", []))
        catalog_config = config["catalog"]
        self.learned = learned_rules or LearnedRules(
            config.get("learning", {}).get("path", "data/local_learned_rules.json")
        )
        self.identifier = Identifier(
            catalog,
            catalog_config["allow_class_only_sku"],
            catalog_config["conflict_on_class_mismatch"],
            self.learned,
            ignored_classes=self.ignored_classes,
        )

    def process_image(self, img):
        """Single image: detection + barcode + identification only. No cart events (no temporal evidence)."""
        t0 = time.perf_counter()
        detections = self.detector.detect(img)
        t_det = _milliseconds_since(t0)
        t0 = time.perf_counter()
        barcode_reads = self.decoder.decode(img)
        t_bc = _milliseconds_since(t0)
        barcodes_by_detection, orphan_barcodes = associate(detections, barcode_reads)
        identifications = [
            self.identifier.identify(d.label, d.confidence, barcodes)
            for d, barcodes in zip(detections, barcodes_by_detection)
        ]
        orphan_identifications = [
            self.identifier.identify(None, None, [barcode.value])
            for barcode in orphan_barcodes
        ]
        return dict(
            detections=detections,
            identifications=identifications,
            orphan_barcodes=list(zip(orphan_barcodes, orphan_identifications)),
            annotated=annotate(
                img,
                detections,
                identifications,
                orphan_barcodes=orphan_barcodes,
            ),
            latency_ms=dict(detect=t_det, barcode=t_bc),
        )

    def process_video(
        self,
        source,
        out_path=None,
        max_frames=None,
        on_frame=None,
        display=False,
        unit_price_cents=None,
        terminal_controls=False,
    ):
        video_capture, source_fps = open_video(source)
        frame_stride = self.config["video"]["frame_stride"]
        tracker = ByteTrackTracker(source_fps / frame_stride)
        tracker_config = self.config.get("tracker", {})
        inventory = CartInventory(
            tracker_config.get("confirm_hits", 3),
            tracker_config.get("removal_misses", 15),
        )
        event_config = self.config["events"]
        engine = EventEngine(self.identifier, self.session.session_id, event_config)
        latency_ms = dict(detect=[], barcode=[], track=[], total=[])
        new_events = []
        video_writer = None
        processed_frame_count = 0
        frame_size = None
        last_frame_index = 0
        stop = False
        commands = []
        live_ui = None
        current_rows = []

        def do_live_action(action, args):
            nonlocal stop
            if action == "quit":
                stop = True
                return "Stopping live view…"
            row = args.get("row") or next(
                (r for r in current_rows if r.get("track_id") == args.get("track_id")),
                None,
            )
            track_id, sku_id = args.get("track_id"), args.get("sku_id")
            label = args.get("label") or (row.get("label") if row else "")
            if action == "learn_barcode":
                self.learned.teach_barcode(args.get("barcode"), sku_id, self.catalog)
                if track_id is not None:
                    ident = self.identifier.identify(
                        label, row["confidence"], [args["barcode"]]
                    )
                    inventory.set_identity(track_id, ident)
                return f"Saved barcode rule for {sku_id}. It applies on future matching barcode reads."
            if action == "learn_class":
                if row is None:
                    raise ValueError("Select a detected object row first.")
                if label.lower() in {x.lower() for x in self.ignored_classes}:
                    raise ValueError(
                        f"'{label}' is ignored for cart tracking and cannot be learned as a product class."
                    )
                self.learned.teach_class(label, sku_id, self.catalog)
                if track_id is not None:
                    inventory.set_identity(
                        track_id, self.identifier.identify(label, row["confidence"], [])
                    )
                return f"Saved {label} → {sku_id}. Generic class rules can confuse brands and variants."
            if action == "forget_class":
                if not label:
                    raise ValueError("Select an object class first.")
                self.learned.class_to_sku.pop(label.lower(), None)
                self.learned.save()
                return f"Removed the saved rule for {label}."
            if action == "confirm":
                pending = [
                    e
                    for e in self.session.pending()
                    if (row is None and track_id is None) or e.track_id == track_id
                ]
                if not pending:
                    raise ValueError("No pending review event for the selected track.")
                ev = pending[0]
                return (
                    "Review applied."
                    if self.session.confirm(
                        ev.event_id,
                        "confirmed in live inspection console",
                        sku_id=sku_id or ev.sku_id,
                    )
                    else "Review remains unresolved. Check the selected SKU and cart quantity."
                )
            if action == "reject":
                pending = [
                    e
                    for e in self.session.pending()
                    if (row is None and track_id is None) or e.track_id == track_id
                ]
                if not pending:
                    raise ValueError("No pending review event for the selected track.")
                self.session.reject(
                    pending[0].event_id, "rejected in live inspection console"
                )
                return "Selected event rejected."
            if action in ("add", "remove"):
                if not row:
                    raise ValueError("Select a detected object first.")
                if not sku_id:
                    raise ValueError("Choose a catalogue SKU first.")
                self.session.manual_event(
                    action.upper(),
                    sku_id,
                    f"operator selected track {track_id} ({label}) in live console",
                )
                if track_id is not None:
                    inventory.set_identity(
                        track_id, self.identifier.identify(label, row["confidence"], [])
                    )
                return f"Manual {action.upper()} recorded for {sku_id}."
            if action == "manual_add":
                if not sku_id:
                    raise ValueError("Choose a catalogue SKU first.")
                self.session.manual_event(
                    "ADD", sku_id, "operator manually added SKU from live image review"
                )
                return f"Manual ADD recorded for {sku_id}."
            return ""

        if display:
            from src.live_ui import LiveConsole

            live_ui = LiveConsole(self.catalog, do_live_action)
        if display and terminal_controls:

            def read_commands():
                print(
                    "Terminal controls remain available: y [SKU], n, add SKU, remove SKU, or q."
                )
                for line in sys.stdin:
                    command = line.strip().lower()
                    if command:
                        commands.append(command)
                    if command == "q" or command.startswith("q "):
                        break

            threading.Thread(target=read_commands, daemon=True).start()
        for frame_index, media_time_seconds, frame in iter_frames(
            video_capture, source_fps, frame_stride, max_frames
        ):
            loop_start_time = time.perf_counter()
            frame_height, frame_width = frame.shape[:2]
            frame_size = (frame_width, frame_height)
            last_frame_index = frame_index
            stage_start_time = time.perf_counter()
            all_detections = self.detector.detect(frame)
            detections = [
                detection
                for detection in all_detections
                if detection.label not in self.ignored_classes
            ]
            latency_ms["detect"].append(_milliseconds_since(stage_start_time))
            stage_start_time = time.perf_counter()
            tracked_detections = tracker.update(detections)
            latency_ms["track"].append(_milliseconds_since(stage_start_time))
            inventory.update(tracked_detections, now=media_time_seconds)
            stage_start_time = time.perf_counter()
            barcode_reads = self.decoder.decode(frame)
            latency_ms["barcode"].append(_milliseconds_since(stage_start_time))
            barcodes_by_detection, orphan_barcodes = associate(
                tracked_detections, barcode_reads
            )
            observations = [
                TrackObs(d.track_id, d.bbox, d.label, d.confidence, barcodes)
                for d, barcodes in zip(tracked_detections, barcodes_by_detection)
            ]
            frame_events = engine.update(
                frame_index, media_time_seconds, frame_size, observations
            )
            for event in frame_events:
                self.session.submit(event)
                new_events.append(event)
            identifications = [
                self.identifier.identify(d.label, d.confidence, barcodes)
                for d, barcodes in zip(tracked_detections, barcodes_by_detection)
            ]
            for detection, identification in zip(tracked_detections, identifications):
                inventory.set_identity(
                    detection.track_id, identification, verified_at=media_time_seconds
                )
            rendered = annotate(
                frame,
                tracked_detections,
                identifications,
                event_config["zone"],
                orphan_barcodes,
            )
            current_rows = []
            track_to_ident = {
                detection.track_id: (
                    detection,
                    identification,
                    barcodes_by_detection[index],
                )
                for index, (detection, identification) in enumerate(
                    zip(tracked_detections, identifications)
                )
            }
            inventory_items = inventory.snapshot()["items"]
            inventory_by_track_id = {item["track_id"]: item for item in inventory_items}
            for raw_detection in all_detections:
                matches = [
                    (
                        iou(raw_detection.bbox, tracked.bbox),
                        track_id,
                        tracked,
                        identification,
                        barcodes,
                    )
                    for track_id, (
                        tracked,
                        identification,
                        barcodes,
                    ) in track_to_ident.items()
                    if tracked.label == raw_detection.label
                ]
                match = max(
                    matches, default=(0, None, None, None, []), key=lambda x: x[0]
                )
                if match[0] >= 0.3:
                    _, track_id, _, identification, barcodes = match
                else:
                    track_id, barcodes = None, []
                    identification = self.identifier.identify(
                        raw_detection.label, raw_detection.confidence, []
                    )
                    x1, y1, x2, y2 = map(int, raw_detection.bbox)
                    cv2.rectangle(rendered, (x1, y1), (x2, y2), (130, 130, 130), 1)
                    cv2.putText(
                        rendered,
                        f"{raw_detection.label} {raw_detection.confidence:.2f} (untracked)",
                        (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (190, 190, 190),
                        1,
                    )
                item = inventory_by_track_id.get(track_id, {})
                current_rows.append(
                    {
                        "track_id": track_id,
                        "label": raw_detection.label,
                        "confidence": raw_detection.confidence,
                        "bbox": raw_detection.bbox,
                        "model": getattr(
                            self.detector, "model_id", type(self.detector).__name__
                        ),
                        "barcode": ",".join(barcodes),
                        "ocr": "disabled (adapter only)",
                        "sku": identification.sku_id,
                        "identity": f"{identification.method}/{identification.status}/{identification.confidence_category}: {identification.reason}",
                        "lifecycle": item.get("lifecycle", "untracked"),
                        "candidates": identification.candidates,
                        "evidence_sources": identification.evidence_sources,
                        "detection_confidence": raw_detection.confidence,
                        "model_id": raw_detection.model_id,
                        "track_confidence": "not exposed by ByteTrack adapter",
                        "last_verified": item.get("last_verified"),
                        "misses": item.get("misses", 0),
                        "cart_trackable": raw_detection.label
                        not in self.ignored_classes,
                        "saved_class_rule": self.learned.class_to_sku.get(
                            raw_detection.label.lower()
                        ),
                    }
                )
            for orphan in orphan_barcodes:
                identification = self.identifier.identify(None, None, [orphan.value])
                current_rows.append(
                    {
                        "track_id": None,
                        "label": "orphan barcode",
                        "confidence": 0.0,
                        "bbox": orphan.bbox,
                        "model": "barcode",
                        "barcode": orphan.value,
                        "ocr": "n/a",
                        "sku": identification.sku_id,
                        "identity": f"barcode/{identification.status}: {identification.reason}",
                        "lifecycle": "barcode candidate",
                        "candidates": identification.candidates,
                        "evidence_sources": identification.evidence_sources,
                        "detection_confidence": None,
                        "model_id": "barcode",
                        "track_confidence": "n/a",
                        "last_verified": None,
                        "misses": 0,
                    }
                )
            loop_ms = _milliseconds_since(loop_start_time)
            frame_diagnostics = {
                "model": getattr(
                    self.detector, "model_id", type(self.detector).__name__
                ),
                "detections": len(all_detections),
                "tracks": len(tracked_detections),
                "fps": 1000 / loop_ms if loop_ms > 0 else 0,
                "mean_conf": (
                    sum(d.confidence for d in all_detections) / len(all_detections)
                    if all_detections
                    else 0
                ),
                "p50_conf": (
                    float(np.percentile([d.confidence for d in all_detections], 50))
                    if all_detections
                    else 0
                ),
                "p95_conf": (
                    float(np.percentile([d.confidence for d in all_detections], 95))
                    if all_detections
                    else 0
                ),
                "classes": ", ".join(
                    f"{name}:{sum(d.label == name for d in all_detections)}"
                    for name in dict.fromkeys(d.label for d in all_detections)
                )[:45]
                or "none",
                "detect_ms": latency_ms["detect"][-1],
                "track_ms": latency_ms["track"][-1],
                "barcode_ms": latency_ms["barcode"][-1],
                "barcodes": len(barcode_reads),
                "verified": sum(
                    identification.status == "RESOLVED"
                    and identification.method == "barcode"
                    for identification in identifications
                ),
                "unresolved": sum(
                    identification.status != "RESOLVED"
                    for identification in identifications
                ),
                "identity_rate": (
                    (
                        sum(
                            identification.status == "RESOLVED"
                            for identification in identifications
                        )
                        / len(identifications)
                    )
                    if identifications
                    else 0.0
                ),
                "track_rate": (
                    len(tracked_detections) / len(detections) if detections else 0.0
                ),
                "accepted_events": sum(
                    e.state == "ACCEPTED" or e.processing_status == "APPLIED"
                    for e in self.session.events.values()
                ),
                "review_events": len(self.session.pending()),
                "rejected_events": sum(
                    e.state == "REJECTED" for e in self.session.events.values()
                ),
                "present": sum(
                    item["lifecycle"] == "confirmed_present" for item in inventory_items
                ),
                "occluded": sum(
                    item["lifecycle"] == "occluded" for item in inventory_items
                ),
                "removal": sum(
                    item["lifecycle"] == "removal_candidate" for item in inventory_items
                ),
                "ocr": "not installed/enabled; adapter only",
                "yolo": "active YOLOX-ONNX on COCO-80 classes; no second YOLO model configured",
            }
            if display:
                rendered = annotate_cart_status(
                    rendered,
                    self.session,
                    self.catalog,
                    frame_events,
                    unit_price_cents,
                    frame_diagnostics,
                )
            if out_path:
                if video_writer is None:
                    Path(out_path).expanduser().parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    video_writer = cv2.VideoWriter(
                        str(out_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps / frame_stride,
                        frame_size,
                    )
                video_writer.write(rendered)
            if display and live_ui:
                state_counts = {
                    name: sum(
                        item["lifecycle"] == name
                        for item in inventory_by_track_id.values()
                    )
                    for name in (
                        "confirmed_present",
                        "occluded",
                        "removal_candidate",
                        "unresolved",
                    )
                }
                metrics_text = (
                    f"ACTIVE DETECTOR: {frame_diagnostics['model']} | generic COCO-80 labels\n"
                    f"RAW detections={len(all_detections)} | tracked={len(tracked_detections)} | classes={frame_diagnostics['classes']}\n"
                    f"Detection confidence: mean {frame_diagnostics['mean_conf']:.3f} (raw, not calibrated); per-box score is in table\n"
                    f"Detector score p50={frame_diagnostics['p50_conf']:.3f}, p95={frame_diagnostics['p95_conf']:.3f}; these are not accuracy rates\n"
                    f"Latency ms: detector {latency_ms['detect'][-1]:.1f} | tracker {latency_ms['track'][-1]:.1f} | barcode {latency_ms['barcode'][-1]:.1f} | loop {loop_ms:.1f}\n"
                    f"Identity: barcode matches={frame_diagnostics['verified']} | unresolved={frame_diagnostics['unresolved']} | OCR {frame_diagnostics['ocr']}\n"
                    f"Resolved identity ratio={frame_diagnostics['identity_rate']:.0%} | track/detection ratio={frame_diagnostics['track_rate']:.0%} (association coverage, not accuracy)\n"
                    f"Session events: accepted/applied={frame_diagnostics['accepted_events']} | needs review={frame_diagnostics['review_events']} | rejected={frame_diagnostics['rejected_events']}\n"
                    f"Lifecycle counts: {state_counts} | cart events pending review={len(self.session.pending())}\n"
                    "Model availability: COCO-SSD is in standalone browser demo; active Python model is YOLOX ONNX. No second YOLO weights or OCR engine are loaded."
                )
                bill = compute_bill(self.session, self.catalog, unit_price_cents)
                cart_lines = (
                    ", ".join(
                        f"{line['name']} ×{line['qty']}" for line in bill["lines"]
                    )
                    or "empty"
                )
                cart_summary = (
                    f"{cart_lines}   |   TOTAL {bill['subtotal_cents'] / 100:.2f} {bill['currency']} ({bill['status']})\n"
                    f"Session {self.session.session_id} | pending reviews {len(self.session.pending())} | accepted items {sum(self.session.items.values())}"
                )
                live_ui.update(rendered, current_rows, metrics_text, cart_summary)
            if display and commands:
                parts = commands.pop(0).split()
                action = parts[0].lower()
                sku = parts[1].upper() if len(parts) > 1 else ""
                if action == "q":
                    stop = True
                elif action in ("y", "n"):
                    pending = self.session.pending()
                    if pending and action == "n":
                        self.session.reject(
                            pending[0].event_id, "rejected by terminal operator"
                        )
                    elif pending:
                        self.session.confirm(
                            pending[0].event_id,
                            "confirmed by terminal operator",
                            sku_id=sku or pending[0].sku_id,
                        )
                elif action in ("add", "remove") and sku:
                    self.session.manual_event(
                        action.upper(), sku, "entered by live terminal operator"
                    )
            latency_ms["total"].append(_milliseconds_since(loop_start_time))
            processed_frame_count += 1
            if on_frame:
                on_frame(frame_index, tracked_detections, frame_events)
            if stop:
                break
        for event in engine.flush(last_frame_index, last_frame_index / source_fps):
            self.session.submit(event)
            new_events.append(event)
        if stop or (max_frames is not None and processed_frame_count >= max_frames):
            inventory.interruption()
        if video_writer:
            video_writer.release()
        if live_ui:
            live_ui.close()
        return dict(
            frames=processed_frame_count,
            events=new_events,
            latency_ms=latency_ms,
            fps_source=source_fps,
            inventory=inventory.snapshot(),
            out_path=str(out_path) if out_path else None,
        )
