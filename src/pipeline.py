"""Orchestration: input -> detect -> (track) -> barcode -> identify -> events -> cart -> bill."""
import sys, threading, time, cv2, numpy as np
from src.barcode.decoder import make_decoder
from src.tracking.tracker import ByteTrackTracker
from src.identification.identifier import Identifier
from src.events.engine import EventEngine, TrackObs
from src.input.sources import open_video, iter_frames
from src.pricing.billing import compute_bill
from src.cart.inventory import CartInventory
from src.identification.learned_rules import LearnedRules
from src.tracking.geometry import iou

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
        zx1, zy1 = int(zone[0] * W), int(zone[1] * H)
        zx2, zy2 = int(zone[2] * W) - 1, int(zone[3] * H) - 1
        cv2.rectangle(out, (zx1, zy1), (zx2, zy2), (255, 160, 0), 2)
        cv2.putText(out, "CART ZONE", (zx1 + 10, max(24, zy1 + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 0), 2)
        if zx1 > 10:
            cv2.putText(out, "OUTSIDE ZONE", (10, max(24, zy1 + 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
    for d, idn in zip(dets, idents):
        x1, y1, x2, y2 = map(int, d.bbox)
        col = (0, 170, 0) if idn.status == "RESOLVED" else (0, 140, 255) if idn.status == "UNRESOLVED" else (0, 0, 220)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        txt = (f"#{d.track_id} " if d.track_id is not None else "") + f"{d.label} {d.confidence:.2f}" + (f" {idn.sku_id}" if idn.sku_id else " ?")
        cv2.putText(out, txt, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    for bc in orphans:
        cv2.rectangle(out, tuple(map(int, bc.bbox[:2])), tuple(map(int, bc.bbox[2:])), (200, 0, 200), 2)
    return out

def annotate_cart_status(img, session, catalog, events=(), unit_price_cents=None, diagnostics=None):
    """Show cart state and transparent per-frame component diagnostics on live output."""
    out = img.copy()
    bill = compute_bill(session, catalog, unit_price_cents)
    cart_text = ", ".join(
        f"{line['sku_id']} x{line['qty']}" for line in bill["lines"]
    ) or "empty"
    lines = [
        f"CART: {cart_text}",
        f"BILL: {bill['subtotal_cents'] / 100:.2f} {bill['currency']} ({bill['status']})",
        "REVIEW: Y=confirm, N=reject",
    ]
    lines.extend(
        f"{e.event_type}: {e.sku_id or 'unknown'} [{e.state}]"
        for e in events[-2:]
    )
    if bill["unresolved_events"]:
        lines.append(f"REVIEW: {len(bill['unresolved_events'])} event(s)")
    x, y = 10, 70
    box_h = 28 * len(lines) + 12
    overlay = out.copy()
    cv2.rectangle(overlay, (x - 6, y - 24), (min(out.shape[1] - 10, 520), y + box_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
    for line in lines:
        color = (0, 220, 0) if "ACCEPTED" in line else (0, 190, 255) if "REVIEW" in line or "NEEDS_REVIEW" in line else (255, 255, 255)
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
        cv2.rectangle(overlay, (dx, dy), (min(out.shape[1] - 4, dx + width), min(out.shape[0] - 4, dy + height)), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)
        for row in rows:
            cv2.putText(out, row, (dx + 10, dy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (230, 230, 230), 1, cv2.LINE_AA)
            dy += 24
    return out

class Pipeline:
    def __init__(self, cfg, detector, catalog, session, decoder=None, learned_rules=None):
        self.cfg, self.det, self.cat, self.session = cfg, detector, catalog, session
        self.decoder = decoder or make_decoder(cfg["barcode"].get("backend", "zxing") if cfg["barcode"]["enabled"] else "none")
        self.ignore = set(cfg.get("ignore_classes", []))
        cc = cfg["catalog"]
        self.learned = learned_rules or LearnedRules(cfg.get("learning", {}).get("path", "data/local_learned_rules.json"))
        self.idf = Identifier(catalog, cc["allow_class_only_sku"], cc["conflict_on_class_mismatch"],
                              self.learned, ignored_classes=self.ignore)

    def process_image(self, img):
        """Single image: detection + barcode + identification only. No cart events (no temporal evidence)."""
        t0 = time.perf_counter(); dets = self.det.detect(img); t_det = _ms(t0)
        t0 = time.perf_counter(); bcs = self.decoder.decode(img); t_bc = _ms(t0)
        per, orphans = associate(dets, bcs)
        idents = [self.idf.identify(d.label, d.confidence, b) for d, b in zip(dets, per)]
        orphan_ids = [self.idf.identify(None, None, [o.value]) for o in orphans]
        return dict(detections=dets, identifications=idents, orphan_barcodes=list(zip(orphans, orphan_ids)),
                    annotated=annotate(img, dets, idents, orphans=orphans), latency_ms=dict(detect=t_det, barcode=t_bc))

    def process_video(self, source, out_path=None, max_frames=None, on_frame=None,
                      display=False, unit_price_cents=None, terminal_controls=False):
        cap, fps = open_video(source); stride = self.cfg["video"]["frame_stride"]
        tracker = ByteTrackTracker(fps / stride)
        tc = self.cfg.get("tracker", {})
        inventory = CartInventory(tc.get("confirm_hits", 3), tc.get("removal_misses", 15))
        ev_cfg = self.cfg["events"]; engine = EventEngine(self.idf, self.session.session_id, ev_cfg)
        lat = dict(detect=[], barcode=[], track=[], total=[]); new_events, writer, n, wh, last = [], None, 0, None, 0
        stop = False
        commands = []
        live_ui = None
        current_rows = []

        def do_live_action(action, args):
            nonlocal stop
            if action == "quit":
                stop = True; return "Stopping live view…"
            row = args.get("row") or next((r for r in current_rows if r.get("track_id") == args.get("track_id")), None)
            track_id, sku_id = args.get("track_id"), args.get("sku_id")
            label = args.get("label") or (row.get("label") if row else "")
            if action == "learn_barcode":
                self.learned.teach_barcode(args.get("barcode"), sku_id, self.cat)
                if track_id is not None:
                    ident = self.idf.identify(label, row["confidence"], [args["barcode"]])
                    inventory.set_identity(track_id, ident)
                return f"Saved barcode rule for {sku_id}. It applies on future matching barcode reads."
            if action == "learn_class":
                if row is None: raise ValueError("Select a detected object row first.")
                if label.lower() in {x.lower() for x in self.ignore}:
                    raise ValueError(f"'{label}' is ignored for cart tracking and cannot be learned as a product class.")
                self.learned.teach_class(label, sku_id, self.cat)
                if track_id is not None:
                    inventory.set_identity(track_id, self.idf.identify(label, row["confidence"], []))
                return f"Saved {label} → {sku_id}. Generic class rules can confuse brands and variants."
            if action == "forget_class":
                if not label: raise ValueError("Select an object class first.")
                self.learned.class_to_sku.pop(label.lower(), None); self.learned.save()
                return f"Removed the saved rule for {label}."
            if action == "confirm":
                pending = [e for e in self.session.pending() if (row is None and track_id is None) or e.track_id == track_id]
                if not pending: raise ValueError("No pending review event for the selected track.")
                ev = pending[0]
                return "Review applied." if self.session.confirm(ev.event_id, "confirmed in live inspection console", sku_id=sku_id or ev.sku_id) else "Review remains unresolved. Check the selected SKU and cart quantity."
            if action == "reject":
                pending = [e for e in self.session.pending() if (row is None and track_id is None) or e.track_id == track_id]
                if not pending: raise ValueError("No pending review event for the selected track.")
                self.session.reject(pending[0].event_id, "rejected in live inspection console"); return "Selected event rejected."
            if action in ("add", "remove"):
                if not row: raise ValueError("Select a detected object first.")
                if not sku_id: raise ValueError("Choose a catalogue SKU first.")
                self.session.manual_event(action.upper(), sku_id, f"operator selected track {track_id} ({label}) in live console")
                if track_id is not None:
                    inventory.set_identity(track_id, self.idf.identify(label, row["confidence"], []))
                return f"Manual {action.upper()} recorded for {sku_id}."
            if action == "manual_add":
                if not sku_id: raise ValueError("Choose a catalogue SKU first.")
                self.session.manual_event("ADD", sku_id, "operator manually added SKU from live image review")
                return f"Manual ADD recorded for {sku_id}."
            return ""

        if display:
            from src.live_ui import LiveConsole
            live_ui = LiveConsole(self.cat, do_live_action)
        if display and terminal_controls:
            def read_commands():
                print("Terminal controls remain available: y [SKU], n, add SKU, remove SKU, or q.")
                for line in sys.stdin:
                    command = line.strip().lower()
                    if command: commands.append(command)
                    if command == "q" or command.startswith("q "): break
            threading.Thread(target=read_commands, daemon=True).start()
        for idx, t, frame in iter_frames(cap, fps, stride, max_frames):
            T0 = time.perf_counter(); H, W = frame.shape[:2]; wh = (W, H); last = idx
            t0 = time.perf_counter(); all_dets = self.det.detect(frame)
            dets = [d for d in all_dets if d.label not in self.ignore]; lat["detect"].append(_ms(t0))
            t0 = time.perf_counter(); tr = tracker.update(dets); lat["track"].append(_ms(t0))
            inventory.update(tr, now=t)
            t0 = time.perf_counter(); bcs = self.decoder.decode(frame); lat["barcode"].append(_ms(t0))
            per, orphans = associate(tr, bcs)
            obs = [TrackObs(d.track_id, d.bbox, d.label, d.confidence, b) for d, b in zip(tr, per)]
            evs = engine.update(idx, t, wh, obs)
            for e in evs:
                self.session.submit(e); new_events.append(e)
            idents = [self.idf.identify(d.label, d.confidence, b) for d, b in zip(tr, per)]
            for d, ident in zip(tr, idents):
                inventory.set_identity(d.track_id, ident, verified_at=t)
            rendered = annotate(frame, tr, idents, ev_cfg["zone"], orphans)
            current_rows = []
            track_to_ident = {d.track_id: (d, ident, per[i]) for i, (d, ident) in enumerate(zip(tr, idents))}
            inv_by_id = {x["track_id"]: x for x in inventory.snapshot()["items"]}
            matched_raw = set()
            for raw_index, raw in enumerate(all_dets):
                matches = [(iou(raw.bbox, td.bbox), tid, td, ident, codes)
                           for tid, (td, ident, codes) in track_to_ident.items() if td.label == raw.label]
                match = max(matches, default=(0, None, None, None, []), key=lambda x: x[0])
                if match[0] >= 0.3:
                    _, tid, td, ident, codes = match; matched_raw.add(tid)
                else:
                    tid, codes = None, []
                    ident = self.idf.identify(raw.label, raw.confidence, [])
                    x1, y1, x2, y2 = map(int, raw.bbox)
                    cv2.rectangle(rendered, (x1, y1), (x2, y2), (130, 130, 130), 1)
                    cv2.putText(rendered, f"{raw.label} {raw.confidence:.2f} (untracked)",
                                (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1)
                item = inv_by_id.get(tid, {})
                current_rows.append({"track_id": tid, "label": raw.label, "confidence": raw.confidence,
                                     "bbox": raw.bbox, "model": getattr(self.det, "model_id", type(self.det).__name__),
                                     "barcode": ",".join(codes), "ocr": "disabled (adapter only)",
                                     "sku": ident.sku_id, "identity": f"{ident.method}/{ident.status}/{ident.confidence_category}: {ident.reason}",
                                     "lifecycle": item.get("lifecycle", "untracked"),
                                     "candidates": ident.candidates, "evidence_sources": ident.evidence_sources,
                                     "detection_confidence": raw.confidence, "model_id": raw.model_id,
                                     "track_confidence": "not exposed by ByteTrack adapter",
                                     "last_verified": item.get("last_verified"), "misses": item.get("misses", 0),
                                     "cart_trackable": raw.label not in self.ignore,
                                     "saved_class_rule": self.learned.class_to_sku.get(raw.label.lower())})
            for orphan in orphans:
                ident = self.idf.identify(None, None, [orphan.value])
                current_rows.append({"track_id": None, "label": "orphan barcode", "confidence": 0.0,
                                     "bbox": orphan.bbox, "model": "barcode", "barcode": orphan.value,
                                     "ocr": "n/a", "sku": ident.sku_id, "identity": f"barcode/{ident.status}: {ident.reason}",
                                     "lifecycle": "barcode candidate", "candidates": ident.candidates, "evidence_sources": ident.evidence_sources,
                                     "detection_confidence": None, "model_id": "barcode", "track_confidence": "n/a",
                                     "last_verified": None, "misses": 0})
            loop_ms = _ms(T0)
            inv = inventory.snapshot()["items"]
            frame_diagnostics = {
                "model": getattr(self.det, "model_id", type(self.det).__name__),
                "detections": len(all_dets), "tracks": len(tr), "fps": 1000 / loop_ms if loop_ms > 0 else 0,
                "mean_conf": sum(d.confidence for d in all_dets) / len(all_dets) if all_dets else 0,
                "p50_conf": float(np.percentile([d.confidence for d in all_dets], 50)) if all_dets else 0,
                "p95_conf": float(np.percentile([d.confidence for d in all_dets], 95)) if all_dets else 0,
                "classes": ", ".join(f"{name}:{sum(d.label == name for d in all_dets)}" for name in dict.fromkeys(d.label for d in all_dets))[:45] or "none",
                "detect_ms": lat["detect"][-1], "track_ms": lat["track"][-1],
                "barcode_ms": lat["barcode"][-1], "barcodes": len(bcs),
                "verified": sum(i.status == "RESOLVED" and i.method == "barcode" for i in idents),
                "unresolved": sum(i.status != "RESOLVED" for i in idents),
                "identity_rate": (sum(i.status == "RESOLVED" for i in idents) / len(idents)) if idents else 0.0,
                "track_rate": (len(tr) / len(dets)) if dets else 0.0,
                "accepted_events": sum(e.state == "ACCEPTED" or e.processing_status == "APPLIED" for e in self.session.events.values()),
                "review_events": len(self.session.pending()),
                "rejected_events": sum(e.state == "REJECTED" for e in self.session.events.values()),
                "present": sum(x["lifecycle"] == "confirmed_present" for x in inv),
                "occluded": sum(x["lifecycle"] == "occluded" for x in inv),
                "removal": sum(x["lifecycle"] == "removal_candidate" for x in inv),
                "ocr": "not installed/enabled; adapter only",
                "yolo": "active YOLOX-ONNX on COCO-80 classes; no second YOLO model configured",
            }
            if display:
                rendered = annotate_cart_status(
                    rendered, self.session, self.cat, evs, unit_price_cents, frame_diagnostics
                )
            if out_path:
                if writer is None:
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, wh)
                writer.write(rendered)
            if display and live_ui:
                state_counts = {name: sum(x["lifecycle"] == name for x in inv_by_id.values())
                                for name in ("confirmed_present", "occluded", "removal_candidate", "unresolved")}
                metrics_text = (f"ACTIVE DETECTOR: {frame_diagnostics['model']} | generic COCO-80 labels\n"
                                f"RAW detections={len(all_dets)} | tracked={len(tr)} | classes={frame_diagnostics['classes']}\n"
                                f"Detection confidence: mean {frame_diagnostics['mean_conf']:.3f} (raw, not calibrated); per-box score is in table\n"
                                f"Detector score p50={frame_diagnostics['p50_conf']:.3f}, p95={frame_diagnostics['p95_conf']:.3f}; these are not accuracy rates\n"
                                f"Latency ms: detector {lat['detect'][-1]:.1f} | tracker {lat['track'][-1]:.1f} | barcode {lat['barcode'][-1]:.1f} | loop {loop_ms:.1f}\n"
                                f"Identity: barcode matches={frame_diagnostics['verified']} | unresolved={frame_diagnostics['unresolved']} | OCR {frame_diagnostics['ocr']}\n"
                                f"Resolved identity ratio={frame_diagnostics['identity_rate']:.0%} | track/detection ratio={frame_diagnostics['track_rate']:.0%} (association coverage, not accuracy)\n"
                                f"Session events: accepted/applied={frame_diagnostics['accepted_events']} | needs review={frame_diagnostics['review_events']} | rejected={frame_diagnostics['rejected_events']}\n"
                                f"Lifecycle counts: {state_counts} | cart events pending review={len(self.session.pending())}\n"
                                "Model availability: COCO-SSD is in standalone browser demo; active Python model is YOLOX ONNX. No second YOLO weights or OCR engine are loaded.")
                bill = compute_bill(self.session, self.cat, unit_price_cents)
                cart_lines = ", ".join(f"{line['name']} ×{line['qty']}" for line in bill["lines"]) or "empty"
                cart_summary = (f"{cart_lines}   |   TOTAL {bill['subtotal_cents'] / 100:.2f} {bill['currency']} ({bill['status']})\n"
                                f"Session {self.session.session_id} | pending reviews {len(self.session.pending())} | accepted items {sum(self.session.items.values())}")
                live_ui.update(rendered, current_rows, metrics_text, cart_summary)
            if display and commands:
                parts = commands.pop(0).split()
                action = parts[0].lower()
                sku = parts[1].upper() if len(parts) > 1 else ""
                if action == "q": stop = True
                elif action in ("y", "n"):
                    pending = self.session.pending()
                    if pending and action == "n": self.session.reject(pending[0].event_id, "rejected by terminal operator")
                    elif pending:
                        self.session.confirm(pending[0].event_id, "confirmed by terminal operator", sku_id=sku or pending[0].sku_id)
                elif action in ("add", "remove") and sku:
                    self.session.manual_event(action.upper(), sku, "entered by live terminal operator")
            lat["total"].append(_ms(T0)); n += 1
            if on_frame: on_frame(idx, tr, evs)
            if stop:
                break
        for e in engine.flush(last, last / fps):
            self.session.submit(e); new_events.append(e)
        if stop or (max_frames is not None and n >= max_frames):
            inventory.interruption()
        if writer: writer.release()
        if live_ui: live_ui.close()
        return dict(frames=n, events=new_events, latency_ms=lat, fps_source=fps,
                    inventory=inventory.snapshot(), out_path=str(out_path) if out_path else None)
