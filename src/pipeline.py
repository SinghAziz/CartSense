"""Orchestration: input -> detect -> (track) -> barcode -> identify -> events -> cart -> bill."""
import sys, threading, time, cv2, numpy as np
from src.barcode.decoder import make_decoder
from src.tracking.tracker import ByteTrackTracker
from src.identification.identifier import Identifier
from src.events.engine import EventEngine, TrackObs
from src.input.sources import open_video, iter_frames
from src.pricing.billing import compute_bill

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

def annotate_cart_status(img, session, catalog, events=(), unit_price_cents=None):
    """Overlay the live cart state and newly emitted events on a rendered frame."""
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

    def process_video(self, source, out_path=None, max_frames=None, on_frame=None,
                      display=False, unit_price_cents=None, terminal_controls=False):
        cap, fps = open_video(source); stride = self.cfg["video"]["frame_stride"]
        tracker = ByteTrackTracker(fps / stride)
        ev_cfg = self.cfg["events"]; engine = EventEngine(self.idf, self.session.session_id, ev_cfg)
        lat = dict(detect=[], barcode=[], track=[], total=[]); new_events, writer, n, wh, last = [], None, 0, None, 0
        stop = False
        commands = []
        control_root = control_status = None
        if display and terminal_controls:
            try:
                import tkinter as tk
            except ImportError:
                tk = None
            if tk is not None:
                try:
                    control_root = tk.Tk()
                    control_root.title("CartSense review controls")
                    control_status = tk.StringVar(value="No review pending")
                    tk.Label(control_root, textvariable=control_status, width=42).pack(padx=12, pady=8)
                    buttons = tk.Frame(control_root)
                    buttons.pack(pady=(0, 10))
                    tk.Button(buttons, text="YES - Confirm ADD", width=18,
                              command=lambda: commands.append("confirm_add")).pack(side=tk.LEFT, padx=4)
                    tk.Button(buttons, text="YES - Confirm REMOVE", width=21,
                              command=lambda: commands.append("confirm_remove")).pack(side=tk.LEFT, padx=4)
                    tk.Button(buttons, text="NO - Reject", width=16,
                              command=lambda: commands.append("n")).pack(side=tk.LEFT, padx=4)
                    tk.Label(control_root, text="Close this window or press Q to stop").pack(pady=(0, 8))
                    control_root.protocol("WM_DELETE_WINDOW", lambda: commands.append("q"))
                except tk.TclError:
                    control_root = None

            def read_commands():
                print("Commands: y [SKU] + Enter, n + Enter, add SKU, remove SKU, or q + Enter.")
                for line in sys.stdin:
                    command = line.strip().lower()
                    if command:
                        commands.append(command)
                    if command == "q" or command.startswith("q "):
                        break
            threading.Thread(target=read_commands, daemon=True).start()
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
            rendered = annotate(frame, tr, idents, ev_cfg["zone"], orphans)
            if display:
                rendered = annotate_cart_status(
                    rendered, self.session, self.cat, evs, unit_price_cents
                )
            if out_path:
                if writer is None:
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, wh)
                writer.write(rendered)
            if display:
                if control_root is not None:
                    pending = self.session.pending()
                    control_status.set(
                        f"Review: {pending[0].event_id} ({pending[0].event_type})"
                        if pending else "No review pending"
                    )
                    control_root.update_idletasks()
                    control_root.update()
                cv2.imshow("CartSense live camera (press Q or ESC to stop)", rendered)
                key = cv2.waitKey(1) & 0xFF
                command = commands.pop(0) if commands else None
                if key in (ord("y"), ord("Y")):
                    command = "y"
                elif key in (ord("n"), ord("N")):
                    command = "n"
                elif key in (ord("q"), 27):
                    command = "q"
                parts = command.split() if command else []
                action = parts[0].lower() if parts else None
                requested_sku = parts[1].upper() if len(parts) > 1 else None
                if action in ("add", "remove") and requested_sku:
                    try:
                        event = self.session.manual_event(
                            action.upper(),
                            requested_sku,
                            "entered by live operator",
                        )
                        print(f"Manual {action.upper()}: {requested_sku}")
                    except (KeyError, ValueError) as exc:
                        print(f"Could not process {command!r}: {exc}")
                elif action in ("y", "confirm_add", "confirm_remove"):
                    pending = self.session.pending()
                    if pending:
                        event = pending[0]
                        sku_id = requested_sku or event.sku_id
                        if not sku_id and self.cat.products:
                            sku_id = next(iter(self.cat.products))
                            print(f"No SKU detected; confirming as default {sku_id}.")
                        event_type = (
                            "ADD" if action == "confirm_add"
                            else "REMOVE" if action == "confirm_remove"
                            else event.event_type
                        )
                        if event_type not in ("ADD", "REMOVE"):
                            print(
                                f"Cannot confirm {event.event_id}: choose "
                                "Confirm ADD or Confirm REMOVE."
                            )
                        elif sku_id:
                            try:
                                applied = self.session.confirm(
                                    event.event_id,
                                    "confirmed by live operator",
                                    sku_id=sku_id,
                                    event_type=event_type,
                                )
                            except ValueError as exc:
                                print(f"Could not confirm {event.event_id}: {exc}")
                            else:
                                if applied:
                                    print(f"Confirmed {event.event_id}: {event_type} {sku_id}")
                                else:
                                    print(
                                        f"Not applied: {event.event_id} {event_type} "
                                        f"{sku_id} remains under review."
                                    )
                        else:
                            print(
                                f"Cannot confirm {event.event_id}: "
                                "no SKU identified; use `y SKU-1001`."
                            )
                elif action == "n":
                    pending = self.session.pending()
                    if pending:
                        event = pending[0]
                        self.session.reject(
                            event.event_id,
                            "rejected by live operator",
                        )
                        print(f"Rejected {event.event_id}")
                if action == "q":
                    stop = True
            lat["total"].append(_ms(T0)); n += 1
            if on_frame: on_frame(idx, tr, evs)
            if stop:
                break
        for e in engine.flush(last, last / fps):
            self.session.submit(e); new_events.append(e)
        if writer: writer.release()
        if display: cv2.destroyAllWindows()
        if control_root is not None:
            control_root.destroy()
        return dict(frames=n, events=new_events, latency_ms=lat, fps_source=fps, out_path=str(out_path) if out_path else None)
