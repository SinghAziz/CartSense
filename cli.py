"""python cli.py image <path> | video <path|0|url> [--out f.mp4] | live [camera] | demo"""
import argparse, cv2, os
from pathlib import Path
from src.setup import build, load_config
from src.input.sources import read_image
from src.pricing.billing import compute_bill
from src.evaluation.metrics import latency_stats

def show(sess, cat, unit_price_cents=None):
    b = compute_bill(sess, cat, unit_price_cents)
    print(f"\nCART (session {sess.session_id}) status={b['status']}")
    for l in b["lines"]: print(f"  {l['qty']} x {l['name']} ({l['sku_id']}) @ {l['unit_cents']} = {l['line_cents']}")
    print(f"  SUBTOTAL: {b['subtotal_cents']/100:.2f} {b['currency']} (payment NOT processed)")
    for w in b["warnings"]: print("  ! " + w)
    for e in sess.events.values(): print(f"  event {e.event_id} {e.event_type:7} {e.state:12} sku={e.sku_id} track={e.track_id} :: {e.evidence}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["image", "video", "live", "demo"]); ap.add_argument("src", nargs="?")
    ap.add_argument("--out"); ap.add_argument("--config"); a = ap.parse_args()
    if a.mode in ("image", "video") and not a.src:
        ap.error(f"{a.mode} mode requires an input path or source")
    os.makedirs("outputs", exist_ok=True)
    if a.mode == "demo":
        from src.input.synthetic import make_transfer_video, StubDetector
        cat0 = build(load_config(a.config), detector=object())[1]
        code = cat0.get("SKU-1001").barcodes[0]
        boxes = make_transfer_video("outputs/synthetic_add.mp4", code, "add")
        pipe, cat, sess = build(load_config(a.config), detector=StubDetector(boxes))
        print("SYNTHETIC DEMO (stub detector; barcode decoding is real) - not evidence of real-world accuracy")
        r = pipe.process_video("outputs/synthetic_add.mp4", "outputs/synthetic_add_annotated.mp4"); show(sess, cat); return
    pipe, cat, sess = build(load_config(a.config))
    if a.mode == "live":
        source = a.src or "0"
        print(f"Opening live camera source {source!r}. Press Q or ESC in the camera window to stop.")
        print("Commands: y [SKU] + Enter, n + Enter, add SKU, remove SKU, or q + Enter.")
        live_price_cents = 1000
        print("Live pricing: $10.00 per accepted product; REMOVE subtracts $10.00.")
        pipe.process_video(source, display=True, unit_price_cents=live_price_cents, terminal_controls=True)
        show(sess, cat, live_price_cents)
        return
    if a.mode == "image":
        r = pipe.process_image(read_image(a.src)); out = a.out or "outputs/annotated.jpg"
        Path(out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), r["annotated"]):
            raise RuntimeError(f"could not write annotated image to {out!r}")
        for d, i in zip(r["detections"], r["identifications"]):
            print(f"{d.label:14} conf={d.confidence:.2f} bbox={[round(x) for x in d.bbox]} -> {i.status} {i.sku_id} ({i.reason})")
        for bc, i in r["orphan_barcodes"]: print(f"barcode {bc.value} ({bc.symbology}) -> {i.status} {i.sku_id} ({i.reason})")
        print("latency ms:", {k: round(v, 1) for k, v in r["latency_ms"].items()}, "| saved", out)
    else:
        out = a.out or "outputs/annotated.mp4"; r = pipe.process_video(a.src, out)
        print(f"{r['frames']} frames; latency:", {k: latency_stats(v) for k, v in r["latency_ms"].items() if k == 'total'}, "| saved", out); show(sess, cat)
if __name__ == "__main__": main()
