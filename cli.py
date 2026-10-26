"""Image, video, webcam, and synthetic demo commands for CartSense."""

import argparse
import os
from pathlib import Path

import cv2

from src.setup import build, load_config
from src.input.sources import read_image
from src.pricing.catalog import Catalog
from src.pricing.billing import compute_bill
from src.evaluation.metrics import latency_stats


def print_cart_summary(session, catalog, unit_price_cents=None):
    bill = compute_bill(session, catalog, unit_price_cents)
    print(f"\nCART (session {session.session_id}) status={bill['status']}")
    for line in bill["lines"]:
        print(
            f"  {line['qty']} x {line['name']} ({line['sku_id']}) "
            f"@ {line['unit_cents']} = {line['line_cents']}"
        )
    print(
        f"  SUBTOTAL: {bill['subtotal_cents'] / 100:.2f} "
        f"{bill['currency']} (payment NOT processed)"
    )
    for warning in bill["warnings"]:
        print("  ! " + warning)
    for event in session.events.values():
        print(
            f"  event {event.event_id} {event.event_type:7} {event.state:12} "
            f"sku={event.sku_id} track={event.track_id} :: {event.evidence}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run the CartSense pipeline.")
    parser.add_argument("mode", choices=["image", "video", "live", "demo"])
    parser.add_argument("source", nargs="?")
    parser.add_argument("--out", help="Path for annotated output")
    parser.add_argument("--config", help="Path to a YAML configuration file")
    args = parser.parse_args()
    if args.mode in ("image", "video") and not args.source:
        parser.error(f"{args.mode} mode requires an input path or source")

    os.makedirs("outputs", exist_ok=True)
    if args.mode == "demo":
        from src.input.synthetic import make_transfer_video, StubDetector

        config = load_config(args.config)
        catalog = Catalog(config["catalog"]["path"])
        barcode = catalog.get("SKU-1001").barcodes[0]
        boxes = make_transfer_video("outputs/synthetic_add.mp4", barcode, "add")
        pipeline, catalog, session = build(config, detector=StubDetector(boxes))
        print(
            "SYNTHETIC DEMO (stub detector; barcode decoding is real) - not evidence of real-world accuracy"
        )
        pipeline.process_video(
            "outputs/synthetic_add.mp4", "outputs/synthetic_add_annotated.mp4"
        )
        print_cart_summary(session, catalog)
        return

    pipeline, catalog, session = build(load_config(args.config))
    if args.mode == "live":
        source = args.source or "0"
        print(f"Opening live camera source {source!r}. Press Q or ESC to stop.")
        print(
            "Commands: y [SKU] + Enter, n + Enter, add SKU, remove SKU, or q + Enter."
        )
        live_price_cents = 1000
        print("Live pricing: $10.00 per accepted product; REMOVE subtracts $10.00.")
        pipeline.process_video(
            source,
            display=True,
            unit_price_cents=live_price_cents,
            terminal_controls=True,
        )
        print_cart_summary(session, catalog, live_price_cents)
        return

    if args.mode == "image":
        result = pipeline.process_image(read_image(args.source))
        output_path = args.out or "outputs/annotated.jpg"
        Path(output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), result["annotated"]):
            raise RuntimeError(f"could not write annotated image to {output_path!r}")
        for detection, identification in zip(
            result["detections"], result["identifications"]
        ):
            print(
                f"{detection.label:14} conf={detection.confidence:.2f} "
                f"bbox={[round(x) for x in detection.bbox]} -> "
                f"{identification.status} {identification.sku_id} "
                f"({identification.reason})"
            )
        for barcode, identification in result["orphan_barcodes"]:
            print(
                f"barcode {barcode.value} ({barcode.symbology}) -> "
                f"{identification.status} {identification.sku_id} "
                f"({identification.reason})"
            )
        latency = {
            name: round(milliseconds, 1)
            for name, milliseconds in result["latency_ms"].items()
        }
        print("latency ms:", latency, "| saved", output_path)
    else:
        output_path = args.out or "outputs/annotated.mp4"
        result = pipeline.process_video(args.source, output_path)
        total_latency = latency_stats(result["latency_ms"]["total"])
        print(
            f"{result['frames']} frames; latency: {total_latency} "
            f"| saved {output_path}"
        )
        print_cart_summary(session, catalog)


if __name__ == "__main__":
    main()
