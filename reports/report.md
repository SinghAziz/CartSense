# CartSense prototype report
Generated on the machine that ran it: `Linux-6.18.44-fc-v140-x86_64-with-glibc2.39`, Python 3.12.3, 1 CPU(s), ORT providers ['CPUExecutionProvider'].

## Model
- Detector: **YOLOX-tiny** (416x416, COCO-80 pretrained, `models/yolox_tiny.onnx`), Megvii YOLOX release 0.1.1rc0, **Apache-2.0** (commercial use permitted; verify COCO-weight terms with counsel before shipping).
- Ultralytics YOLO was NOT used: not installable in the build sandbox (disk) and AGPL-3.0/enterprise licence is a commercial consideration.
- Tracker: ByteTrack via `supervision` 0.25.1. Barcode: `zxing-cpp` 3.1.1 (fallback OpenCV 4.13.0.92). Runtime: onnxruntime 1.24.4.

## MEASURED (real detector, real image `bus.jpg`, not a grocery scene)
- Detector-only latency, 30 runs after warm-up, **this sandbox CPU (not a Mac/phone)**: median 69.7 ms, p95 85.0 ms.
- Detections on the sample image: [('bus', 0.95), ('person', 0.88), ('person', 0.88), ('person', 0.85)]. All remained UNRESOLVED (no SKU), as intended.
- Detection mAP, precision/recall: **not measured** - no labelled grocery evaluation set. Use `pycocotools` once one exists.

## SYNTHETIC (code-path validation only; NOT recognition accuracy)
- Automated tests: `28 passed in 2.02s`.
- Synthetic demo (stub detector, real barcode decode + ByteTrack + events):
```
SYNTHETIC DEMO (stub detector; barcode decoding is real) - not evidence of real-world accuracy

CART (session S-542a8376) status=CONFIRMED_UNPAID
  1 x Mock Mineral Water 500ml (SKU-1001) @ 120 = 120
  SUBTOTAL: 1.20 USD (payment NOT processed)
  ! Catalogue prices are MOCK sample data
  event E-c568c00c ADD     ACCEPTED     sku=SKU-1001 track=1 :: entered cart zone; track hits=35; class=bottle; id=RESOLVED/barcode: barcode matched catalogue
```

## Catalogue
SKU-1001 (Mock Mineral Water 500ml), SKU-1002 (Mock Sparkling Water 1L), SKU-2001 (Mock Banana (each)), SKU-2002 (Mock Apple (each)), SKU-2003 (Mock Orange (each)), SKU-3001 (Mock Ceramic Mug) - **MOCK data**.

## Implemented with existing libraries
ONNX Runtime + YOLOX weights (detection), supervision ByteTrack (tracking), zxing-cpp/OpenCV (barcodes), OpenCV (I/O, drawing, NMS), Streamlit (UI), pytest.

## Custom CartSense code (no suitable off-the-shelf equivalent)
Catalogue, identification rules, zone-crossing event engine + state machine, idempotent audited cart, billing, manual-correction workflow, metrics helpers, synthetic fixtures.

## Stubs / not validated
- Grocery SKU recognition from vision (generic COCO classes only; SKU comes from barcode or manual review).
- Real shopping-cart footage, real-world event accuracy, cart-state accuracy: **untested**.
- Phone-stream input (`cli.py video <url>`) and webcam: supported by OpenCV capture, **not tested here**. CoreML/Apple-Silicon acceleration: configured, **not tested**. Streamlit UI: launched headlessly only.
- On-device mobile inference and ML Kit: not started.
- Cross-track dedup (same item re-tracked under a new ID) is NOT implemented.

## Known failure cases
- Barcode must be visible and large enough; OpenCV backend failed on frame-sized images where zxing succeeded (see tests).
- Zone model assumes a fixed camera; moving camera = invalid events.
- Items starting inside the zone never generate ADD; their later REMOVE is held for review.
