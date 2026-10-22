"""python -m src.evaluation.report  -> reports/report.md  (measures latency on the sample image; runs pytest; separates measured/synthetic/untested)"""
import platform, subprocess, sys, json, time, cv2, onnxruntime as ort, importlib.metadata as md
from src.setup import build, load_config
from src.evaluation.metrics import latency_stats

def v(p):
    try: return md.version(p)
    except Exception: return "n/a"

def main():
    cfg = load_config(); pipe, cat, _ = build(cfg); img = cv2.imread("data/samples/bus.jpg")
    for _ in range(3): pipe.det.detect(img)            # warm-up
    lat = []
    for _ in range(30):
        t = time.perf_counter(); pipe.det.detect(img); lat.append((time.perf_counter() - t) * 1000)
    st = latency_stats(lat); res = pipe.process_image(img)
    pt = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    demo = subprocess.run([sys.executable, "cli.py", "demo"], capture_output=True, text=True).stdout
    md_ = f"""# CartSense prototype report
Generated on the machine that ran it: `{platform.platform()}`, Python {platform.python_version()}, {__import__('os').cpu_count()} CPU(s), ORT providers {pipe.det.providers}.

## Model
- Detector: **YOLOX-tiny** (416x416, COCO-80 pretrained, `models/yolox_tiny.onnx`), Megvii YOLOX release 0.1.1rc0, **Apache-2.0** (commercial use permitted; verify COCO-weight terms with counsel before shipping).
- Ultralytics YOLO was NOT used: not installable in the build sandbox (disk) and AGPL-3.0/enterprise licence is a commercial consideration.
- Tracker: ByteTrack via `supervision` {v('supervision')}. Barcode: `zxing-cpp` {v('zxing-cpp')} (fallback OpenCV {v('opencv-contrib-python')}). Runtime: onnxruntime {v('onnxruntime')}.

## MEASURED (real detector, real image `bus.jpg`, not a grocery scene)
- Detector-only latency, 30 runs after warm-up, **this sandbox CPU (not a Mac/phone)**: median {st['median_ms']:.1f} ms, p95 {st['p95_ms']:.1f} ms.
- Detections on the sample image: {[(d.label, round(d.confidence, 2)) for d in res['detections']]}. All remained UNRESOLVED (no SKU), as intended.
- Detection mAP, precision/recall: **not measured** - no labelled grocery evaluation set. Use `pycocotools` once one exists.

## SYNTHETIC (code-path validation only; NOT recognition accuracy)
- Automated tests: `{pt}`.
- Synthetic demo (stub detector, real barcode decode + ByteTrack + events):
```
{demo.strip()}
```

## Catalogue
{', '.join(f'{p.sku_id} ({p.name})' for p in cat.products.values())} - **MOCK data**.

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
"""
    open("reports/report.md", "w").write(md_); print(md_)
if __name__ == "__main__": main()
