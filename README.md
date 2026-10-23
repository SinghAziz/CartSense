# CartSense ML prototype

CartSense has two retained interfaces: the Python video pipeline/dashboard and an older standalone browser webcam demo (`index.html`). The Python path is the primary modular pipeline:

`frame → YOLOX ONNX detection → ByteTrack → barcode association/identification → zone-crossing event → audited cart session → bill`

The detector recognizes generic COCO classes, **not grocery SKUs**. Catalogue data and prices are mock data. A catalogue-matched barcode can identify a SKU; generic class, OCR, and unknown barcode evidence do not silently create a SKU. Unresolved events need operator review.

## Setup and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py image data/samples/bus.jpg
python cli.py demo
python cli.py video path/to/clip.mp4 --out outputs/out.mp4
python cli.py live 0
streamlit run app.py
python -m pytest -q
```

YOLOX Tiny ONNX weights are included in `models/`. Nothing downloads model weights at runtime. The default Python pipeline uses `yolox` configured in `config/config.yaml`; selecting `backend: yolo` requires user-supplied compatible ONNX weights and the optional `ultralytics` package. It raises an explicit unavailable error if either is missing. Do not load both detectors except when deliberately benchmarking.

Barcode decoding uses zxing-cpp, with OpenCV as fallback. Unknown codes remain unresolved and are distinguishable from catalogue matches. OCR has an optional EasyOCR adapter (`src/identification/ocr.py`), disabled by default; install EasyOCR separately to enable it. OCR is exposed as review evidence only and is not currently invoked by the frame pipeline. OCR scores are separate from detection confidence.

## Architecture

- `src/detection`: detector interface implementations: YOLOX ONNX baseline and optional YOLO adapter.
- `src/common.py`: shared observations, identification results, barcode and event schemas.
- `src/tracking`: ByteTrack adapter and detector-independent geometry.
- `src/cart/inventory.py`: session-local tracked-instance lifecycle and diagnostics. Occlusion and prolonged absence preserve cart state; absence becomes a removal candidate, not a committed removal.
- `src/identification`: catalogue rules, explainable fusion, text normalization and optional OCR adapter.
- `src/barcode`: independent ZXing/OpenCV decoding.
- `src/events`: zone-crossing transition evidence and event decisions.
- `src/cart/session.py`: authoritative, idempotent cart events, operator review, audit, session snapshots.
- `src/evaluation`: metric helpers. Detection mAP is left unavailable without labelled ground truth.

The event engine requires a fixed camera and configured cart zone. Items already in the zone at track start do not create an ADD event because transfer evidence is absent. ByteTrack IDs are active-lifecycle IDs, not durable re-identification; if tracking resets, physical reconciliation requires review. The in-memory instance inventory is diagnostic and deliberately does not override the audited cart event ledger.

`index.html` remains an independent TensorFlow.js COCO-SSD prototype with browser barcode scanning. Its placeholder barcode catalogue is mock data and it is not wired to the Python inventory/event ledger. Use the Python dashboard/pipeline for audited cart state.

### Live inspection console

`python cli.py live 0` opens a Tk inspection console with the video feed, a prominent cart summary and total, and a selectable detection table. It shows model/class, raw detection score, box coordinates, associated barcode, OCR availability, SKU hypothesis, and lifecycle. Click a box or select its row; double-click a row for the detailed evidence record. To add a visible item, choose its catalogue SKU in the top selector and press **Add selected image item**. **Add SKU manually** adds the chosen SKU without a selected detection. Use **Remove selected** or **Confirm selected review** for the selected track. The console reports per-frame detector/tracker/barcode latency, FPS, score distribution, identity/tracking coverage, event counts, and model availability. These detector scores are not accuracy rates or calibrated probabilities.

Select a barcode result or object row, choose a catalogue SKU, then use **Save barcode rule** or **Learn class rule**. Rules persist locally in `data/local_learned_rules.json` and are ignored by Git. A barcode rule can resolve future reads of that barcode. A class rule maps a generic class such as `bottle` to one SKU, so it cannot distinguish brands or variants; events from learned rules still require review. Ignored classes such as `person` cannot be learned as products. **Forget class rule** removes a selected class mapping. Unknown COCO classes do not trigger model training.

The Python live view currently runs YOLOX ONNX with COCO-80 labels. The separate browser COCO-SSD, an additional YOLO model/weights, and OCR engine are not all active in this Python view. The console labels those components accurately; it does not fabricate comparison boxes when their model resources are absent. OCR remains adapter-only and disabled by default.

## Thresholds and evaluation

Detector, tracking, event and inventory thresholds live in `config/config.yaml`. They are provisional engineering defaults, not statistically calibrated. Synthetic tests exercise state and event logic only; they do not measure model accuracy. Metrics helpers accept labelled examples for detection precision/recall, exact SKU results, barcode outcomes, cart events and latency. No real grocery-labelled evaluation set is included, so no real-world accuracy or mAP is reported.

Edit `data/catalog.json` to replace mock SKUs, barcodes and prices. No backend, database, cloud service or automatic footage collection is used.
