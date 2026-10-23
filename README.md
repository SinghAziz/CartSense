# CartSense ML prototype

Pipeline: image/video/webcam → YOLOX detection → ByteTrack → barcode (zxing-cpp) → identification → zone-crossing candidate events → validated, idempotent cart → running bill.

**Honest scope.** The detector is a *generic COCO* model (80 classes). It does **not** recognise grocery SKUs. A SKU is only assigned from a catalogue **barcode** match (or manual review). Catalogue data is **mock**. No payments. The PRD referenced in the brief was not attached to the build session, so requirements come from the brief only.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# weights (Apache-2.0, ~20 MB) - already in models/ if you got the zip:
curl -L -o models/yolox_tiny.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
```
On Apple Silicon `onnxruntime` will use CoreML if available (config `detector.providers`); otherwise CPU. Untested on Mac.

## Run
```bash
python cli.py image data/samples/bus.jpg          # real detector -> outputs/annotated.jpg
python cli.py demo                                # SYNTHETIC end-to-end: video -> ADD event -> cart -> bill
python cli.py video path/to/phone_clip.mp4 --out outputs/out.mp4   # also: 0 (webcam) or a phone-stream URL
python cli.py live                                      # live camera window with zones, boxes, labels, track IDs, cart, bill, and events
# In the live window: Y confirms the oldest review with a known SKU; N rejects it.
python cli.py live 0                                    # choose a specific webcam index
# In the terminal: type `y` + Enter to confirm, `n` + Enter to reject, or `q` + Enter to stop.
streamlit run app.py                              # dashboard with review/correct/reset
python -m pytest -q                               # tests
python -m src.evaluation.report                   # writes reports/report.md
```
## Demo workflow
Set the camera so the cart interior is in `events.zone` (config). Record a clip where a barcode-facing item moves from outside the zone into it → `python cli.py video clip.mp4` → ADD event ACCEPTED if barcode matches catalogue, else NEEDS_REVIEW → fix it in Streamlit → bill updates.

## Edit the catalogue
`data/catalog.json` (sku, name, price_cents, barcodes, detector_classes). No retraining needed.

## Layout
`src/detection` (YOLOX ONNX) · `src/barcode` · `src/tracking` · `src/identification` · `src/events` (state machine) · `src/cart` · `src/pricing` · `src/evaluation` · `src/input` (sources + synthetic fixtures) · `src/pipeline.py` · `config/config.yaml`.

See `reports/report.md` for measured vs synthetic vs untested results.
