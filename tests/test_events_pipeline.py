"""SYNTHETIC tests of tracking/event logic. They validate code paths, NOT real-world recognition accuracy."""
import os, pytest
from src.events.engine import EventEngine, TrackObs
from src.identification.identifier import Identifier
from src.input.synthetic import ean13_image, make_transfer_video, StubDetector
from src.barcode.decoder import ZXingBarcodeDecoder, OpenCVBarcodeDecoder
from src.setup import build, load_config
from src.evaluation.metrics import event_metrics, cart_accuracy

CFG = dict(zone=[0.5, 0, 1, 1], min_track_hits=3, min_frames_confirm=3, max_lost_frames=5,
           reject_below_detector_conf=0.35, auto_accept_requires_barcode=True)
WH = (1000, 500)

def run(cat, xs, barcode=None, conf=0.9, label="bottle", start=0):
    eng = EventEngine(Identifier(cat), "S", CFG); evs = []
    for i, x in enumerate(xs):
        bc = [barcode] if barcode and i % 2 == 0 else []
        evs += eng.update(start + i, i / 30, WH, [TrackObs(7, (x - 20, 200, x + 20, 260), label, conf, bc)])
    return evs, eng

def test_add_event_barcode_accepted(cat):
    evs, _ = run(cat, [100, 200, 300, 400, 600, 700, 800, 800, 800, 800], cat.get("SKU-1001").barcodes[0])
    assert [e.event_type for e in evs] == ["ADD"] and evs[0].state == "ACCEPTED" and evs[0].sku_id == "SKU-1001"
    assert evs[0].state_history[:3] == ["OBSERVED", "TRACKED", "CANDIDATE_ADD"]

def test_no_event_for_item_already_inside(cat):
    evs, _ = run(cat, [700] * 20, cat.get("SKU-1001").barcodes[0]); assert evs == []

def test_remove_after_inside(cat):
    evs, _ = run(cat, [700] * 5 + [400] * 6, cat.get("SKU-1001").barcodes[0])
    assert [e.event_type for e in evs] == ["REMOVE"]

def test_no_barcode_needs_review_not_accepted(cat):
    evs, _ = run(cat, [100] * 5 + [700] * 5); assert evs[0].state == "NEEDS_REVIEW" and evs[0].sku_id is None

def test_low_conf_rejected(cat):
    evs, _ = run(cat, [100] * 5 + [700] * 5, cat.get("SKU-1001").barcodes[0], conf=0.31); assert evs[0].state == "REJECTED"

def test_repeated_frames_single_event(cat):
    evs, _ = run(cat, [100] * 5 + [700] * 60, cat.get("SKU-1001").barcodes[0]); assert len(evs) == 1

def test_unknown_when_track_lost_mid_transfer(cat):
    evs, eng = run(cat, [100] * 5 + [700] * 2, cat.get("SKU-1001").barcodes[0])
    evs += eng.flush(99, 3.0)
    assert [e.event_type for e in evs] == ["UNKNOWN"] and evs[0].state == "NEEDS_REVIEW"

@pytest.mark.parametrize("dec", [ZXingBarcodeDecoder, OpenCVBarcodeDecoder])
def test_barcode_decoders_on_synthetic_ean13(cat, dec):
    import numpy as np
    code = cat.get("SKU-1001").barcodes[0]
    bc = ean13_image(code); canvas = np.full((bc.shape[0] + 600, bc.shape[1] + 600, 3), 255, np.uint8)
    canvas[300:300 + bc.shape[0], 300:300 + bc.shape[1]] = bc          # OpenCV backend needs lots of surrounding context
    assert [b.value for b in dec().decode(canvas)] == [code]

def test_zxing_reads_barcode_in_frame_sized_image(cat):
    import numpy as np
    code = cat.get("SKU-1001").barcodes[0]; bc = ean13_image(code)
    f = np.full((540, 960, 3), 255, np.uint8); f[200:340, 300:300 + bc.shape[1]] = bc
    assert [b.value for b in ZXingBarcodeDecoder().decode(f)] == [code]

@pytest.mark.parametrize("direction", ["add"])
def test_pipeline_synthetic_video_end_to_end(tmp_path, cat, direction):
    code = cat.get("SKU-1001").barcodes[0]
    boxes = make_transfer_video(tmp_path / "v.mp4", code, direction)
    pipe, cat2, sess = build(load_config(), detector=StubDetector(boxes))
    r = pipe.process_video(str(tmp_path / "v.mp4"), str(tmp_path / "o.mp4"))
    assert sess.items == {"SKU-1001": 1} and os.path.getsize(tmp_path / "o.mp4") > 0
    assert cart_accuracy(sess.items, {"SKU-1001": 1})["exact_match"]
    assert event_metrics([dict(type="ADD", sku="SKU-1001", t=e.media_time_s) for e in r["events"]],
                         [dict(type="ADD", sku="SKU-1001", t=0.8)], tol_s=1.0)["tp"] == 1

@pytest.mark.skipif(not os.path.exists("models/yolox_tiny.onnx") or not os.path.exists("data/samples/bus.jpg"), reason="weights/sample missing")
def test_real_detector_smoke_on_sample_image():
    import cv2
    pipe, _, _ = build(load_config()); r = pipe.process_image(cv2.imread("data/samples/bus.jpg"))
    labels = {d.label for d in r["detections"]}
    assert "bus" in labels and "person" in labels                       # structural/sanity check only, 1 image
    assert all(i.sku_id is None for i in r["identifications"])          # generic COCO classes never become SKUs
