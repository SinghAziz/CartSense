"""Synthetic state and evidence tests; these are not model accuracy benchmarks."""
from types import SimpleNamespace
from src.cart.inventory import CartInventory, Lifecycle
from src.identification.fusion import EvidenceFusion
from src.identification.ocr import OCRResult, normalize_text
from src.evaluation.metrics import detection_metrics, identification_metrics
from src.common import Detection


def obs(track=1, label="bottle", box=(0, 0, 10, 10), conf=.9):
    return SimpleNamespace(track_id=track, label=label, bbox=box, conf=conf)


def test_repeated_detection_confirms_one_instance():
    inv = CartInventory(confirm_hits=3)
    for _ in range(5): inv.update([obs()])
    assert len(inv.instances) == 1
    assert inv.instances[1].lifecycle == Lifecycle.CONFIRMED_PRESENT


def test_identical_products_are_separate_instances():
    inv = CartInventory(confirm_hits=2)
    inv.update([obs(1), obs(2)])
    inv.update([obs(1), obs(2)])
    assert len(inv.instances) == 2


def test_occlusion_and_short_disappearance_preserve_instance():
    inv = CartInventory(confirm_hits=2, removal_misses=4)
    inv.update([obs()]); inv.update([obs()]); inv.update([])
    assert inv.instances[1].lifecycle == Lifecycle.OCCLUDED
    assert inv.instances[1].misses == 1
    inv.update([obs()])
    assert inv.instances[1].lifecycle == Lifecycle.CONFIRMED_PRESENT


def test_prolonged_absence_is_candidate_not_removal():
    inv = CartInventory(confirm_hits=1, removal_misses=3)
    inv.update([obs()]); inv.update([]); inv.update([]); inv.update([])
    assert inv.instances[1].lifecycle == Lifecycle.REMOVAL_CANDIDATE


def test_interruption_preserves_inventory_and_requests_review():
    inv = CartInventory(confirm_hits=1); inv.update([obs()]); inv.interruption()
    assert inv.instances[1].lifecycle == Lifecycle.CONFIRMED_PRESENT
    assert inv.snapshot()["needs_verification"]


def test_fusion_barcode_conflict_and_unknown_barcode(cat):
    fusion = EvidenceFusion(cat)
    code = cat.get("SKU-1001").barcodes[0]
    assert fusion.identify("banana", .99, [code])["conflict"]
    result = fusion.identify("bottle", .99, ["9999999999999"])
    assert result["sku_id"] is None and result["review_required"]


def test_ocr_is_normalized_and_unreadable_is_safe():
    assert normalize_text("Café, 500ml!") == "cafe 500ml"
    assert OCRResult("", None, "UNREADABLE").text == ""


def test_missing_yolo_weights_has_clear_unavailable_state(tmp_path):
    from src.detection.yolox_onnx import YoloDetectorAdapter
    import pytest
    with pytest.raises(FileNotFoundError, match="YOLO model unavailable"):
        YoloDetectorAdapter(tmp_path / "missing.onnx")


def test_evaluation_metrics_count_unmatched_and_unknown():
    d = detection_metrics([{"bbox": (0, 0, 10, 10), "label": "bottle"}],
                          [{"bbox": (0, 0, 10, 10), "label": "bottle"},
                           {"bbox": (20, 20, 30, 30), "label": "cup"}])
    assert (d["tp"], d["fp"], d["fn"], d["map"]) == (1, 0, 1, None)
    assert identification_metrics(["SKU-1", None], ["SKU-1", "SKU-2"])["exact_sku_accuracy"] == .5


def test_observation_schema_includes_time_model_and_id():
    d = Detection((1, 2, 3, 4), 1, "bottle", .8, model_id="fixture")
    assert d.observation_id and d.timestamp and d.model_id == "fixture"


def test_explicit_barcode_and_class_rules_persist_locally(tmp_path, cat):
    from src.identification.learned_rules import LearnedRules
    from src.identification.identifier import Identifier
    path = tmp_path / "rules.json"
    rules = LearnedRules(path)
    sku = "SKU-1001"
    rules.teach_barcode("123456789012", sku, cat)
    rules.teach_class("bottle", sku, cat)
    loaded = LearnedRules(path)
    barcode = Identifier(cat, learned_rules=loaded).identify("bottle", .8, ["123456789012"])
    visual = Identifier(cat, learned_rules=loaded).identify("bottle", .8, [])
    assert barcode.sku_id == sku and barcode.method == "learned_barcode"
    assert visual.sku_id == sku and visual.method == "learned_class" and visual.review_required


def test_ignored_classes_cannot_use_learned_sku_rule(tmp_path, cat):
    from src.identification.learned_rules import LearnedRules
    from src.identification.identifier import Identifier
    rules = LearnedRules(tmp_path / "rules.json")
    rules.teach_class("person", "SKU-1001", cat)
    result = Identifier(cat, learned_rules=rules, ignored_classes=("person",)).identify("person", .99, [])
    assert result.sku_id is None and result.status == "UNRESOLVED"
