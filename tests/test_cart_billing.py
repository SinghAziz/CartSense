"""Cart / catalogue / billing / identification unit tests (pure logic; no ML involved)."""
import pytest
from tests.conftest import mk
from src.pricing.billing import compute_bill
from src.identification.identifier import Identifier

def test_01_supported_product_in_catalogue(cat):
    p = cat.get("SKU-1001"); assert p.price_cents == 120 and cat.is_mock

def test_02_add_known_sku(sess):
    assert sess.submit(mk()) and sess.items == {"SKU-1001": 1}

def test_03_valid_removal_reduces_quantity(sess):
    sess.submit(mk()); sess.submit(mk()); sess.submit(mk(typ="REMOVE"))
    assert sess.items == {"SKU-1001": 1}

def test_03b_removal_beyond_cart_is_held_not_negative(sess):
    sess.submit(mk(typ="REMOVE")); assert sess.items == {} and len(sess.pending()) == 1

def test_04_replay_same_event_not_applied_twice(sess):
    e = mk(eid="E-same"); sess.submit(e); sess.submit(e); sess._apply(e)
    assert sess.items == {"SKU-1001": 1}

def test_04b_duplicate_dedupe_key_ignored(sess):
    sess.submit(mk(key="k1")); sess.submit(mk(key="k1")); assert sess.items["SKU-1001"] == 1

def test_05_unresolved_never_silently_in_cart(sess, cat):
    sess.submit(mk(sku=None, state="NEEDS_REVIEW"))
    assert sess.items == {} and compute_bill(sess, cat)["status"] == "PROVISIONAL"
    with pytest.raises(ValueError): sess._apply(mk(sku=None, state="NEEDS_REVIEW"))

def test_06_bill_matches_quantities_and_prices(sess, cat):
    for s in ["SKU-1001", "SKU-1001", "SKU-2001"]: sess.submit(mk(sku=s))
    b = compute_bill(sess, cat)
    assert b["subtotal_cents"] == 2 * 120 + 35 and b["status"] == "CONFIRMED_UNPAID" and not b["payment_processed"]

def test_07_unknown_barcode_not_fabricated(cat):
    i = Identifier(cat).identify("bottle", 0.9, ["4006381333931"])
    assert i.status == "UNRESOLVED" and i.sku_id is None

def test_07b_class_only_is_not_a_sku(cat):
    assert Identifier(cat).identify("banana", 0.9, []).status == "UNRESOLVED"

def test_07c_barcode_class_conflict(cat):
    i = Identifier(cat).identify("banana", 0.9, [cat.get("SKU-1001").barcodes[0]]); assert i.status == "CONFLICT"

def test_08_rejected_event_does_not_change_cart(sess):
    sess.submit(mk(state="REJECTED")); assert sess.items == {}

def test_09_manual_confirm_and_correct_with_audit(sess):
    e = mk(sku=None, state="NEEDS_REVIEW"); sess.submit(e)
    assert sess.confirm(e.event_id, "operator saw water bottle", sku_id="SKU-1002")
    assert sess.items == {"SKU-1002": 1} and e.resolution == "MANUAL"
    a = [x for x in sess.audit if x["action"] == "MANUAL_CONFIRM"][0]
    assert a["reason"] and a["new_sku"] == "SKU-1002"

def test_09b_manual_reject_leaves_audit(sess):
    e = mk(state="NEEDS_REVIEW"); sess.submit(e); sess.reject(e.event_id, "false positive")
    assert sess.items == {} and any(x["action"] == "MANUAL_REJECT" for x in sess.audit)

def test_09c_unknown_sku_correction_refused(sess):
    e = mk(state="NEEDS_REVIEW"); sess.submit(e)
    with pytest.raises(ValueError): sess.confirm(e.event_id, "x", sku_id="SKU-NOPE")

def test_10_reset_clears_active_keeps_history(sess):
    sess.submit(mk()); old = sess.session_id; sess.reset()
    assert sess.items == {} and sess.events == {} and sess.session_id != old
    assert sess.history[0]["session_id"] == old and sess.history[0]["items"] == {"SKU-1001": 1}
