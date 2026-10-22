import pytest, uuid
from src.pricing.catalog import Catalog
from src.cart.session import CartSession
from src.common import Event

@pytest.fixture
def cat(): return Catalog("data/catalog.json")

@pytest.fixture
def sess(cat): return CartSession(cat)

def mk(sku="SKU-1001", typ="ADD", state="ACCEPTED", key=None, eid=None):
    return Event(event_id=eid or "E-" + uuid.uuid4().hex[:6], session_id="t", timestamp="x", media_time_s=0.0,
                 event_type=typ, state=state, sku_id=sku, track_id=1, verification="BARCODE_VERIFIED",
                 detector_confidence=0.9, evidence="synthetic test", dedupe_key=key or uuid.uuid4().hex)
