"""Small explicit operator feedback store for the local machine; no model training occurs."""
import json
from pathlib import Path
from src.pricing.catalog import norm_barcode


class LearnedRules:
    def __init__(self, path="data/local_learned_rules.json"):
        self.path = Path(path)
        self.barcode_to_sku = {}
        self.class_to_sku = {}
        self._load()

    def _load(self):
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text())
                self.barcode_to_sku = dict(raw.get("barcode_to_sku", {}))
                self.class_to_sku = dict(raw.get("class_to_sku", {}))
            except (OSError, ValueError):
                # Corrupt feedback is ignored; never prevent the baseline from starting.
                self.barcode_to_sku, self.class_to_sku = {}, {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps({"barcode_to_sku": self.barcode_to_sku,
                                    "class_to_sku": self.class_to_sku}, indent=2) + "\n")
        temp.replace(self.path)

    def teach_barcode(self, code, sku_id, catalog):
        code = norm_barcode(code)
        if not 8 <= len(code) <= 14 or catalog.get(sku_id) is None:
            raise ValueError("enter a plausible barcode and select a catalogue SKU")
        self.barcode_to_sku[code] = sku_id; self.save()

    def teach_class(self, label, sku_id, catalog):
        if not label or catalog.get(sku_id) is None:
            raise ValueError("select a detected class and a catalogue SKU")
        self.class_to_sku[label.strip().lower()] = sku_id; self.save()
