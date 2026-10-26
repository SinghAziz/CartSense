"""Local JSON product catalogue (mock data). Edit data/catalog.json - no model change needed."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


def norm_barcode(v: str) -> str:
    v = "".join(ch for ch in str(v) if ch.isdigit())
    return (
        v[1:] if len(v) == 13 and v.startswith("0") else v
    )  # EAN-13 with leading 0 == UPC-A


@dataclass
class Product:
    sku_id: str
    name: str
    price_cents: int
    category: str = ""
    barcodes: list = field(default_factory=list)
    detector_classes: list = field(default_factory=list)


class Catalog:
    def __init__(self, path=None, products=None, meta=None):
        self.path = Path(path) if path is not None else None
        self.meta = meta or {}
        self.products = {}
        if self.path:
            with self.path.open(encoding="utf-8") as catalog_file:
                catalog_data = json.load(catalog_file)
            self.meta = catalog_data.get("_meta", {})
            products = [Product(**product) for product in catalog_data["products"]]
        for p in products or []:
            self.add(p)

    def add(self, p: Product):
        self.products[p.sku_id] = p

    def get(self, sku_id):
        return self.products.get(sku_id)

    def by_barcode(self, code):
        c = norm_barcode(code)
        for p in self.products.values():
            if c in {norm_barcode(b) for b in p.barcodes}:
                return p
        return None

    def skus_for_class(self, label):
        return [p for p in self.products.values() if label in p.detector_classes]

    def save(self, path=None):
        output_path = Path(path) if path is not None else self.path
        if output_path is None:
            raise ValueError(
                "a path is required to save a catalogue without a source file"
            )
        with output_path.open("w", encoding="utf-8") as catalog_file:
            json.dump(
                {
                    "_meta": self.meta,
                    "products": [asdict(product) for product in self.products.values()],
                },
                catalog_file,
                indent=2,
            )
            catalog_file.write("\n")

    @property
    def is_mock(self):
        return bool(self.meta.get("mock_data"))
