"""Combine barcode + detector class + catalogue into an Identification. No fused/calibrated confidence."""
from src.common import Identification
from src.pricing.catalog import norm_barcode

class Identifier:
    def __init__(self, catalog, allow_class_only_sku=False, conflict_on_class_mismatch=True):
        self.cat, self.allow_class_only = catalog, allow_class_only_sku
        self.conflict_on_mismatch = conflict_on_class_mismatch

    def identify(self, label, det_conf, barcodes):
        base = dict(detector_label=label, detector_confidence=det_conf)
        codes = {norm_barcode(b) for b in (barcodes or []) if b}
        if len(codes) > 1:
            return Identification("CONFLICT", None, "barcode", f"multiple barcodes: {sorted(codes)}", **base)
        if codes:
            code = next(iter(codes)); p = self.cat.by_barcode(code)
            if p is None:
                return Identification("UNRESOLVED", None, "barcode", f"barcode {code} not in catalogue", barcode=code, **base)
            if self.conflict_on_mismatch and p.detector_classes and label and label not in p.detector_classes:
                return Identification("CONFLICT", None, "barcode",
                                      f"barcode maps to {p.sku_id} but detector says '{label}'", barcode=code, **base)
            return Identification("RESOLVED", p.sku_id, "barcode", "barcode matched catalogue", barcode=code, **base)
        cands = self.cat.skus_for_class(label) if label else []
        if self.allow_class_only and len(cands) == 1:
            return Identification("RESOLVED", cands[0].sku_id, "class_only",
                                  "single catalogue SKU for class (generic - NOT SKU-verified)", **base)
        return Identification("UNRESOLVED", None, "none",
                              f"no barcode; class '{label}' is generic ({len(cands)} candidate SKUs)", **base)
