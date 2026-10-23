"""Combine barcode + detector class + catalogue into an Identification. No fused/calibrated confidence."""
from src.common import Identification
from src.pricing.catalog import norm_barcode
from src.identification.fusion import EvidenceFusion

class Identifier:
    def __init__(self, catalog, allow_class_only_sku=False, conflict_on_class_mismatch=True,
                 learned_rules=None, ignored_classes=()):
        self.cat, self.allow_class_only = catalog, allow_class_only_sku
        self.conflict_on_mismatch = conflict_on_class_mismatch
        self.fusion = EvidenceFusion(catalog)
        self.learned = learned_rules
        self.ignored_classes = {str(x).lower() for x in ignored_classes}

    def _make(self, status, sku, method, reason, label, det_conf, barcode=None, conflict=False):
        fusion = self.fusion.identify(label, det_conf, [barcode] if barcode else [])
        if method in ("learned_class", "learned_barcode"):
            fusion["evidence_sources"].append({"source": "operator_rule", "method": method})
        return Identification(status, sku, method, reason, barcode=barcode, detector_label=label,
                              detector_confidence=det_conf,
                              confidence_category=("strong" if method == "barcode" and status == "RESOLVED" else "unknown"),
                              evidence_sources=fusion["evidence_sources"], candidates=fusion["candidates"],
                              conflict=conflict,
                              review_required=(status != "RESOLVED" or method in ("class_only", "learned_class", "learned_barcode")))

    def identify(self, label, det_conf, barcodes):
        codes = {norm_barcode(b) for b in (barcodes or []) if b}
        if len(codes) > 1:
            return self._make("CONFLICT", None, "barcode", f"multiple barcodes: {sorted(codes)}", label, det_conf, conflict=True)
        if codes:
            code = next(iter(codes)); p = self.cat.by_barcode(code)
            learned_sku = self.learned.barcode_to_sku.get(code) if self.learned else None
            if p is None and learned_sku:
                p = self.cat.get(learned_sku)
                method = "learned_barcode"
            else:
                method = "barcode"
            if p is None:
                return self._make("UNRESOLVED", None, "barcode", f"barcode {code} not in catalogue", label, det_conf, code)
            if self.conflict_on_mismatch and p.detector_classes and label and label not in p.detector_classes:
                return self._make("CONFLICT", None, "barcode", f"barcode maps to {p.sku_id} but detector says '{label}'", label, det_conf, code, True)
            return self._make("RESOLVED", p.sku_id, method,
                              "barcode matched catalogue" if method == "barcode" else "barcode matched saved operator rule",
                              label, det_conf, code)
        learned_sku = (self.learned.class_to_sku.get((label or "").lower())
                       if self.learned and (label or "").lower() not in self.ignored_classes else None)
        if learned_sku and self.cat.get(learned_sku):
            return self._make("RESOLVED", learned_sku, "learned_class",
                              "generic COCO class matched saved operator rule; verify brand/variant",
                              label, det_conf)
        cands = self.cat.skus_for_class(label) if label else []
        if self.allow_class_only and len(cands) == 1:
            return self._make("RESOLVED", cands[0].sku_id, "class_only",
                              "single catalogue SKU for class (generic - NOT SKU-verified)", label, det_conf)
        return self._make("UNRESOLVED", None, "none",
                          f"no barcode; class '{label}' is generic ({len(cands)} candidate SKUs)", label, det_conf)
