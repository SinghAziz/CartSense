"""Explainable evidence policy. Component scores remain separate and are not averaged."""
from src.identification.ocr import catalogue_text_candidates


class EvidenceFusion:
    def __init__(self, catalog):
        self.catalog = catalog

    def identify(self, label=None, detector_confidence=None, barcodes=(), ocr_text=""):
        codes = {str(x).strip() for x in barcodes if x}
        barcode_skus = {p.sku_id for code in codes if (p := self.catalog.by_barcode(code))}
        candidates = catalogue_text_candidates(ocr_text, self.catalog)
        ocr_skus = {x["sku_id"] for x in candidates if x["token_overlap"] >= 2}
        class_skus = {p.sku_id for p in self.catalog.skus_for_class(label)} if label else set()
        sources = []
        if label: sources.append({"source": "detector", "label": label, "score": detector_confidence})
        if ocr_text: sources.append({"source": "ocr", "text": ocr_text, "score": None})
        if codes: sources.append({"source": "barcode", "values": sorted(codes), "catalogue_matches": sorted(barcode_skus)})
        if len(codes) > 1 or len(barcode_skus) > 1 or (barcode_skus and ocr_skus and not barcode_skus.intersection(ocr_skus)):
            return self._result("CONFLICT", None, "conflict", sources, candidates, True)
        if barcode_skus:
            sku = next(iter(barcode_skus))
            if class_skus and sku not in class_skus:
                return self._result("CONFLICT", None, "conflict", sources, candidates, True)
            return self._result("RESOLVED", sku, "strong", sources, candidates, False)
        if len(ocr_skus) == 1 and (not class_skus or ocr_skus <= class_skus):
            return self._result("UNRESOLVED", None, "review", sources, candidates, False)
        return self._result("UNRESOLVED", None, "unknown", sources, candidates, False)

    @staticmethod
    def _result(status, sku, category, sources, candidates, conflict):
        return {"status": status, "sku_id": sku, "confidence_category": category,
                "evidence_sources": sources, "component_scores": {
                    x["source"]: x.get("score") for x in sources if x.get("score") is not None},
                "candidates": candidates, "conflict": conflict,
                "review_required": status != "RESOLVED"}
