"""Selective OCR adapters. OCR text is evidence only; it never independently commits a SKU."""
from dataclasses import dataclass
import re
import unicodedata


@dataclass
class OCRResult:
    text: str
    confidence: float | None
    status: str  # READ | UNREADABLE | UNAVAILABLE | ERROR


def normalize_text(value):
    value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", value))


class OCRAdapter:
    def read(self, crop):
        return OCRResult("", None, "UNAVAILABLE")


class EasyOCRAdapter(OCRAdapter):
    """Optional EasyOCR backend; instantiated only when enabled and installed."""
    def __init__(self, languages=("en",)):
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("OCR unavailable: install the optional easyocr package") from exc
        self.reader = easyocr.Reader(list(languages), gpu=False, verbose=False)

    def read(self, crop):
        try:
            rows = self.reader.readtext(crop, detail=1)
            text = " ".join(row[1] for row in rows).strip()
            confs = [float(row[2]) for row in rows]
            return OCRResult(text, (sum(confs) / len(confs)) if confs else None,
                             "READ" if text else "UNREADABLE")
        except Exception:
            return OCRResult("", None, "ERROR")


def catalogue_text_candidates(text, catalog):
    """Return normalized name/category matches for review; no score is called probability."""
    needle = normalize_text(text)
    if not needle:
        return []
    found = []
    for product in catalog.products.values():
        haystack = normalize_text(f"{product.name} {product.category}")
        tokens = set(needle.split())
        overlap = len(tokens & set(haystack.split()))
        if overlap:
            found.append({"sku_id": product.sku_id, "token_overlap": overlap})
    return sorted(found, key=lambda x: x["token_overlap"], reverse=True)
