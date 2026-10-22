"""Barcode decoding behind an interface. Default backend: zxing-cpp (ZXing C++ port, Apache-2.0). Fallback: OpenCV cv2.barcode.
Google ML Kit is mobile-only (Android/iOS) and cannot be imported in Python - it is the intended on-device backend."""
import cv2
from src.common import BarcodeRead

class BarcodeDecoder:
    def decode(self, bgr) -> list: raise NotImplementedError

class NullBarcodeDecoder(BarcodeDecoder):
    def decode(self, bgr): return []

class OpenCVBarcodeDecoder(BarcodeDecoder):
    def __init__(self):
        self.det = cv2.barcode.BarcodeDetector()

    def decode(self, bgr):
        try:
            ok, infos, types, pts = self.det.detectAndDecodeWithType(bgr)
        except cv2.error:
            return []
        out = []
        if not ok or pts is None:
            return out
        for v, t, p in zip(infos, types, pts):
            if v:                                   # empty string == detected but not decoded
                out.append(BarcodeRead(v, str(t), (float(p[:, 0].min()), float(p[:, 1].min()),
                                                  float(p[:, 0].max()), float(p[:, 1].max()))))
        return out


class ZXingBarcodeDecoder(BarcodeDecoder):
    def __init__(self):
        import zxingcpp; self.z = zxingcpp

    def decode(self, bgr):
        out = []
        for r in self.z.read_barcodes(bgr):
            p = r.position; xs = [q.x for q in (p.top_left, p.top_right, p.bottom_right, p.bottom_left)]
            ys = [q.y for q in (p.top_left, p.top_right, p.bottom_right, p.bottom_left)]
            out.append(BarcodeRead(r.text, str(r.format), (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))))
        return out

def make_decoder(backend="zxing"):
    if backend == "none": return NullBarcodeDecoder()
    if backend == "opencv": return OpenCVBarcodeDecoder()
    try: return ZXingBarcodeDecoder()
    except ImportError: return OpenCVBarcodeDecoder()
