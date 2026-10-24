"""SYNTHETIC fixtures (EAN-13 renderer + moving-product video). For tests/demos only - NOT real-world evidence."""
import numpy as np, cv2
from pathlib import Path
from src.common import Detection

_L = ["0001101","0011001","0010011","0111101","0100011","0110001","0101111","0111011","0110111","0001011"]
_G = ["0100111","0110011","0011011","0100001","0011101","0111001","0000101","0010001","0001001","0010111"]
_R = ["".join("1" if c == "0" else "0" for c in s) for s in _L]
_P = ["LLLLLL","LLGLGG","LLGGLG","LLGGGL","LGLLGG","LGGLLG","LGGGLL","LGLGLG","LGLGGL","LGGLGL"]

def ean13_image(code, module=3, height=140, quiet=10):
    assert len(code) == 13
    par = _P[int(code[0])]
    bits = "101" + "".join((_L if p == "L" else _G)[int(d)] for p, d in zip(par, code[1:7])) + "01010" \
           + "".join(_R[int(d)] for d in code[7:]) + "101"
    row = np.array([0 if b == "1" else 255 for b in bits], np.uint8).repeat(module)
    row = np.pad(row, quiet * module, constant_values=255)
    return np.tile(row, (height, 1))[:, :, None].repeat(3, 2)

class StubDetector:
    """Returns pre-defined boxes frame by frame (stands in for the real detector in synthetic tests)."""
    def __init__(self, boxes, label="bottle", class_id=39, conf=0.9):
        self.boxes, self.i, self.label, self.cid, self.conf = boxes, 0, label, class_id, conf
    def detect(self, bgr):
        b = self.boxes[self.i] if self.i < len(self.boxes) else None; self.i += 1
        return [Detection(b, self.cid, self.label, self.conf)] if b else []

def make_transfer_video(path, code, direction="add", n=60, size=(960, 540), fps=30):
    """Product with a barcode slides from outside the right-half cart zone into it (add) or out of it (remove)."""
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    W, H = size; bc = ean13_image(code)
    ph, pw = bc.shape[0] + 40, bc.shape[1] + 40
    prod = np.full((ph, pw, 3), 200, np.uint8); prod[20:20 + bc.shape[0], 20:20 + bc.shape[1]] = bc
    xs = np.linspace(pw / 2 + 5, W * 0.75, n)
    xs = xs if direction == "add" else xs[::-1]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size); boxes = []
    for cx in xs:
        f = np.full((H, W, 3), 255, np.uint8)
        x0, y0 = int(cx - pw / 2), int(H / 2 - ph / 2)
        f[y0:y0 + ph, x0:x0 + pw] = prod; vw.write(f)
        boxes.append((float(x0), float(y0), float(x0 + pw), float(y0 + ph)))
    vw.release()
    return boxes
