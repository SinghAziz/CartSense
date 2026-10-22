"""Input adapters via OpenCV: image file, video file, webcam index, or phone-stream URL (e.g. IP-camera apps).
A native mobile app can later replace this adapter."""
import cv2

def read_image(path):
    img = cv2.imread(str(path))
    if img is None: raise FileNotFoundError(path)
    return img

def open_video(source):
    src = int(source) if str(source).isdigit() else str(source)     # "0" -> webcam
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError(f"cannot open video source {source!r}")
    return cap, (cap.get(cv2.CAP_PROP_FPS) or 30.0)

def iter_frames(cap, fps, stride=1, max_frames=None):
    i = n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % stride == 0:
            yield i, i / fps, f
            n += 1
            if max_frames and n >= max_frames: break
        i += 1
    cap.release()
