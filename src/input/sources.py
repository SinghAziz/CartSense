"""Input adapters via OpenCV: image file, video file, webcam index, or phone-stream URL (e.g. IP-camera apps).
A native mobile app can later replace this adapter."""

import cv2


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def open_video(source):
    video_source = int(source) if str(source).isdigit() else str(source)
    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video source {source!r}")
    frames_per_second = capture.get(cv2.CAP_PROP_FPS) or 30.0
    return capture, frames_per_second


def iter_frames(capture, frames_per_second, stride=1, max_frames=None):
    frame_index = yielded_frames = 0
    while True:
        read_successfully, frame = capture.read()
        if not read_successfully:
            break
        if frame_index % stride == 0:
            yield frame_index, frame_index / frames_per_second, frame
            yielded_frames += 1
            if max_frames is not None and yielded_frames >= max_frames:
                break
        frame_index += 1
    capture.release()
