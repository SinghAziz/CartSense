"""Build detector/catalogue/session/pipeline from config."""
import yaml
from src.detection.yolox_onnx import YoloxOnnxDetector
from src.pricing.catalog import Catalog
from src.cart.session import CartSession
from src.pipeline import Pipeline

def load_config(path="config/config.yaml"):
    return yaml.safe_load(open(path))

def build(cfg=None, detector=None, decoder=None, session=None):
    cfg = cfg or load_config()
    cat = Catalog(cfg["catalog"]["path"]); sess = session or CartSession(cat)
    if detector is None:
        d = cfg["detector"]
        detector = YoloxOnnxDetector(d["model_path"], d["input_size"], d["conf_threshold"], d["nms_threshold"], d["providers"])
    return Pipeline(cfg, detector, cat, sess, decoder), cat, sess
