"""Build detector/catalogue/session/pipeline from config."""
import yaml
from src.detection.yolox_onnx import make_detector
from src.pricing.catalog import Catalog
from src.cart.session import CartSession
from src.pipeline import Pipeline
from src.identification.learned_rules import LearnedRules

def load_config(path="config/config.yaml"):
    return yaml.safe_load(open(path))

def build(cfg=None, detector=None, decoder=None, session=None):
    cfg = cfg or load_config()
    cat = Catalog(cfg["catalog"]["path"]); sess = session or CartSession(cat)
    if detector is None:
        detector = make_detector(cfg["detector"])
    learned = LearnedRules(cfg.get("learning", {}).get("path", "data/local_learned_rules.json"))
    return Pipeline(cfg, detector, cat, sess, decoder, learned), cat, sess
