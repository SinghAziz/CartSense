"""Build detector/catalogue/session/pipeline from config."""
from pathlib import Path
import yaml
from src.detection.yolox_onnx import make_detector
from src.pricing.catalog import Catalog
from src.cart.session import CartSession
from src.pipeline import Pipeline
from src.identification.learned_rules import LearnedRules

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    """Load config and anchor its file resources to the config file location."""
    config_path = Path(path) if path is not None else PROJECT_ROOT / "config" / "config.yaml"
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    base = config_path.parent.parent
    for key_path in (("detector", "model_path"), ("catalog", "path"), ("learning", "path")):
        section, key = key_path
        value = cfg.get(section, {}).get(key)
        if value:
            resource = Path(value).expanduser()
            cfg[section][key] = str(resource if resource.is_absolute() else (base / resource).resolve())
    return cfg

def build(cfg=None, detector=None, decoder=None, session=None):
    cfg = cfg or load_config()
    cat = Catalog(cfg["catalog"]["path"]); sess = session or CartSession(cat)
    if detector is None:
        detector = make_detector(cfg["detector"])
    learned = LearnedRules(cfg.get("learning", {}).get("path", "data/local_learned_rules.json"))
    return Pipeline(cfg, detector, cat, sess, decoder, learned), cat, sess
