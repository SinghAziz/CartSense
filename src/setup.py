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
    config_path = (
        Path(path) if path is not None else PROJECT_ROOT / "config" / "config.yaml"
    )
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    resource_root = config_path.parent.parent
    for key_path in (
        ("detector", "model_path"),
        ("catalog", "path"),
        ("learning", "path"),
    ):
        section, key = key_path
        value = config.get(section, {}).get(key)
        if value:
            resource = Path(value).expanduser()
            config[section][key] = str(
                resource
                if resource.is_absolute()
                else (resource_root / resource).resolve()
            )
    return config


def build(config=None, detector=None, decoder=None, session=None):
    config = config or load_config()
    catalog = Catalog(config["catalog"]["path"])
    cart_session = session or CartSession(catalog)
    if detector is None:
        detector = make_detector(config["detector"])
    learned_rules = LearnedRules(
        config.get("learning", {}).get("path", "data/local_learned_rules.json")
    )
    pipeline = Pipeline(config, detector, catalog, cart_session, decoder, learned_rules)
    return pipeline, catalog, cart_session
