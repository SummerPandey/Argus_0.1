"""Just-enough stand-ins for the `nanoowl` package (and PIL, when it
isn't installed) so nanoowl_monitor can be imported and driven outside
the Jetson container. Shared by test_async_owl.py and
test_monitor_loop.py; install() is idempotent and never shadows a real
install."""

import sys
import types


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class Tree:
    """Mirrors the one nanoowl_monitor behavior that matters off-device:
    detection_name() reads tree.labels by index."""

    def __init__(self, labels):
        self.labels = labels

    @classmethod
    def from_prompt(cls, prompt):
        return cls([part.strip() for part in prompt.strip("[]").split(",")])


class OwlPredictor:
    def __init__(self, image_encoder_engine=None):
        self.image_encoder_engine = image_encoder_engine


class TreePredictor:
    def __init__(self, owl_predictor=None):
        self.owl_predictor = owl_predictor

    def encode_clip_text(self, tree):
        return "clip-encodings"

    def encode_owl_text(self, tree):
        return "owl-encodings"


def install():
    if "nanoowl" not in sys.modules:
        _module("nanoowl")
        _module("nanoowl.tree", Tree=Tree)
        _module("nanoowl.tree_predictor", TreePredictor=TreePredictor)
        _module("nanoowl.owl_predictor", OwlPredictor=OwlPredictor)
    try:
        import PIL.Image  # noqa: F401  (real PIL, when installed)
    except ImportError:
        pil = _module("PIL")
        pil.Image = _module("PIL.Image", fromarray=lambda array: array)
