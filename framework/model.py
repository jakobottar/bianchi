"""
model definitions and model building stuff
"""

from jsonargparse import Namespace
from torch import nn

from framework.resnet import _build_resnet


def build_model(configs: Namespace) -> nn.Module:
    """Builds a model based on the provided configurations"""

    if "resnet" in configs.model.arch.lower():
        model = _build_resnet(configs)
    else:
        raise ValueError(f"Model architecture {configs.model.arch} not supported")

    # ========================================
    # TEMPLATE: Adding new model architectures
    # ========================================
    # To add a new model architecture:
    #
    # 1. Create your model class in a separate file (e.g., framework/efficientnet.py)
    #    - Follow the interface pattern from resnet.py templates
    #    - Implement: __init__, forward, __str__, _initialize_weights
    #    - Include feature_size attribute
    #    - Return dict from forward: {"logits", "loss", "encoding"}
    #
    # 2. Create a builder function (e.g., _build_efficientnet)
    #    - Follow the pattern from _build_template_model in resnet.py
    #    - Handle configs.model.checkpoint loading
    #    - Return the initialized model
    #
    # 3. Import your builder function:
    #    from framework.your_model import _build_your_model
    #
    # 4. Add a new condition here:
    #    elif "efficientnet" in configs.model.arch.lower():
    #        model = _build_efficientnet(configs)
    #    elif "vit" in configs.model.arch.lower():
    #        model = _build_vision_transformer(configs)
    #    elif "mobilenet" in configs.model.arch.lower():
    #        model = _build_mobilenet(configs)
    #
    # Example imports (uncomment and modify as needed):
    # from framework.efficientnet import _build_efficientnet
    # from framework.vision_transformer import _build_vision_transformer
    # from framework.mobilenet import _build_mobilenet
    #
    # Example conditions (uncomment and modify as needed):
    # elif "efficientnet" in configs.model.arch.lower():
    #     model = _build_efficientnet(configs)
    # elif "vit" in configs.model.arch.lower() or "transformer" in configs.model.arch.lower():
    #     model = _build_vision_transformer(configs)
    # elif "mobilenet" in configs.model.arch.lower():
    #     model = _build_mobilenet(configs)

    return model
