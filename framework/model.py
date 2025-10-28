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

    # import additional models and add them here as needed

    # from XXX import _build_xxx_model
    # case "xxx":
    #     return _build_xxx_model(configs)

    return model
