# pylint: disable=dangerous-default-value

"""
resnet model definitions

our models should have a standard interface
    - load from checkpoint
    - edit final layer to match num classes
    - forward passes that return a dict like this:
        {
            "logits": Tensor,
            "loss": Tensor,
            "encoding": Tensor,
        }
"""

import torch
from jsonargparse import Namespace
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class ResNet50(ResNet):
    """
    resnet50 model
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
    ):
        super(ResNet50, self).__init__(
            block=block, layers=layers, num_classes=num_classes
        )
        self.feature_size = 2048
        self.loss_fn = loss_fn

        ## initialize weights
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, targets: torch.Tensor | None = None) -> dict:

        out = {
            "logits": None,
            "loss": None,
            "encoding": None,
        }

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        logits_cls = self.fc(feature)

        out["logits"] = logits_cls
        out["encoding"] = feature

        if targets is not None:
            out["loss"] = self.loss_fn(logits_cls, targets)

        return out

    def __str__(self) -> str:
        return f"ResNet50(num_classes={self.fc.out_features}, loss_fn={self.loss_fn})"


class ResNet18(ResNet):
    """
    resnet18 model
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
    ):
        super(ResNet18, self).__init__(
            block=block, layers=layers, num_classes=num_classes
        )
        self.feature_size = 512
        self.loss_fn = loss_fn

        ## initialize weights
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, targets: torch.Tensor | None = None) -> dict:

        out = {
            "logits": None,
            "loss": None,
            "encoding": None,
        }

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        logits_cls = self.fc(feature)

        out["logits"] = logits_cls
        out["encoding"] = feature

        if targets is not None:
            out["loss"] = self.loss_fn(logits_cls, targets)

        return out

    def __str__(self) -> str:
        return f"ResNet18(num_classes={self.fc.out_features}, loss_fn={self.loss_fn})"


def _build_resnet(configs: Namespace) -> nn.Module:
    """Builds a ResNet model based on the provided configurations"""

    num_classes = configs.data.num_classes

    match configs.model.arch.lower():
        case "resnet18":
            model = ResNet18(
                loss_fn=nn.CrossEntropyLoss(),
                block=BasicBlock,
                layers=[2, 2, 2, 2],
                num_classes=num_classes,
            )
        case "resnet50":
            model = ResNet50(
                loss_fn=nn.CrossEntropyLoss(),
                block=Bottleneck,
                layers=[3, 4, 6, 3],
                num_classes=num_classes,
            )
        case _:
            raise ValueError(f"Model architecture {configs.model.arch} not supported")

    # Load checkpoint if specified
    if configs.model.checkpoint is not None:
        checkpoint = torch.load(configs.model.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model
