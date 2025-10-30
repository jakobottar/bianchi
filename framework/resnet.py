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
        # load imagenet pretrained weights if config == 'imagenet'
        if configs.model.checkpoint == "imagenet":
            checkpoint = torch.hub.load_state_dict_from_url(
                (
                    "https://download.pytorch.org/models/resnet50-0676ba61.pth"
                    if configs.model.arch.lower() == "resnet50"
                    else "https://download.pytorch.org/models/resnet18-f37072fd.pth"
                ),
                progress=True,
            )
            # Remove fc layer weights to avoid size mismatch
            checkpoint.pop("fc.weight", None)
            checkpoint.pop("fc.bias", None)
            model.load_state_dict(checkpoint, strict=False)

        else:
            checkpoint = torch.load(configs.model.checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model


# ========================================
# TEMPLATE MODEL CLASSES
# ========================================
# Copy and modify these templates when adding new model architectures
# Follow the same interface pattern as ResNet18/ResNet50


class ModelTemplate(nn.Module):
    """
    TEMPLATE: Base template for new model architectures

    All models should follow this interface pattern to ensure consistency
    with the training framework.

    Required interface:
        - __init__(self, loss_fn, num_classes=1000, **kwargs)
        - forward(self, x, targets=None) -> dict
        - __str__(self) -> str
        - feature_size: int attribute

    Args:
        loss_fn (nn.Module): Loss function for training (e.g., nn.CrossEntropyLoss())
        num_classes (int): Number of output classes
        **kwargs: Additional architecture-specific parameters
    """

    def __init__(self, loss_fn: nn.Module, num_classes: int = 1000, **kwargs):
        super(ModelTemplate, self).__init__()

        # Store essential attributes
        self.loss_fn = loss_fn
        self.num_classes = num_classes

        # Define your model architecture here
        # Example layers (replace with your architecture):
        # self.backbone = nn.Sequential(...)
        # self.classifier = nn.Linear(feature_dim, num_classes)

        # Set feature_size - dimension of features before final classifier
        # This is used for downstream tasks, transfer learning, etc.
        self.feature_size = 512  # Replace with actual feature dimension

        # Initialize weights
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        """
        Initialize model weights

        Customize this method based on your architecture's needs.
        Common initialization schemes:
        - Xavier/Glorot for general layers
        - Kaiming/He for ReLU networks
        - Zero initialization for batch norm bias
        """
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, targets: torch.Tensor | None = None) -> dict:
        """
        Forward pass through the model

        Args:
            x (torch.Tensor): Input tensor (typically images)
            targets (torch.Tensor, optional): Ground truth labels for loss computation

        Returns:
            dict: Dictionary containing:
                - "logits": torch.Tensor - Raw model outputs (pre-softmax)
                - "loss": torch.Tensor or None - Computed loss if targets provided
                - "encoding": torch.Tensor - Feature representation before classifier
        """

        # Forward pass through your architecture
        # Example (replace with your actual architecture):
        # features = self.backbone(x)
        # features_flat = features.view(features.size(0), -1)  # Flatten if needed
        # logits = self.classifier(features_flat)

        # For template, return dummy outputs
        batch_size = x.size(0)
        features = torch.zeros(
            batch_size, self.feature_size
        )  # Replace with actual features
        logits = torch.zeros(batch_size, self.num_classes)  # Replace with actual logits

        # Prepare output dictionary
        out = {
            "logits": logits,
            "loss": None,
            "encoding": features,  # Feature representation before final layer
        }

        # Compute loss if targets are provided
        if targets is not None:
            out["loss"] = self.loss_fn(logits, targets)

        return out

    def __str__(self) -> str:
        """String representation of the model"""
        return f"ModelTemplate(num_classes={self.num_classes}, loss_fn={self.loss_fn})"


class CNNTemplate(nn.Module):
    """
    TEMPLATE: For CNN-based architectures

    This template provides a structure for convolutional neural networks.
    Modify the architecture blocks as needed for your specific design.

    Args:
        loss_fn (nn.Module): Loss function
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
        hidden_dims (list): List of hidden dimensions for conv layers
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        num_classes: int = 1000,
        input_channels: int = 3,
        hidden_dims: list = [64, 128, 256, 512],
    ):
        super(CNNTemplate, self).__init__()

        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims

        # Build convolutional layers
        # Example architecture - modify as needed
        layers = []
        in_channels = input_channels

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = hidden_dim

        self.features = nn.Sequential(*layers)

        # Global average pooling to handle variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature size is the last conv layer's output dimension
        self.feature_size = hidden_dims[-1]

        # Classifier
        self.classifier = nn.Linear(self.feature_size, num_classes)

        # Initialize weights
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        """Initialize weights using standard schemes"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, targets: torch.Tensor | None = None) -> dict:
        """Forward pass through CNN"""

        # Feature extraction
        features = self.features(x)  # Shape: (B, C, H, W)
        pooled = self.global_pool(features)  # Shape: (B, C, 1, 1)
        features_flat = pooled.view(pooled.size(0), -1)  # Shape: (B, C)

        # Classification
        logits = self.classifier(features_flat)

        out = {
            "logits": logits,
            "loss": None,
            "encoding": features_flat,
        }

        if targets is not None:
            out["loss"] = self.loss_fn(logits, targets)

        return out

    def __str__(self) -> str:
        return (
            f"CNNTemplate(num_classes={self.num_classes}, "
            f"input_channels={self.input_channels}, "
            f"hidden_dims={self.hidden_dims}, loss_fn={self.loss_fn})"
        )


class TransformerTemplate(nn.Module):
    """
    TEMPLATE: For Vision Transformer architectures

    This template provides a basic structure for transformer-based models.
    Modify as needed for your specific transformer design.

    Args:
        loss_fn (nn.Module): Loss function
        num_classes (int): Number of output classes
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        patch_size (int): Size of image patches
        image_size (int): Input image size (assumed square)
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        patch_size: int = 16,
        image_size: int = 224,
    ):
        super(TransformerTemplate, self).__init__()

        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Feature size is embedding dimension
        self.feature_size = embed_dim

        # Initialize weights
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        """Initialize transformer weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, targets: torch.Tensor | None = None) -> dict:
        """Forward pass through transformer"""

        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer encoding
        x = self.transformer(x)

        # Layer norm and extract class token
        x = self.layer_norm(x)
        cls_output = x[:, 0]  # Use class token for classification

        # Classification
        logits = self.classifier(cls_output)

        out = {
            "logits": logits,
            "loss": None,
            "encoding": cls_output,  # Class token features
        }

        if targets is not None:
            out["loss"] = self.loss_fn(logits, targets)

        return out

    def __str__(self) -> str:
        return (
            f"TransformerTemplate(num_classes={self.num_classes}, "
            f"embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
            f"loss_fn={self.loss_fn})"
        )


# ========================================
# TEMPLATE BUILDER FUNCTION
# ========================================


def _build_template_model(configs: Namespace) -> nn.Module:
    """
    TEMPLATE: Builder function for new model architectures

    This template shows how to create a builder function for your new model.
    Copy and modify this function when adding new architectures.

    Args:
        configs: Configuration namespace containing:
            - configs.data.num_classes: int - Number of output classes
            - configs.model.arch: str - Model architecture name
            - configs.model.checkpoint: str or None - Path to pretrained weights
            - Any other model-specific configs

    Returns:
        nn.Module: Initialized model ready for training

    Usage:
        1. Copy this function and rename it (e.g., _build_efficientnet)
        2. Update the model instantiation with your architecture
        3. Modify checkpoint loading if needed
        4. Add your case to the main build_model() function in model.py
    """

    num_classes = configs.data.num_classes

    # Create your model instance
    # Replace this with your actual model constructor
    model = ModelTemplate(
        loss_fn=nn.CrossEntropyLoss(),
        num_classes=num_classes,
        # Add any architecture-specific parameters from configs:
        # hidden_dims=configs.model.get('hidden_dims', [64, 128, 256]),
        # dropout_rate=configs.model.get('dropout', 0.1),
        # etc.
    )

    # Load pretrained weights if specified
    if hasattr(configs.model, "checkpoint") and configs.model.checkpoint is not None:

        if configs.model.checkpoint.lower() == "imagenet":
            # Load ImageNet pretrained weights
            # You'll need to implement this based on your model's availability
            # Example:
            # checkpoint = torch.hub.load_state_dict_from_url(
            #     "https://download.pytorch.org/models/your_model.pth",
            #     progress=True
            # )
            # # Remove classifier weights to avoid size mismatch
            # checkpoint.pop("classifier.weight", None)
            # checkpoint.pop("classifier.bias", None)
            # model.load_state_dict(checkpoint, strict=False)
            pass

        else:
            # Load custom checkpoint
            checkpoint = torch.load(configs.model.checkpoint)

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

    return model
