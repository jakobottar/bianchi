"""
Tests for model building functions and ResNet implementations
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import jsonargparse
import pytest
import torch
from torch import nn

from framework.model import build_model
from framework.resnet import ResNet18, ResNet50, _build_resnet


class TestBuildModel:
    """Tests for the main build_model function"""

    def test_build_model_resnet18(self):
        """Test building ResNet18 model"""
        configs = jsonargparse.Namespace()
        configs.model = jsonargparse.Namespace()
        configs.model.arch = "resnet18"
        configs.model.checkpoint = None
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10

        with patch("framework.model._build_resnet") as mock_build_resnet:
            mock_model = MagicMock()
            mock_build_resnet.return_value = mock_model

            result = build_model(configs)

            mock_build_resnet.assert_called_once_with(configs)
            assert result == mock_model

    def test_build_model_resnet50(self):
        """Test building ResNet50 model"""
        configs = jsonargparse.Namespace()
        configs.model = jsonargparse.Namespace()
        configs.model.arch = "resnet50"
        configs.model.checkpoint = None
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 1000

        with patch("framework.model._build_resnet") as mock_build_resnet:
            mock_model = MagicMock()
            mock_build_resnet.return_value = mock_model

            result = build_model(configs)

            mock_build_resnet.assert_called_once_with(configs)
            assert result == mock_model

    def test_build_model_case_insensitive(self):
        """Test that model architecture names are case insensitive"""
        test_cases = ["ResNet18", "RESNET50", "resnet18", "ReSNet50"]

        for arch in test_cases:
            configs = jsonargparse.Namespace()
            configs.model = jsonargparse.Namespace()
            configs.model.arch = arch
            configs.model.checkpoint = None
            configs.data = jsonargparse.Namespace()
            configs.data.num_classes = 10

            with patch("framework.model._build_resnet") as mock_build_resnet:
                mock_model = MagicMock()
                mock_build_resnet.return_value = mock_model

                result = build_model(configs)

                mock_build_resnet.assert_called_once_with(configs)
                assert result == mock_model

    def test_build_model_unsupported_architecture(self):
        """Test error handling for unsupported architectures"""
        configs = jsonargparse.Namespace()
        configs.model = jsonargparse.Namespace()
        configs.model.arch = "vgg16"  # Not supported
        configs.model.checkpoint = None
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10

        with pytest.raises(ValueError) as exc_info:
            build_model(configs)

        assert "Model architecture vgg16 not supported" in str(exc_info.value)

    def test_build_model_config_validation(self):
        """Test config validation in build_model"""
        # Test missing model namespace
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10

        with pytest.raises(AttributeError):
            build_model(configs)

        # Test missing arch attribute
        configs.model = jsonargparse.Namespace()
        configs.model.checkpoint = None

        with pytest.raises(AttributeError):
            build_model(configs)


class TestBuildResnet:
    """Tests for the _build_resnet function"""

    def test_build_resnet18(self):
        """Test building ResNet18 through _build_resnet"""
        configs = jsonargparse.Namespace()
        configs.model = jsonargparse.Namespace()
        configs.model.arch = "resnet18"
        configs.model.checkpoint = None
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10

        model = _build_resnet(configs)

        assert isinstance(model, ResNet18)
        assert model.fc.out_features == 10  # num_classes
        assert isinstance(model.loss_fn, nn.CrossEntropyLoss)
        assert model.feature_size == 512

    def test_build_resnet50(self):
        """Test building ResNet50 through _build_resnet"""
        configs = jsonargparse.Namespace()
        configs.model = jsonargparse.Namespace()
        configs.model.arch = "resnet50"
        configs.model.checkpoint = None
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 100

        model = _build_resnet(configs)

        assert isinstance(model, ResNet50)
        assert model.fc.out_features == 100  # num_classes
        assert isinstance(model.loss_fn, nn.CrossEntropyLoss)
        assert model.feature_size == 2048

    def test_build_resnet_different_num_classes(self):
        """Test ResNet building with different numbers of classes"""
        num_classes_list = [1, 10, 100, 1000, 10000]

        for num_classes in num_classes_list:
            configs = jsonargparse.Namespace()
            configs.model = jsonargparse.Namespace()
            configs.model.arch = "resnet18"
            configs.model.checkpoint = None
            configs.data = jsonargparse.Namespace()
            configs.data.num_classes = num_classes

            model = _build_resnet(configs)

            assert model.fc.out_features == num_classes

    def test_build_resnet_with_checkpoint(self):
        """Test ResNet building with checkpoint loading"""
        # Create a temporary checkpoint file
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, "model.pth")

        try:
            # Create a dummy checkpoint
            dummy_state_dict = {
                "conv1.weight": torch.randn(64, 3, 7, 7),
                "bn1.weight": torch.randn(64),
                "bn1.bias": torch.randn(64),
            }
            torch.save({"model_state_dict": dummy_state_dict}, checkpoint_path)

            configs = jsonargparse.Namespace()
            configs.model = jsonargparse.Namespace()
            configs.model.arch = "resnet18"
            configs.model.checkpoint = checkpoint_path
            configs.data = jsonargparse.Namespace()
            configs.data.num_classes = 10

            with patch("torch.load") as mock_load:
                mock_load.return_value = {"model_state_dict": dummy_state_dict}

                model = _build_resnet(configs)

                # Verify torch.load was called with the checkpoint path
                mock_load.assert_called_once_with(checkpoint_path)
                assert isinstance(model, ResNet18)

        finally:
            shutil.rmtree(temp_dir)

    def test_build_resnet_unsupported_arch(self):
        """Test error handling for unsupported ResNet architectures"""
        configs = jsonargparse.Namespace()
        configs.model = jsonargparse.Namespace()
        configs.model.arch = "resnet34"  # Not implemented
        configs.model.checkpoint = None
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10

        with pytest.raises(ValueError) as exc_info:
            _build_resnet(configs)

        assert "Model architecture resnet34 not supported" in str(exc_info.value)


class TestResNet18:
    """Tests for ResNet18 model class"""

    def test_resnet18_initialization(self):
        """Test ResNet18 initialization"""
        loss_fn = nn.CrossEntropyLoss()
        model = ResNet18(loss_fn=loss_fn, num_classes=10)

        assert model.feature_size == 512
        assert model.loss_fn == loss_fn
        assert model.fc.out_features == 10
        assert isinstance(model, nn.Module)

    def test_resnet18_forward_without_targets(self):
        """Test ResNet18 forward pass without targets"""
        loss_fn = nn.CrossEntropyLoss()
        model = ResNet18(loss_fn=loss_fn, num_classes=10)
        model.eval()

        # Create dummy input (batch_size=2, channels=3, height=224, width=224)
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        # Check output structure
        assert isinstance(output, dict)
        assert "logits" in output
        assert "loss" in output
        assert "encoding" in output

        # Check output shapes
        assert output["logits"].shape == (2, 10)  # batch_size x num_classes
        assert output["encoding"].shape == (2, 512)  # batch_size x feature_size
        assert output["loss"] is None  # No targets provided

    def test_resnet18_forward_with_targets(self):
        """Test ResNet18 forward pass with targets"""
        loss_fn = nn.CrossEntropyLoss()
        model = ResNet18(loss_fn=loss_fn, num_classes=10)
        model.eval()

        # Create dummy input and targets
        x = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([0, 5])

        with torch.no_grad():
            output = model(x)

        # Check output structure
        assert isinstance(output, dict)
        assert "logits" in output
        assert "loss" in output
        assert "encoding" in output

        # Check output shapes
        assert output["logits"].shape == (2, 10)
        assert output["encoding"].shape == (2, 512)
        assert output["loss"] is None  # targets not passed to forward

        # Test with targets
        with torch.no_grad():
            output_with_targets = model(x, targets)

        assert output_with_targets["loss"] is not None
        assert isinstance(output_with_targets["loss"], torch.Tensor)

    def test_resnet18_weight_initialization(self):
        """Test that ResNet18 weights are properly initialized"""
        loss_fn = nn.CrossEntropyLoss()
        model = ResNet18(loss_fn=loss_fn, num_classes=10)

        # Check that weights are not all zeros (indicating initialization occurred)
        conv_weights = model.conv1.weight.data
        assert not torch.allclose(conv_weights, torch.zeros_like(conv_weights))

        # Check batch norm initialization
        bn_weights = model.bn1.weight.data
        assert torch.allclose(bn_weights, torch.ones_like(bn_weights))

        bn_bias = model.bn1.bias.data
        assert torch.allclose(bn_bias, torch.zeros_like(bn_bias))


class TestResNet50:
    """Tests for ResNet50 model class"""

    def test_resnet50_initialization(self):
        """Test ResNet50 initialization"""
        loss_fn = nn.CrossEntropyLoss()
        model = ResNet50(loss_fn=loss_fn, num_classes=1000)

        assert model.feature_size == 2048
        assert model.loss_fn == loss_fn
        assert model.fc.out_features == 1000
        assert isinstance(model, nn.Module)

    def test_resnet50_forward_without_targets(self):
        """Test ResNet50 forward pass without targets"""
        loss_fn = nn.CrossEntropyLoss()
        model = ResNet50(loss_fn=loss_fn, num_classes=100)
        model.eval()

        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        # Check output structure
        assert isinstance(output, dict)
        assert "logits" in output
        assert "loss" in output
        assert "encoding" in output

        # Check output shapes
        assert output["logits"].shape == (1, 100)  # batch_size x num_classes
        assert output["encoding"].shape == (1, 2048)  # batch_size x feature_size
        assert output["loss"] is None  # No targets provided

    def test_resnet50_forward_with_targets(self):
        """Test ResNet50 forward pass with targets"""
        loss_fn = nn.CrossEntropyLoss()
        model = ResNet50(loss_fn=loss_fn, num_classes=10)
        model.eval()

        # Create dummy input and targets
        x = torch.randn(1, 3, 224, 224)
        targets = torch.tensor([3])

        with torch.no_grad():
            output_with_targets = model(x, targets)

        # Check that loss is computed when targets are provided
        assert output_with_targets["loss"] is not None
        assert isinstance(output_with_targets["loss"], torch.Tensor)
        assert output_with_targets["logits"].shape == (1, 10)
        assert output_with_targets["encoding"].shape == (1, 2048)

    def test_resnet50_different_input_sizes(self):
        """Test ResNet50 with different input batch sizes"""
        loss_fn = nn.CrossEntropyLoss()
        model = ResNet50(loss_fn=loss_fn, num_classes=10)
        model.eval()

        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 224, 224)

            with torch.no_grad():
                output = model(x)

            assert output["logits"].shape == (batch_size, 10)
            assert output["encoding"].shape == (batch_size, 2048)


class TestModelIntegration:
    """Integration tests for model building"""

    def test_end_to_end_model_building(self):
        """Test complete model building workflow"""
        configs = jsonargparse.Namespace()
        configs.model = jsonargparse.Namespace()
        configs.model.arch = "resnet18"
        configs.model.checkpoint = None
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10

        # Build model through the main interface
        model = build_model(configs)

        # Verify the model is properly constructed
        assert isinstance(model, ResNet18)
        assert model.fc.out_features == 10

        # Test a forward pass with appropriate input size and eval mode
        model.eval()  # Set to eval mode to avoid batch norm issues
        x = torch.randn(2, 3, 224, 224)  # Use larger input and batch size
        with torch.no_grad():
            output = model(x)

        assert "logits" in output
        assert "encoding" in output
        assert output["logits"].shape == (2, 10)

    def test_model_consistency_across_architectures(self):
        """Test that different ResNet architectures have consistent interfaces"""
        architectures = [("resnet18", ResNet18, 512), ("resnet50", ResNet50, 2048)]

        for arch_name, expected_class, expected_feature_size in architectures:
            configs = jsonargparse.Namespace()
            configs.model = jsonargparse.Namespace()
            configs.model.arch = arch_name
            configs.model.checkpoint = None
            configs.data = jsonargparse.Namespace()
            configs.data.num_classes = 10

            model = build_model(configs)

            # Check model type and feature size
            assert isinstance(model, expected_class)
            assert model.feature_size == expected_feature_size
            assert model.fc.out_features == 10

            # Check that all models have the same output interface
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(x)

            required_keys = {"logits", "loss", "encoding"}
            assert set(output.keys()) == required_keys

    def test_checkpoint_loading_integration(self):
        """Test checkpoint loading integration"""
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, "test_model.pth")

        try:
            # Create a real ResNet18 model and save it
            original_model = ResNet18(loss_fn=nn.CrossEntropyLoss(), num_classes=10)
            torch.save(
                {"model_state_dict": original_model.state_dict()}, checkpoint_path
            )

            # Now test loading it through build_model
            configs = jsonargparse.Namespace()
            configs.model = jsonargparse.Namespace()
            configs.model.arch = "resnet18"
            configs.model.checkpoint = checkpoint_path
            configs.data = jsonargparse.Namespace()
            configs.data.num_classes = 10

            loaded_model = build_model(configs)

            # Verify the model was loaded correctly
            assert isinstance(loaded_model, ResNet18)
            assert loaded_model.fc.out_features == 10

        finally:
            shutil.rmtree(temp_dir)


class TestModelErrorHandling:
    """Tests for error handling in model building"""

    def test_invalid_config_types(self):
        """Test handling of invalid config types"""
        # Test with None config
        with pytest.raises(AttributeError):
            build_model(None)

        # Test with missing required attributes
        configs = jsonargparse.Namespace()
        with pytest.raises(AttributeError):
            build_model(configs)

    def test_checkpoint_loading_errors(self):
        """Test error handling for checkpoint loading issues"""
        configs = jsonargparse.Namespace()
        configs.model = jsonargparse.Namespace()
        configs.model.arch = "resnet18"
        configs.model.checkpoint = "/nonexistent/path/model.pth"
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10

        # This should raise a FileNotFoundError when trying to load the checkpoint
        with pytest.raises(FileNotFoundError):
            _build_resnet(configs)

    def test_invalid_num_classes(self):
        """Test handling of invalid number of classes"""
        configs = jsonargparse.Namespace()
        configs.model = jsonargparse.Namespace()
        configs.model.arch = "resnet18"
        configs.model.checkpoint = None
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 0  # Invalid

        # Should still create model but with 0 output features
        model = _build_resnet(configs)
        assert model.fc.out_features == 0
