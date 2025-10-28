"""
Tests for dataset building functions
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, call, patch

import jsonargparse
import pytest
import torch
from torch.utils.data import Dataset

from framework.data import (
    _build_cifar10_datasets,
    _build_mnist_datasets,
    build_datasets,
)


class TestBuildDatasets:
    """Tests for the main build_datasets function"""

    def test_build_datasets_mnist(self):
        """Test building MNIST datasets"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.name = "mnist"
        configs.data.root = "./test_data"

        with patch("framework.data._build_mnist_datasets") as mock_build_mnist:
            expected_return = {
                "train": MagicMock(),
                "test": MagicMock(),
                "num_classes": 10,
            }
            mock_build_mnist.return_value = expected_return

            result = build_datasets(configs)

            mock_build_mnist.assert_called_once_with(configs)
            assert result == expected_return

    def test_build_datasets_cifar10(self):
        """Test building CIFAR-10 datasets"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.name = "cifar10"
        configs.data.root = "./test_data"

        with patch("framework.data._build_cifar10_datasets") as mock_build_cifar10:
            expected_return = {
                "train": MagicMock(),
                "test": MagicMock(),
                "num_classes": 10,
            }
            mock_build_cifar10.return_value = expected_return

            result = build_datasets(configs)

            mock_build_cifar10.assert_called_once_with(configs)
            assert result == expected_return

    def test_build_datasets_unsupported(self):
        """Test error handling for unsupported datasets"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.name = "unsupported_dataset"
        configs.data.root = "./test_data"

        with pytest.raises(ValueError) as exc_info:
            build_datasets(configs)

        assert "Dataset unsupported_dataset not supported" in str(exc_info.value)

    def test_build_datasets_case_sensitivity(self):
        """Test that dataset names work regardless of case"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.root = "./test_data"

        # Test various case combinations that should all work
        test_cases = [
            "MNIST",
            "mnist",
            "Mnist",
            "MnIsT",
            "CIFAR10",
            "cifar10",
            "Cifar10",
            "CiFaR10",
        ]

        for dataset_name in test_cases:
            configs.data.name = dataset_name

            expected_function = dataset_name.lower()
            if expected_function == "mnist":
                with patch("framework.data._build_mnist_datasets") as mock_build:
                    mock_build.return_value = {
                        "train": MagicMock(),
                        "test": MagicMock(),
                        "num_classes": 10,
                    }
                    result = build_datasets(configs)
                    mock_build.assert_called_once_with(configs)
                    assert result["num_classes"] == 10
            elif expected_function == "cifar10":
                with patch("framework.data._build_cifar10_datasets") as mock_build:
                    mock_build.return_value = {
                        "train": MagicMock(),
                        "test": MagicMock(),
                        "num_classes": 10,
                    }
                    result = build_datasets(configs)
                    mock_build.assert_called_once_with(configs)
                    assert result["num_classes"] == 10


class TestMnistDatasets:
    """Tests for MNIST dataset building"""

    def test_build_mnist_datasets_structure(self):
        """Test that MNIST datasets are built with correct structure"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.data = jsonargparse.Namespace()
            configs.data.root = temp_dir

            with patch("torchvision.datasets.MNIST") as mock_mnist:
                # Create mock datasets
                mock_train_dataset = MagicMock()
                mock_test_dataset = MagicMock()

                # Configure the mock to return different instances for train/test
                def mnist_side_effect(*args, **kwargs):
                    if kwargs.get("train", True):
                        return mock_train_dataset
                    else:
                        return mock_test_dataset

                mock_mnist.side_effect = mnist_side_effect

                result = _build_mnist_datasets(configs)

                # Check the returned structure
                assert isinstance(result, dict)
                assert "train" in result
                assert "test" in result
                assert "num_classes" in result
                assert result["num_classes"] == 10
                assert result["train"] == mock_train_dataset
                assert result["test"] == mock_test_dataset

        finally:
            shutil.rmtree(temp_dir)

    def test_build_mnist_datasets_calls(self):
        """Test that MNIST datasets are created with correct parameters"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.data = jsonargparse.Namespace()
            configs.data.root = temp_dir

            with patch("torchvision.datasets.MNIST") as mock_mnist:
                mock_dataset = MagicMock()
                mock_mnist.return_value = mock_dataset

                _build_mnist_datasets(configs)

                # Check that MNIST was called twice (train and test)
                assert mock_mnist.call_count == 2

                # Check the calls
                calls = mock_mnist.call_args_list

                # First call should be for training data
                train_call = calls[0]
                assert train_call[1]["root"] == temp_dir
                assert train_call[1]["train"] is True
                assert train_call[1]["download"] is True
                assert "transform" in train_call[1]

                # Second call should be for test data
                test_call = calls[1]
                assert test_call[1]["root"] == temp_dir
                assert test_call[1]["train"] is False
                assert test_call[1]["download"] is True
                assert "transform" in test_call[1]

        finally:
            shutil.rmtree(temp_dir)

    def test_build_mnist_datasets_transforms(self):
        """Test that MNIST datasets use correct transforms"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.data = jsonargparse.Namespace()
            configs.data.root = temp_dir

            with patch("torchvision.datasets.MNIST") as mock_mnist:
                mock_dataset = MagicMock()
                mock_mnist.return_value = mock_dataset

                _build_mnist_datasets(configs)

                # Check that transforms were passed
                calls = mock_mnist.call_args_list
                for call_args in calls:
                    transform = call_args[1]["transform"]
                    assert transform is not None
                    # The transform should be a Compose object with ToTensor
                    assert hasattr(transform, "transforms")

        finally:
            shutil.rmtree(temp_dir)


class TestCifar10Datasets:
    """Tests for CIFAR-10 dataset building"""

    def test_build_cifar10_datasets_structure(self):
        """Test that CIFAR-10 datasets are built with correct structure"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.data = jsonargparse.Namespace()
            configs.data.root = temp_dir

            with patch("torchvision.datasets.CIFAR10") as mock_cifar10:
                # Create mock datasets
                mock_train_dataset = MagicMock()
                mock_test_dataset = MagicMock()

                # Configure the mock to return different instances for train/test
                def cifar10_side_effect(*args, **kwargs):
                    if kwargs.get("train", True):
                        return mock_train_dataset
                    else:
                        return mock_test_dataset

                mock_cifar10.side_effect = cifar10_side_effect

                result = _build_cifar10_datasets(configs)

                # Check the returned structure
                assert isinstance(result, dict)
                assert "train" in result
                assert "test" in result
                assert "num_classes" in result
                assert result["num_classes"] == 10
                assert result["train"] == mock_train_dataset
                assert result["test"] == mock_test_dataset

        finally:
            shutil.rmtree(temp_dir)

    def test_build_cifar10_datasets_calls(self):
        """Test that CIFAR-10 datasets are created with correct parameters"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.data = jsonargparse.Namespace()
            configs.data.root = temp_dir

            with patch("torchvision.datasets.CIFAR10") as mock_cifar10:
                mock_dataset = MagicMock()
                mock_cifar10.return_value = mock_dataset

                _build_cifar10_datasets(configs)

                # Check that CIFAR10 was called twice (train and test)
                assert mock_cifar10.call_count == 2

                # Check the calls
                calls = mock_cifar10.call_args_list

                # First call should be for training data
                train_call = calls[0]
                assert train_call[1]["root"] == temp_dir
                assert train_call[1]["train"] is True
                assert train_call[1]["download"] is True
                assert "transform" in train_call[1]

                # Second call should be for test data
                test_call = calls[1]
                assert test_call[1]["root"] == temp_dir
                assert test_call[1]["train"] is False
                assert test_call[1]["download"] is True
                assert "transform" in test_call[1]

        finally:
            shutil.rmtree(temp_dir)

    def test_build_cifar10_datasets_transforms(self):
        """Test that CIFAR-10 datasets use correct transforms"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.data = jsonargparse.Namespace()
            configs.data.root = temp_dir

            with patch("torchvision.datasets.CIFAR10") as mock_cifar10:
                mock_dataset = MagicMock()
                mock_cifar10.return_value = mock_dataset

                _build_cifar10_datasets(configs)

                # Check that transforms were passed
                calls = mock_cifar10.call_args_list
                for call_args in calls:
                    transform = call_args[1]["transform"]
                    assert transform is not None
                    # The transform should be a Compose object with ToTensor
                    assert hasattr(transform, "transforms")

        finally:
            shutil.rmtree(temp_dir)


class TestDatasetIntegration:
    """Integration tests for dataset building"""

    def test_mnist_cifar10_consistency(self):
        """Test that MNIST and CIFAR-10 return consistent structures"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.data = jsonargparse.Namespace()
            configs.data.root = temp_dir

            with patch("torchvision.datasets.MNIST") as mock_mnist, patch(
                "torchvision.datasets.CIFAR10"
            ) as mock_cifar10:

                mock_dataset = MagicMock()
                mock_mnist.return_value = mock_dataset
                mock_cifar10.return_value = mock_dataset

                # Test MNIST
                configs.data.name = "mnist"
                mnist_result = build_datasets(configs)

                # Test CIFAR-10
                configs.data.name = "cifar10"
                cifar10_result = build_datasets(configs)

                # Both should have the same structure
                assert set(mnist_result.keys()) == set(cifar10_result.keys())
                assert mnist_result["num_classes"] == cifar10_result["num_classes"]

        finally:
            shutil.rmtree(temp_dir)

    def test_config_validation(self):
        """Test that configs are properly validated"""
        # Test missing data namespace
        configs = jsonargparse.Namespace()

        with pytest.raises(AttributeError):
            build_datasets(configs)

        # Test missing name attribute
        configs.data = jsonargparse.Namespace()
        configs.data.root = "./test"

        with pytest.raises(AttributeError):
            build_datasets(configs)

    def test_dataset_root_handling(self):
        """Test that different root paths work correctly"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.name = "mnist"

        # Test with different root paths
        test_roots = ["./data", "/tmp/test", "~/datasets", "./nested/path/data"]

        for root in test_roots:
            configs.data.root = root

            with patch("torchvision.datasets.MNIST") as mock_mnist:
                mock_dataset = MagicMock()
                mock_mnist.return_value = mock_dataset

                _build_mnist_datasets(configs)

                # Check that the root was passed correctly
                calls = mock_mnist.call_args_list
                for call_args in calls:
                    assert call_args[1]["root"] == root


class TestDatasetErrorHandling:
    """Tests for error handling in dataset building"""

    def test_invalid_config_types(self):
        """Test handling of invalid config types"""
        # Test with None config
        with pytest.raises(AttributeError):
            build_datasets(None)

        # Test with empty config
        configs = jsonargparse.Namespace()
        with pytest.raises(AttributeError):
            build_datasets(configs)

    def test_missing_required_attributes(self):
        """Test handling of missing required config attributes"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()

        # Missing name
        with pytest.raises(AttributeError):
            build_datasets(configs)

        # Missing root for specific dataset builders
        configs.data.name = "mnist"
        with pytest.raises(AttributeError):
            _build_mnist_datasets(configs)

    def test_filesystem_errors(self):
        """Test handling of filesystem-related errors"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.name = "mnist"
        configs.data.root = "/invalid/path/that/should/not/exist"

        # Test that filesystem errors are propagated appropriately
        with patch("torchvision.datasets.MNIST") as mock_mnist:
            # Simulate a permission error
            mock_mnist.side_effect = PermissionError("Cannot access directory")

            with pytest.raises(PermissionError):
                _build_mnist_datasets(configs)


class TestTransformFunctionality:
    """Tests for transform functionality in dataset building"""

    def test_transform_composition(self):
        """Test that transforms are properly composed"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.data = jsonargparse.Namespace()
            configs.data.root = temp_dir

            with patch("torchvision.datasets.MNIST") as mock_mnist:
                mock_dataset = MagicMock()
                mock_mnist.return_value = mock_dataset

                _build_mnist_datasets(configs)

                # Get the transform that was passed
                calls = mock_mnist.call_args_list
                transform = calls[0][1]["transform"]

                # Verify it's a Compose transform
                assert hasattr(transform, "transforms")
                assert len(transform.transforms) > 0

                # Check that ToImage is included
                transform_types = [type(t).__name__ for t in transform.transforms]
                assert "ToImage" in transform_types

        finally:
            shutil.rmtree(temp_dir)

    def test_identical_transforms(self):
        """Test that train and test datasets use identical transforms"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.data = jsonargparse.Namespace()
            configs.data.root = temp_dir

            with patch("torchvision.datasets.MNIST") as mock_mnist:
                mock_dataset = MagicMock()
                mock_mnist.return_value = mock_dataset

                _build_mnist_datasets(configs)

                # Get transforms from both calls
                calls = mock_mnist.call_args_list
                train_transform = calls[0][1]["transform"]
                test_transform = calls[1][1]["transform"]

                # Transforms should be equivalent (same structure)
                assert type(train_transform) == type(test_transform)
                assert len(train_transform.transforms) == len(test_transform.transforms)

        finally:
            shutil.rmtree(temp_dir)
