"""
Tests for config parsing and setup functions
"""

import json
import os
import random
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import jsonargparse
import pytest
import torch

from framework.config import _set_up_configs, parse_configs


class TestSetUpConfigs:
    """Tests for the set_up_configs function"""

    def test_set_up_configs_with_random_name(self):
        """Test setup with random name generation"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "random"
            configs.seed = -1
            configs.root = temp_dir
            configs.config = None  # Add config key to avoid KeyError

            with patch("namegenerator.gen", return_value="test-generated-name"):
                result_configs = _set_up_configs(configs)

                assert result_configs.name == "test-generated-name"
                assert os.path.exists(result_configs.root)
                assert result_configs.root.endswith("test-generated-name")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_set_up_configs_with_custom_name(self):
        """Test setup with custom name"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "my_custom_run"
            configs.seed = -1
            configs.root = temp_dir
            configs.config = None  # Add config key to avoid KeyError

            result_configs = _set_up_configs(configs)

            assert result_configs.name == "my_custom_run"
            assert os.path.exists(result_configs.root)
            assert result_configs.root.endswith("my_custom_run")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_set_up_configs_with_seed(self):
        """Test setup with deterministic seed"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "seeded_run"
            configs.seed = 42
            configs.root = temp_dir
            configs.config = None  # Add config key to avoid KeyError

            with patch("random.seed") as mock_random_seed, patch(
                "torch.manual_seed"
            ) as mock_torch_seed:

                result_configs = _set_up_configs(configs)

                mock_random_seed.assert_called_once_with(42)
                mock_torch_seed.assert_called_once_with(42)
                assert torch.backends.cudnn.deterministic is True
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_set_up_configs_without_seed(self):
        """Test setup without setting seed (seed = -1)"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "no_seed_run"
            configs.seed = -1
            configs.root = temp_dir
            configs.config = None  # Add config key to avoid KeyError

            with patch("random.seed") as mock_random_seed, patch(
                "torch.manual_seed"
            ) as mock_torch_seed:

                result_configs = _set_up_configs(configs)

                mock_random_seed.assert_not_called()
                mock_torch_seed.assert_not_called()
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_set_up_configs_creates_directory(self):
        """Test that setup creates the run directory"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "test_dir_creation"
            configs.seed = -1
            configs.root = temp_dir
            configs.config = None  # Add config key to avoid KeyError

            result_configs = _set_up_configs(configs)

            expected_path = os.path.join(temp_dir, "test_dir_creation")
            assert result_configs.root == expected_path
            assert os.path.exists(expected_path)
            assert os.path.isdir(expected_path)
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_set_up_configs_saves_config_file(self):
        """Test that setup saves config.json file"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "config_save_test"
            configs.seed = -1
            configs.root = temp_dir
            configs.data = jsonargparse.Namespace()
            configs.data.batch_size = 16
            configs.epochs = 5
            configs.config = "some_config_path"  # This should be removed

            result_configs = _set_up_configs(configs)

            config_file_path = os.path.join(result_configs.root, "config.json")
            assert os.path.exists(config_file_path)

            # Read and verify the saved config
            with open(config_file_path, "r") as f:
                saved_config = json.load(f)

            assert saved_config["data"]["batch_size"] == 16
            assert saved_config["epochs"] == 5
            assert "config" not in saved_config  # Should be removed
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_set_up_configs_handles_path_objects(self):
        """Test that setup converts Path objects to strings"""
        from pathlib import Path

        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "path_test"
            configs.seed = -1
            configs.root = temp_dir
            configs.some_path = Path("/some/test/path")
            configs.config = "config_path"

            result_configs = _set_up_configs(configs)

            config_file_path = os.path.join(result_configs.root, "config.json")
            with open(config_file_path, "r") as f:
                saved_config = json.load(f)

            assert saved_config["some_path"] == "/some/test/path"
            assert isinstance(saved_config["some_path"], str)
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestConfigIntegration:
    """Integration tests focused on config setup without parsing issues"""

    def test_name_generation_random(self):
        """Test that random name generation works"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "random"
            configs.seed = -1
            configs.root = temp_dir
            configs.config = None

            with patch("namegenerator.gen", return_value="mighty-zebra-42"):
                result = _set_up_configs(configs)

                assert result.name == "mighty-zebra-42"
                assert "mighty-zebra-42" in result.root
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_seed_setting_behavior(self):
        """Test that seed setting works correctly"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "seed_test"
            configs.seed = 12345
            configs.root = temp_dir
            configs.config = None

            # Test that the actual functions are called
            original_random_seed = random.seed
            original_torch_seed = torch.manual_seed

            random_called = []
            torch_called = []

            def mock_random_seed(seed):
                random_called.append(seed)
                return original_random_seed(seed)

            def mock_torch_seed(seed):
                torch_called.append(seed)
                return original_torch_seed(seed)

            with patch("random.seed", side_effect=mock_random_seed), patch(
                "torch.manual_seed", side_effect=mock_torch_seed
            ):

                result = _set_up_configs(configs)

                assert random_called == [12345]
                assert torch_called == [12345]
                assert torch.backends.cudnn.deterministic is True
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_directory_and_file_creation(self):
        """Test that directories and config files are created properly"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "test_creation"
            configs.seed = -1
            configs.root = temp_dir
            configs.config = None
            configs.model = jsonargparse.Namespace()
            configs.model.arch = "resnet18"
            configs.epochs = 10

            result = _set_up_configs(configs)

            # Check directory creation
            expected_dir = os.path.join(temp_dir, "test_creation")
            assert os.path.exists(expected_dir)
            assert os.path.isdir(expected_dir)
            assert result.root == expected_dir

            # Check config file creation
            config_file = os.path.join(expected_dir, "config.json")
            assert os.path.exists(config_file)

            with open(config_file, "r") as f:
                saved_config = json.load(f)

            assert saved_config["name"] == "test_creation"
            assert saved_config["epochs"] == 10
            assert saved_config["model"]["arch"] == "resnet18"
            assert "config" not in saved_config  # Should be removed

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_robust_config_handling(self):
        """Test that setup handles missing config key gracefully after our fix"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "robust_test"
            configs.seed = -1
            configs.root = temp_dir
            # No config key added - should work fine now

            result = _set_up_configs(configs)

            assert result.name == "robust_test"
            assert os.path.exists(result.root)
            assert result.root.endswith("robust_test")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
