"""
Additional tests for parse_configs function with proper mocking
"""

import json
import os
import tempfile
from unittest.mock import patch

import jsonargparse

from framework.config import parse_configs


class TestParseConfigsWithMocking:
    """Tests for parse_configs function using mocking to avoid config file issues"""

    def test_parse_configs_default_arguments(self):
        """Test parse_configs with default arguments using mocking"""
        # Mock the default config file path to avoid validation issues
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Create a valid config that matches expected parameters
            valid_config = {
                "data": {"batch_size": 8, "root": "./data/", "workers": 2},
                "epochs": 1,
                "model": {"arch": "resnet18"},
                "opt": {"lr": 1.0, "weight_decay": 1e-9},
                "name": "random",
                "seed": -1,
                "root": "runs",
            }
            json.dump(valid_config, f)
            temp_config_path = f.name

        try:
            test_args = ["test", "--config", temp_config_path]
            with patch("sys.argv", test_args):
                configs = parse_configs()

                # Verify it's the correct type
                assert isinstance(configs, jsonargparse.Namespace)

                # Check that set_up_configs was called (name should be different from "random")
                assert configs.name != "random"  # Should be generated
                assert hasattr(configs, "root")
                assert configs.data.batch_size == 8
                assert configs.epochs == 1
                assert configs.model.arch == "resnet18"

        finally:
            os.unlink(temp_config_path)

    def test_parse_configs_command_line_override(self):
        """Test that command line arguments override config file values"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "data": {"batch_size": 8},
                "epochs": 1,
                "model": {"arch": "resnet18"},
                "opt": {"lr": 1.0},
            }
            json.dump(config, f)
            temp_config_path = f.name

        try:
            test_args = [
                "test",
                "--config",
                temp_config_path,
                "--data.batch_size",
                "32",
                "--epochs",
                "10",
                "--name",
                "override_test",
            ]

            with patch("sys.argv", test_args):
                configs = parse_configs()

                # Command line should override config file
                assert configs.data.batch_size == 32
                assert configs.epochs == 10
                assert configs.name == "-1_override_test"  # Now includes slurm job id

                # Values not overridden should come from config
                assert configs.model.arch == "resnet18"
                assert configs.opt.lr == 1.0

        finally:
            os.unlink(temp_config_path)

    def test_parse_configs_integration_with_setup(self):
        """Test that parse_configs properly integrates with set_up_configs"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "data": {"batch_size": 16},
                "epochs": 5,
                "model": {"arch": "resnet50"},
                "name": "integration_test_run",
                "seed": 42,
            }
            json.dump(config, f)
            temp_config_path = f.name

        try:
            test_args = ["test", "--config", temp_config_path]

            with patch("sys.argv", test_args):
                configs = parse_configs()

                # Verify the integration worked (name now includes slurm job id)
                assert configs.name == "-1_integration_test_run"
                assert configs.seed == 42
                assert configs.data.batch_size == 16
                assert configs.epochs == 5

                # Verify that set_up_configs was called (directory should exist)
                assert os.path.exists(configs.root)
                assert configs.root.endswith("-1_integration_test_run")

                # Clean up the created directory
                import shutil

                run_dir = configs.root
                if os.path.exists(run_dir):
                    shutil.rmtree(run_dir)

        finally:
            os.unlink(temp_config_path)
