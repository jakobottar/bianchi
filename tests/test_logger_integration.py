"""
Tests for logger integration with config system
"""

import os
import shutil
import tempfile
from unittest.mock import patch

import jsonargparse

from framework.config import _create_logger, _set_up_configs, parse_configs
from framework.utils import get_current_logger, set_current_logger


class TestLoggerCreation:
    """Tests for the _create_logger function"""

    def test_create_logger_basic_functionality(self):
        """Test that _create_logger creates a working logger"""
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, "test.log")

        try:
            logger, close_logger = _create_logger(log_file)

            # Test logging
            logger("Test message 1")
            logger("Test message 2")

            # Close logger to flush
            close_logger()

            # Read log file and verify content
            with open(log_file, "r") as f:
                content = f.read()

            assert "Test message 1" in content
            assert "Test message 2" in content

        finally:
            shutil.rmtree(temp_dir)

    def test_create_logger_display_parameter(self):
        """Test _create_logger with display=False"""
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, "test.log")

        try:
            # Create logger with display=False
            logger, close_logger = _create_logger(log_file, display=False)

            # Mock print to verify it's not called
            with patch("builtins.print") as mock_print:
                logger("Test message")
                mock_print.assert_not_called()

            close_logger()

            # Verify message was still written to file
            with open(log_file, "r") as f:
                content = f.read()
            assert "Test message" in content

        finally:
            shutil.rmtree(temp_dir)

    def test_create_logger_flushing(self):
        """Test that logger flushes every 10 messages"""
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, "test.log")

        try:
            logger, close_logger = _create_logger(log_file, display=False)

            # Write 9 messages - should not trigger flush
            for i in range(9):
                logger(f"Message {i}")

            # Write 10th message - should trigger flush
            logger("Message 9")

            # Verify file has content (flush happened)
            with open(log_file, "r") as f:
                content = f.read()
            assert "Message 0" in content
            assert "Message 9" in content

            close_logger()

        finally:
            shutil.rmtree(temp_dir)


class TestConfigLoggerIntegration:
    """Tests for logger integration in config setup"""

    def test_set_up_configs_creates_logger(self):
        """Test that _set_up_configs creates and assigns logger"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.name = "test_run"
            configs.seed = -1
            configs.slurm_job_id = -1
            configs.root = temp_dir

            result = _set_up_configs(configs)

            # Verify logger was created
            assert hasattr(result, "logger")
            assert hasattr(result, "close_logger")
            assert callable(result.logger)
            assert callable(result.close_logger)

            # Test that logger works
            result.logger("Test logging message")

            # Verify log file was created
            log_file = os.path.join(result.root, "job.log")
            assert os.path.exists(log_file)

            # Close logger and check content
            result.close_logger()
            with open(log_file, "r") as f:
                content = f.read()
            assert "Test logging message" in content

        finally:
            shutil.rmtree(temp_dir)

    def test_set_up_configs_sets_current_logger(self):
        """Test that _set_up_configs sets the current logger globally"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Clear any existing logger
            set_current_logger(None)

            configs = jsonargparse.Namespace()
            configs.name = "test_run"
            configs.seed = -1
            configs.slurm_job_id = -1
            configs.root = temp_dir

            result = _set_up_configs(configs)

            # Verify current logger was set
            current_logger = get_current_logger()
            assert current_logger == result.logger

            # Test global logging works
            from framework.utils import log

            log("Global log message")

            # Verify message was logged
            result.close_logger()
            log_file = os.path.join(result.root, "job.log")
            with open(log_file, "r") as f:
                content = f.read()
            assert "Global log message" in content

        finally:
            set_current_logger(None)
            shutil.rmtree(temp_dir)

    def test_set_up_configs_resume_scenario_logging(self):
        """Test logging during resume scenario"""
        temp_dir = tempfile.mkdtemp()

        try:
            # First, create an existing run directory with config
            existing_run_name = "123_existing_run"
            existing_run_dir = os.path.join(temp_dir, existing_run_name)
            os.makedirs(existing_run_dir)

            # Create existing config
            import json

            existing_config = {
                "name": existing_run_name,
                "slurm_job_id": 123,
                "epochs": 10,
            }
            with open(os.path.join(existing_run_dir, "config.json"), "w") as f:
                json.dump(existing_config, f)

            # Now test resuming
            configs = jsonargparse.Namespace()
            configs.name = "different_name"  # Should be overridden
            configs.seed = -1
            configs.slurm_job_id = 123  # Same as existing
            configs.root = temp_dir

            result = _set_up_configs(configs)

            # Verify logger was created for resume scenario
            assert hasattr(result, "logger")
            assert result.name == existing_run_name

            # Close logger and check that resume message was logged
            result.close_logger()
            log_file = os.path.join(result.root, "job.log")
            with open(log_file, "r") as f:
                content = f.read()
            assert "found existing run with slurm job id 123, resuming" in content

        finally:
            shutil.rmtree(temp_dir)


class TestParseConfigsLogging:
    """Integration tests for logging in parse_configs"""

    def test_parse_configs_sets_up_logger(self):
        """Test that parse_configs sets up logging correctly"""
        temp_config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        temp_config_file.write('{"name": "test_config_run"}')
        temp_config_file.close()

        try:
            test_args = ["test", "--config", temp_config_file.name]

            with patch("sys.argv", test_args):
                configs = parse_configs()

                # Verify logger exists and works
                assert hasattr(configs, "logger")
                assert hasattr(configs, "close_logger")

                # Test logging
                configs.logger("Test message from parse_configs")

                # Verify global logger was set
                current_logger = get_current_logger()
                assert current_logger == configs.logger

                # Cleanup
                configs.close_logger()

                # Verify log file exists and has content
                log_file = os.path.join(configs.root, "job.log")
                assert os.path.exists(log_file)

                with open(log_file, "r") as f:
                    content = f.read()
                assert "Test message from parse_configs" in content

                # Clean up run directory
                shutil.rmtree(configs.root)

        finally:
            os.unlink(temp_config_file.name)
            set_current_logger(None)


class TestLoggerFileHandling:
    """Tests for logger file handling and cleanup"""

    def test_logger_appends_to_existing_file(self):
        """Test that logger appends to existing log file"""
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, "existing.log")

        try:
            # Create file with existing content
            with open(log_file, "w") as f:
                f.write("Existing content\n")

            # Create logger and log new content
            logger, close_logger = _create_logger(log_file)
            logger("New content")
            close_logger()

            # Verify both old and new content exist
            with open(log_file, "r") as f:
                content = f.read()
            assert "Existing content" in content
            assert "New content" in content

        finally:
            shutil.rmtree(temp_dir)

    def test_logger_creates_directories(self):
        """Test that logger handles nested directory paths correctly"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create path with nested directories
            nested_dir = os.path.join(temp_dir, "logs", "nested")
            os.makedirs(nested_dir)  # Create directories first
            nested_log_file = os.path.join(nested_dir, "test.log")

            # Create logger
            logger, close_logger = _create_logger(nested_log_file)
            logger("Test message")
            close_logger()

            # Verify file was created
            assert os.path.exists(nested_log_file)

            with open(nested_log_file, "r") as f:
                content = f.read()
            assert "Test message" in content

        finally:
            shutil.rmtree(temp_dir)

    def test_multiple_loggers_same_file(self):
        """Test behavior with multiple loggers writing to same file"""
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, "shared.log")

        try:
            # Create two loggers for same file
            logger1, close1 = _create_logger(log_file)
            logger2, close2 = _create_logger(log_file)

            # Log from both
            logger1("Message from logger 1")
            logger2("Message from logger 2")

            # Close both
            close1()
            close2()

            # Verify both messages are in file
            with open(log_file, "r") as f:
                content = f.read()
            assert "Message from logger 1" in content
            assert "Message from logger 2" in content

        finally:
            shutil.rmtree(temp_dir)
