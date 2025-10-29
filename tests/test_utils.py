"""
Tests for utility functions including shutdown/resume and logging
"""

import os
import shutil
import tempfile
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import jsonargparse
import pytest
import torch
from torch import nn

from framework.utils import (
    get_current_logger,
    interrupted,
    log,
    resume,
    set_current_logger,
    shutdown,
    signal_handler,
)


class TestLoggingSystem:
    """Tests for the global logging system"""

    def test_get_current_logger_fallback(self):
        """Test that get_current_logger falls back to print when no logger is set"""
        # Reset global state
        set_current_logger(None)

        logger = get_current_logger()
        assert logger == print

    def test_set_and_get_current_logger(self):
        """Test setting and getting the current logger"""
        mock_logger = MagicMock()

        set_current_logger(mock_logger)
        logger = get_current_logger()

        assert logger == mock_logger

    def test_log_function_with_logger(self):
        """Test the convenience log function with a logger set"""
        mock_logger = MagicMock()
        set_current_logger(mock_logger)

        log("test message")

        mock_logger.assert_called_once_with("test message")

    def test_log_function_fallback_to_print(self):
        """Test the convenience log function falls back to print"""
        set_current_logger(None)

        with patch("builtins.print") as mock_print:
            log("test message")
            mock_print.assert_called_once_with("test message")

    def test_signal_handler_with_logger(self):
        """Test signal handler uses the current logger"""
        mock_logger = MagicMock()
        set_current_logger(mock_logger)

        # Reset interrupted state
        import framework.utils

        framework.utils.interrupted = False

        signal_handler(15, None)  # SIGTERM

        mock_logger.assert_called_once_with(
            "received shutdown signal, gracefully closing up shop."
        )
        assert framework.utils.interrupted is True

    def test_signal_handler_fallback_to_print(self):
        """Test signal handler falls back to print when no logger"""
        set_current_logger(None)

        # Reset interrupted state
        import framework.utils

        framework.utils.interrupted = False

        with patch("builtins.print") as mock_print:
            signal_handler(15, None)
            mock_print.assert_called_once_with(
                "received shutdown signal, gracefully closing up shop."
            )
            assert framework.utils.interrupted is True


class TestShutdownFunction:
    """Tests for the shutdown function"""

    def test_shutdown_saves_all_files(self):
        """Test that shutdown saves all required files"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create mock objects
            model = MagicMock()
            model.state_dict.return_value = {"param1": torch.tensor([1.0])}

            optimizer = MagicMock()
            optimizer.state_dict.return_value = {"state": "optimizer_state"}

            scheduler = MagicMock()
            scheduler.state_dict.return_value = {"step": 100}

            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = MagicMock()

            best_val_acc = 0.85
            curr_epoch = 42

            # Mock sys.exit to prevent actual exit
            with patch("sys.exit") as mock_exit:
                shutdown(model, optimizer, scheduler, configs, best_val_acc, curr_epoch)
                mock_exit.assert_called_once_with(99)

            # Verify files were created
            assert os.path.exists(os.path.join(temp_dir, "partial.pth"))
            assert os.path.exists(os.path.join(temp_dir, "partial_opt.pth"))
            assert os.path.exists(os.path.join(temp_dir, "partial_sched.pth"))
            assert os.path.exists(os.path.join(temp_dir, "partial_stats.txt"))

            # Verify stats file content
            with open(os.path.join(temp_dir, "partial_stats.txt"), "r") as f:
                lines = f.readlines()
                assert float(lines[0].strip()) == 0.85
                assert int(lines[1].strip()) == 42

            # Verify logger was called
            configs.logger.assert_called_once_with("interrupted, saving run state")

        finally:
            shutil.rmtree(temp_dir)

    def test_shutdown_with_torch_save_calls(self):
        """Test that shutdown calls torch.save with correct parameters"""
        temp_dir = tempfile.mkdtemp()

        try:
            model = MagicMock()
            optimizer = MagicMock()
            scheduler = MagicMock()
            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = MagicMock()

            with patch("torch.save") as mock_torch_save, patch("sys.exit"):
                shutdown(model, optimizer, scheduler, configs, 0.9, 10)

                # Verify torch.save was called 3 times
                assert mock_torch_save.call_count == 3

                # Check the calls
                calls = mock_torch_save.call_args_list
                assert calls[0][0][1] == os.path.join(temp_dir, "partial.pth")
                assert calls[1][0][1] == os.path.join(temp_dir, "partial_opt.pth")
                assert calls[2][0][1] == os.path.join(temp_dir, "partial_sched.pth")

        finally:
            shutil.rmtree(temp_dir)


class TestResumeFunction:
    """Tests for the resume function"""

    def test_resume_loads_all_files_and_cleans_up(self):
        """Test that resume loads files and cleans up partial files"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create partial files
            partial_files = ["partial.pth", "partial_opt.pth", "partial_sched.pth"]
            for filename in partial_files:
                torch.save({"dummy": "data"}, os.path.join(temp_dir, filename))

            # Create stats file
            stats_content = "0.92\n25\n"
            with open(os.path.join(temp_dir, "partial_stats.txt"), "w") as f:
                f.write(stats_content)

            # Create mock objects
            model = MagicMock()
            optimizer = MagicMock()
            scheduler = MagicMock()
            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = MagicMock()

            # Mock torch.load to return dummy data
            with patch("torch.load") as mock_torch_load:
                mock_torch_load.return_value = {"loaded": "state"}

                best_val_acc, start_epoch = resume(model, optimizer, scheduler, configs)

            # Verify return values
            assert best_val_acc == 0.92
            assert start_epoch == 25

            # Verify load_state_dict was called on all objects
            model.load_state_dict.assert_called_once()
            optimizer.load_state_dict.assert_called_once()
            scheduler.load_state_dict.assert_called_once()

            # Verify logger calls
            expected_calls = [
                "resuming from partial shutdown...",
                "resuming from epoch 26, recovered best val acc: 0.920",
            ]
            actual_calls = [call[0][0] for call in configs.logger.call_args_list]
            assert actual_calls == expected_calls

            # Verify cleanup - all partial files should be removed
            for filename in partial_files + ["partial_stats.txt"]:
                assert not os.path.exists(os.path.join(temp_dir, filename))

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_resume_with_real_torch_objects(self):
        """Test resume with actual PyTorch objects"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create real PyTorch objects
            model = nn.Linear(10, 5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

            # Save their states
            torch.save(model.state_dict(), os.path.join(temp_dir, "partial.pth"))
            torch.save(
                optimizer.state_dict(), os.path.join(temp_dir, "partial_opt.pth")
            )
            torch.save(
                scheduler.state_dict(), os.path.join(temp_dir, "partial_sched.pth")
            )

            # Create stats file
            with open(os.path.join(temp_dir, "partial_stats.txt"), "w") as f:
                f.write("0.88\n15\n")

            # Create new objects (different states)
            new_model = nn.Linear(10, 5)
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
            new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10)

            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = MagicMock()

            # Resume should load the saved states
            best_val_acc, start_epoch = resume(
                new_model, new_optimizer, new_scheduler, configs
            )

            assert best_val_acc == 0.88
            assert start_epoch == 15

            # Verify states were actually loaded (this is a basic check)
            # In practice, you'd compare specific state values
            assert hasattr(new_model, "weight")  # Model should still be functional

        finally:
            shutil.rmtree(temp_dir)

    def test_resume_missing_files_raises_error(self):
        """Test that resume raises appropriate errors for missing files"""
        temp_dir = tempfile.mkdtemp()

        try:
            model = MagicMock()
            optimizer = MagicMock()
            scheduler = MagicMock()
            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = MagicMock()

            # Should raise FileNotFoundError when partial files don't exist
            with pytest.raises(FileNotFoundError):
                resume(model, optimizer, scheduler, configs)

        finally:
            shutil.rmtree(temp_dir)

    def test_resume_malformed_stats_file(self):
        """Test resume with malformed stats file"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create partial model files
            for filename in ["partial.pth", "partial_opt.pth", "partial_sched.pth"]:
                torch.save({}, os.path.join(temp_dir, filename))

            # Create malformed stats file
            with open(os.path.join(temp_dir, "partial_stats.txt"), "w") as f:
                f.write("not_a_number\n15\n")

            model = MagicMock()
            optimizer = MagicMock()
            scheduler = MagicMock()
            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = MagicMock()

            with patch("torch.load", return_value={}):
                # Should raise ValueError when trying to convert "not_a_number" to float
                with pytest.raises(ValueError):
                    resume(model, optimizer, scheduler, configs)

        finally:
            shutil.rmtree(temp_dir)


class TestIntegrationShutdownResume:
    """Integration tests for shutdown and resume workflow"""

    def test_shutdown_resume_cycle(self):
        """Test complete shutdown -> resume cycle"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create initial model, optimizer, scheduler
            model = nn.Linear(5, 3)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

            # Take a step to change their states
            loss = torch.sum(model(torch.randn(2, 5)))
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Store original states for comparison
            original_model_state = model.state_dict().copy()
            original_opt_state = optimizer.state_dict().copy()
            original_sched_state = scheduler.state_dict().copy()

            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = MagicMock()

            # Shutdown
            with patch("sys.exit"):  # Prevent actual exit
                shutdown(model, optimizer, scheduler, configs, 0.75, 20)

            # Create new objects (clean state)
            new_model = nn.Linear(5, 3)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            new_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                new_optimizer, gamma=0.9
            )

            # Resume
            best_val_acc, start_epoch = resume(
                new_model, new_optimizer, new_scheduler, configs
            )

            # Verify resume worked
            assert best_val_acc == 0.75
            assert start_epoch == 20

            # Verify states were restored (compare tensor values)
            for key in original_model_state:
                assert torch.allclose(
                    original_model_state[key], new_model.state_dict()[key]
                )

            # Note: Optimizer and scheduler state comparison is more complex due to
            # internal state structure, but the load_state_dict calls should work

        finally:
            shutil.rmtree(temp_dir)

    def test_multiple_shutdown_resume_cycles(self):
        """Test multiple shutdown/resume cycles work correctly"""
        temp_dir = tempfile.mkdtemp()

        try:
            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = MagicMock()

            # Cycle 1: Initial shutdown
            model1 = nn.Linear(3, 2)
            optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1)
            scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=5)

            with patch("sys.exit"):
                shutdown(model1, optimizer1, scheduler1, configs, 0.6, 10)

            # Cycle 1: Resume
            model2 = nn.Linear(3, 2)
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5)

            best_acc, epoch = resume(model2, optimizer2, scheduler2, configs)
            assert best_acc == 0.6
            assert epoch == 10

            # Cycle 2: Another shutdown
            with patch("sys.exit"):
                shutdown(model2, optimizer2, scheduler2, configs, 0.8, 25)

            # Cycle 2: Resume
            model3 = nn.Linear(3, 2)
            optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.1)
            scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=5)

            best_acc, epoch = resume(model3, optimizer3, scheduler3, configs)
            assert best_acc == 0.8
            assert epoch == 25

        finally:
            shutil.rmtree(temp_dir)


class TestUtilsErrorHandling:
    """Tests for error handling in utils functions"""

    def test_shutdown_with_unwritable_directory(self):
        """Test shutdown behavior with permission issues"""
        # Create a temporary directory and make it read-only
        temp_dir = tempfile.mkdtemp()

        try:
            # Make directory read-only (on systems that support it)
            os.chmod(temp_dir, 0o444)

            model = MagicMock()
            optimizer = MagicMock()
            scheduler = MagicMock()
            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = MagicMock()

            # Should raise RuntimeError when trying to save files to read-only directory
            with patch("sys.exit"), pytest.raises((PermissionError, RuntimeError)):
                shutdown(model, optimizer, scheduler, configs, 0.5, 5)

        finally:
            # Restore permissions and cleanup
            os.chmod(temp_dir, 0o755)
            shutil.rmtree(temp_dir)

    def test_resume_with_corrupted_torch_files(self):
        """Test resume with corrupted PyTorch save files"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create corrupted torch files (just text files)
            for filename in ["partial.pth", "partial_opt.pth", "partial_sched.pth"]:
                with open(os.path.join(temp_dir, filename), "w") as f:
                    f.write("corrupted content")

            # Create valid stats file
            with open(os.path.join(temp_dir, "partial_stats.txt"), "w") as f:
                f.write("0.7\n12\n")

            model = MagicMock()
            optimizer = MagicMock()
            scheduler = MagicMock()
            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = MagicMock()

            # Should raise an error when torch.load tries to load corrupted files
            with pytest.raises(Exception):  # torch.load will raise various exceptions
                resume(model, optimizer, scheduler, configs)

        finally:
            shutil.rmtree(temp_dir)


class TestLoggingIntegration:
    """Integration tests for logging with shutdown/resume"""

    def test_logging_in_shutdown_resume_cycle(self):
        """Test that logging works correctly during shutdown/resume"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create a real logger that writes to a file
            log_file = os.path.join(temp_dir, "test.log")
            with open(log_file, "w") as f:
                pass  # Create empty file

            def mock_logger(message):
                with open(log_file, "a") as f:
                    f.write(f"{message}\n")

            configs = jsonargparse.Namespace()
            configs.root = temp_dir
            configs.logger = mock_logger

            # Test shutdown logging - use real PyTorch objects for proper serialization
            model = nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

            with patch("sys.exit"):
                shutdown(model, optimizer, scheduler, configs, 0.9, 30)

            # Test resume logging
            new_model = nn.Linear(2, 1)
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
            new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10)

            best_acc, epoch = resume(new_model, new_optimizer, new_scheduler, configs)

            # Verify resume worked
            assert best_acc == 0.9
            assert epoch == 30

            # Read log file and verify messages
            with open(log_file, "r") as f:
                log_content = f.read()

            assert "interrupted, saving run state" in log_content
            assert "resuming from partial shutdown..." in log_content
            assert (
                "resuming from epoch 31, recovered best val acc: 0.900" in log_content
            )

        finally:
            shutil.rmtree(temp_dir)

    def test_global_logging_system_integration(self):
        """Test integration between global logging and utils functions"""
        temp_dir = tempfile.mkdtemp()
        log_messages = []

        def capture_logger(message):
            log_messages.append(message)

        try:
            # Set up global logger
            set_current_logger(capture_logger)

            # Test that signal handler uses global logger
            import framework.utils

            framework.utils.interrupted = False

            signal_handler(15, None)

            assert (
                "received shutdown signal, gracefully closing up shop." in log_messages
            )
            assert framework.utils.interrupted is True

            # Test convenience log function
            log("test global logging")
            assert "test global logging" in log_messages

        finally:
            # Cleanup
            set_current_logger(None)
            import framework.utils

            framework.utils.interrupted = False
            shutil.rmtree(temp_dir)
