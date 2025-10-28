"""
Tests for training and validation loop functions
"""

from unittest.mock import MagicMock, patch

import jsonargparse
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from framework.loops import train_one_epoch, val_one_epoch


class TestTrainOneEpoch:
    """Tests for the train_one_epoch function"""

    def create_mock_components(self, num_classes=10, batch_size=4, num_batches=3):
        """Helper to create mock model, dataloader, optimizer, and scheduler"""
        # Create mock model
        model = MagicMock()
        model.train = MagicMock()

        # Create mock loss with proper gradient handling
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5
        mock_loss.backward = MagicMock()

        # Mock model output
        mock_output = {
            "logits": torch.randn(batch_size, num_classes),
            "loss": mock_loss,
            "encoding": torch.randn(batch_size, 512),
        }
        model.return_value = mock_output

        # Create mock dataloader with tensor data
        images = torch.randn(num_batches * batch_size, 3, 224, 224)
        labels = torch.randint(0, num_classes, (num_batches * batch_size,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Create mock optimizer and scheduler
        optimizer = MagicMock()
        scheduler = MagicMock()
        scheduler.get_last_lr.return_value = [0.001]

        return model, dataloader, optimizer, scheduler, mock_output

    def test_train_one_epoch_basic_functionality(self):
        """Test basic training loop functionality"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10
        configs.no_tqdm = True  # Disable progress bar for testing

        model, dataloader, optimizer, scheduler, _ = self.create_mock_components()

        with patch("torch.cuda.is_available", return_value=False), patch(
            "framework.loops.MulticlassAccuracy"
        ) as mock_accuracy_class:

            # Mock the accuracy metric
            mock_accuracy = MagicMock()
            mock_accuracy.compute.return_value = torch.tensor(0.85)
            mock_accuracy.to.return_value = mock_accuracy
            mock_accuracy_class.return_value = mock_accuracy

            result = train_one_epoch(model, dataloader, optimizer, scheduler, configs)

            # Check that model was set to train mode
            model.train.assert_called_once()

            # Check that optimizer and scheduler were called
            assert optimizer.zero_grad.call_count == len(dataloader)
            assert optimizer.step.call_count == len(dataloader)
            assert scheduler.step.call_count == len(dataloader)

            # Check return structure
            assert "train_acc" in result
            assert "train_loss" in result
            assert "learning_rate" in result

            # Check that the mocked accuracy is returned (with floating point tolerance)
            assert abs(result["train_acc"] - 0.85) < 1e-6
            assert result["learning_rate"] == 0.001
            assert isinstance(result["train_loss"], float)

    def test_train_one_epoch_cuda_handling(self):
        """Test CUDA device handling"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10
        configs.no_tqdm = True

        model, dataloader, optimizer, scheduler, _ = self.create_mock_components()

        with patch("torch.cuda.is_available", return_value=False), patch(
            "framework.loops.MulticlassAccuracy"
        ) as mock_accuracy_class:

            mock_accuracy = MagicMock()
            mock_accuracy.compute.return_value = torch.tensor(0.75)
            mock_accuracy.to.return_value = mock_accuracy
            mock_accuracy_class.return_value = mock_accuracy

            _ = train_one_epoch(model, dataloader, optimizer, scheduler, configs)

            # Check that accuracy metric was moved to the device
            mock_accuracy.to.assert_called_once()

    def test_train_one_epoch_accuracy_updates(self):
        """Test that accuracy metric is properly updated"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 5
        configs.no_tqdm = True

        model, dataloader, optimizer, scheduler, _ = self.create_mock_components(
            num_classes=5
        )

        with patch("torch.cuda.is_available", return_value=False), patch(
            "framework.loops.MulticlassAccuracy"
        ) as mock_accuracy_class:

            mock_accuracy = MagicMock()
            mock_accuracy.compute.return_value = torch.tensor(0.95)
            mock_accuracy.to.return_value = mock_accuracy
            mock_accuracy_class.return_value = mock_accuracy

            train_one_epoch(model, dataloader, optimizer, scheduler, configs)

            # Check that accuracy was updated for each batch
            assert mock_accuracy.update.call_count == len(dataloader)

            # Check that accuracy was initialized with correct num_classes
            mock_accuracy_class.assert_called_once_with(num_classes=5)

    def test_train_one_epoch_tqdm_configuration(self):
        """Test tqdm progress bar configuration"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10
        configs.no_tqdm = False  # Enable tqdm

        model, dataloader, optimizer, scheduler, _ = self.create_mock_components()

        with patch("torch.cuda.is_available", return_value=False), patch(
            "framework.loops.MulticlassAccuracy"
        ) as mock_accuracy_class, patch("framework.loops.tqdm") as mock_tqdm:

            mock_accuracy = MagicMock()
            mock_accuracy.compute.return_value = torch.tensor(0.80)
            mock_accuracy.to.return_value = mock_accuracy
            mock_accuracy_class.return_value = mock_accuracy

            # Mock tqdm to return the original dataloader
            mock_tqdm.return_value = dataloader

            train_one_epoch(model, dataloader, optimizer, scheduler, configs)

            # Check that tqdm was called with correct parameters
            mock_tqdm.assert_called_once()
            call_args = mock_tqdm.call_args
            assert call_args[1]["desc"] == "train "
            assert call_args[1]["disable"] == False
            assert call_args[1]["dynamic_ncols"] == True


class TestValOneEpoch:
    """Tests for the val_one_epoch function"""

    def create_mock_components(self, num_classes=10, batch_size=4, num_batches=2):
        """Helper to create mock model and dataloader for validation"""
        # Create mock model
        model = MagicMock()
        model.eval = MagicMock()

        # Create mock loss
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.3

        # Mock model output
        mock_output = {
            "logits": torch.randn(batch_size, num_classes),
            "loss": mock_loss,
            "encoding": torch.randn(batch_size, 512),
        }
        model.return_value = mock_output

        # Create mock dataloader with tensor data
        images = torch.randn(num_batches * batch_size, 3, 224, 224)
        labels = torch.randint(0, num_classes, (num_batches * batch_size,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return model, dataloader, mock_output

    def test_val_one_epoch_basic_functionality(self):
        """Test basic validation loop functionality"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10
        configs.no_tqdm = True

        model, dataloader, _ = self.create_mock_components()

        with patch("torch.cuda.is_available", return_value=False), patch(
            "framework.loops.MulticlassAccuracy"
        ) as mock_accuracy_class:

            mock_accuracy = MagicMock()
            mock_accuracy.compute.return_value = torch.tensor(0.88)
            mock_accuracy.to.return_value = mock_accuracy
            mock_accuracy_class.return_value = mock_accuracy

            result = val_one_epoch(model, dataloader, configs)

            # Check that model was set to eval mode
            model.eval.assert_called_once()

            # Check return structure
            assert "val_acc" in result
            assert "val_loss" in result

            # Check that the mocked accuracy is returned (with floating point tolerance)
            assert abs(result["val_acc"] - 0.88) < 1e-6
            assert isinstance(result["val_loss"], float)

    def test_val_one_epoch_accuracy_updates(self):
        """Test that accuracy metric is properly updated during validation"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 5
        configs.no_tqdm = True

        model, dataloader, _ = self.create_mock_components(num_classes=5)

        with patch("torch.cuda.is_available", return_value=False), patch(
            "framework.loops.MulticlassAccuracy"
        ) as mock_accuracy_class:

            mock_accuracy = MagicMock()
            mock_accuracy.compute.return_value = torch.tensor(0.92)
            mock_accuracy.to.return_value = mock_accuracy
            mock_accuracy_class.return_value = mock_accuracy

            val_one_epoch(model, dataloader, configs)

            # Check that accuracy was updated for each batch
            assert mock_accuracy.update.call_count == len(dataloader)

            # Check that accuracy was initialized with correct num_classes
            mock_accuracy_class.assert_called_once_with(num_classes=5)

    def test_val_one_epoch_loss_accumulation(self):
        """Test that validation loss is properly accumulated"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10
        configs.no_tqdm = True

        model, dataloader, mock_output = self.create_mock_components()

        # Mock loss values for each batch
        loss_values = [0.2, 0.4]
        mock_losses = []
        for val in loss_values:
            mock_loss = MagicMock()
            mock_loss.item.return_value = val
            mock_losses.append(mock_loss)

        def side_effect(*_args, **_kwargs):
            output = mock_output.copy()
            output["loss"] = mock_losses.pop(0)
            return output

        model.side_effect = side_effect

        with patch("torch.cuda.is_available", return_value=False), patch(
            "framework.loops.MulticlassAccuracy"
        ) as mock_accuracy_class:

            mock_accuracy = MagicMock()
            mock_accuracy.compute.return_value = torch.tensor(0.90)
            mock_accuracy.to.return_value = mock_accuracy
            mock_accuracy_class.return_value = mock_accuracy

            result = val_one_epoch(model, dataloader, configs)

            # Check that loss is the average of batch losses
            expected_loss = sum(loss_values) / len(loss_values)
            assert abs(result["val_loss"] - expected_loss) < 1e-6

    def test_val_one_epoch_tqdm_disabled(self):
        """Test validation with tqdm disabled"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10
        configs.no_tqdm = True

        model, dataloader, _ = self.create_mock_components()

        with patch("torch.cuda.is_available", return_value=False), patch(
            "framework.loops.MulticlassAccuracy"
        ) as mock_accuracy_class, patch("framework.loops.tqdm") as mock_tqdm:

            mock_accuracy = MagicMock()
            mock_accuracy.compute.return_value = torch.tensor(0.87)
            mock_accuracy.to.return_value = mock_accuracy
            mock_accuracy_class.return_value = mock_accuracy

            # Mock tqdm to return the original dataloader
            mock_tqdm.return_value = dataloader

            val_one_epoch(model, dataloader, configs)

            # Check that tqdm was called with disable=True
            mock_tqdm.assert_called_once()
            call_args = mock_tqdm.call_args
            assert call_args[1]["disable"] == True


class TestLoopsIntegration:
    """Integration tests for training and validation loops"""

    def test_train_val_consistency(self):
        """Test that training and validation return consistent metric structures"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10
        configs.no_tqdm = True

        # Create simple mock model instead of Sequential
        model = MagicMock()
        model.train = MagicMock()
        model.eval = MagicMock()

        # Mock model output
        mock_output = {
            "logits": torch.randn(4, 10),
            "encoding": torch.randn(4, 128),
            "loss": MagicMock(),
        }
        mock_output["loss"].item.return_value = 0.5
        mock_output["loss"].backward = MagicMock()
        model.return_value = mock_output

        # Create simple dataloader
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 10, (8,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        # Create optimizer and scheduler
        optimizer = MagicMock()
        scheduler = MagicMock()
        scheduler.get_last_lr.return_value = [0.001]

        with patch("torch.cuda.is_available", return_value=False), patch(
            "framework.loops.MulticlassAccuracy"
        ) as mock_accuracy_class:

            mock_accuracy = MagicMock()
            mock_accuracy.compute.return_value = torch.tensor(0.85)
            mock_accuracy.to.return_value = mock_accuracy
            mock_accuracy_class.return_value = mock_accuracy

            # Run training
            train_results = train_one_epoch(
                model, dataloader, optimizer, scheduler, configs
            )

            # Run validation
            val_results = val_one_epoch(model, dataloader, configs)

            # Check that both return dictionaries with expected keys
            assert isinstance(train_results, dict)
            assert isinstance(val_results, dict)

            assert "train_acc" in train_results
            assert "train_loss" in train_results
            assert "learning_rate" in train_results

            assert "val_acc" in val_results
            assert "val_loss" in val_results

    def test_different_num_classes(self):
        """Test loops with different numbers of classes"""
        num_classes_list = [2, 5, 10, 100]

        for num_classes in num_classes_list:
            configs = jsonargparse.Namespace()
            configs.data = jsonargparse.Namespace()
            configs.data.num_classes = num_classes
            configs.no_tqdm = True

            model = MagicMock()
            model.train = MagicMock()
            model.eval = MagicMock()

            # Create mock loss with proper backward
            mock_loss = MagicMock()
            mock_loss.item.return_value = 0.5
            mock_loss.backward = MagicMock()

            mock_output = {
                "logits": torch.randn(2, num_classes),
                "loss": mock_loss,
                "encoding": torch.randn(2, 128),
            }
            model.return_value = mock_output

            # Create simple dataloader
            images = torch.randn(4, 3, 32, 32)
            labels = torch.randint(0, num_classes, (4,))
            dataset = TensorDataset(images, labels)
            dataloader = DataLoader(dataset, batch_size=2)

            optimizer = MagicMock()
            scheduler = MagicMock()
            scheduler.get_last_lr.return_value = [0.001]

            with patch("torch.cuda.is_available", return_value=False), patch(
                "framework.loops.MulticlassAccuracy"
            ) as mock_accuracy_class:

                mock_accuracy = MagicMock()
                mock_accuracy.compute.return_value = torch.tensor(0.85)
                mock_accuracy.to.return_value = mock_accuracy
                mock_accuracy_class.return_value = mock_accuracy

                # Test training
                train_results = train_one_epoch(
                    model, dataloader, optimizer, scheduler, configs
                )
                assert isinstance(train_results, dict)

                # Test validation
                val_results = val_one_epoch(model, dataloader, configs)
                assert isinstance(val_results, dict)

                # Check that accuracy was initialized with correct num_classes
                calls = mock_accuracy_class.call_args_list
                for call_args in calls:
                    assert call_args[1]["num_classes"] == num_classes


class TestLoopsErrorHandling:
    """Tests for error handling in loops"""

    def test_train_missing_config_attributes(self):
        """Test training loop with missing config attributes"""
        configs = jsonargparse.Namespace()
        # Missing data.num_classes

        model = MagicMock()
        dataloader = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()

        with pytest.raises(AttributeError):
            train_one_epoch(model, dataloader, optimizer, scheduler, configs)

    def test_val_missing_config_attributes(self):
        """Test validation loop with missing config attributes"""
        configs = jsonargparse.Namespace()
        # Missing data.num_classes

        model = MagicMock()
        dataloader = MagicMock()

        with pytest.raises(AttributeError):
            val_one_epoch(model, dataloader, configs)

    def test_empty_dataloader_handling(self):
        """Test loops with empty dataloaders"""
        configs = jsonargparse.Namespace()
        configs.data = jsonargparse.Namespace()
        configs.data.num_classes = 10
        configs.no_tqdm = True

        model = MagicMock()

        # Empty dataloader
        empty_dataset = TensorDataset(
            torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
        )
        empty_dataloader = DataLoader(empty_dataset, batch_size=1)

        optimizer = MagicMock()
        scheduler = MagicMock()
        scheduler.get_last_lr.return_value = [0.001]

        with patch("torch.cuda.is_available", return_value=False), patch(
            "framework.loops.MulticlassAccuracy"
        ) as mock_accuracy_class:

            mock_accuracy = MagicMock()
            mock_accuracy.compute.return_value = torch.tensor(
                float("nan")
            )  # No data processed
            mock_accuracy.to.return_value = mock_accuracy
            mock_accuracy_class.return_value = mock_accuracy

            # For training, need to handle division by zero
            with pytest.raises(ZeroDivisionError):
                train_one_epoch(model, empty_dataloader, optimizer, scheduler, configs)

            # For validation, also expect division by zero
            with pytest.raises(ZeroDivisionError):
                val_one_epoch(model, empty_dataloader, configs)
