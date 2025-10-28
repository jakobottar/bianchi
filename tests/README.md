# Test Coverage Summary for Framework Functions

This document summarizes the test coverage for the bianchi framework functions.

## Test Files Created

1. **`tests/test_config.py`** - Primary test suite for `set_up_configs` function
2. **`tests/test_parse_configs.py`** - Additional tests for `parse_configs` function  
3. **`tests/test_data.py`** - Comprehensive tests for dataset building functions
4. **`tests/test_model.py`** - Comprehensive tests for model building and ResNet implementations
5. **`tests/__init__.py`** - Makes tests directory a Python package
6. **`pyproject.toml`** - Pytest configuration

## Functions Tested

### Config Module (`framework/config.py`)

**`set_up_configs` Function Tests:**
- ✅ Random name generation using `namegenerator.gen()`
- ✅ Custom name handling (preserves user-provided names)
- ✅ Directory creation in the specified root path
- ✅ Config file saving as JSON format
- ✅ Deterministic seed setting (calls `random.seed()` and `torch.manual_seed()`)
- ✅ No seed setting when seed = -1 (skips seed functions)
- ✅ CUDNN deterministic mode activation when seed is set
- ✅ Path object conversion to strings in saved config
- ✅ Config key removal from saved output (avoids circular references)
- ✅ Robust handling of missing config keys
- ✅ Nested namespace handling (e.g., `configs.data.batch_size`)

**`parse_configs` Function Tests:**
- ✅ Default argument parsing with valid config files
- ✅ Command line argument overrides of config file values
- ✅ Integration with `set_up_configs` (full workflow testing)
- ✅ Proper namespace object creation
- ✅ Config file validation and parsing
- ✅ Command line parameter precedence over config file

### Data Module (`framework/data.py`)

**`build_datasets` Function Tests:**
- ✅ MNIST dataset building route selection
- ✅ CIFAR-10 dataset building route selection
- ✅ Unsupported dataset error handling
- ✅ Case insensitive dataset name matching
- ✅ Config validation and error handling

**`_build_mnist_datasets` Function Tests:**
- ✅ Correct dataset structure returned (train/test/num_classes)
- ✅ Proper torchvision.datasets.MNIST calls with correct parameters
- ✅ Transform composition and application
- ✅ Root directory path handling
- ✅ Download parameter configuration

**`_build_cifar10_datasets` Function Tests:**
- ✅ Correct dataset structure returned (train/test/num_classes)
- ✅ Proper torchvision.datasets.CIFAR10 calls with correct parameters
- ✅ Transform composition and application
- ✅ Root directory path handling
- ✅ Download parameter configuration

**`build_dataloaders` Function Tests:**
- ✅ DataLoader creation with provided dataset map
- ✅ DataLoader creation without dataset map (auto-builds datasets)
- ✅ Correct DataLoader parameters (batch_size, num_workers, shuffle)
- ✅ Train dataloader shuffle=True, test dataloader shuffle=False
- ✅ Different batch size configurations
- ✅ Different worker count configurations

**Dataset Integration Tests:**
- ✅ Consistency between MNIST and CIFAR-10 structures
- ✅ Config validation across all dataset types
- ✅ Root path handling with various path formats
- ✅ Error handling for invalid configurations

**Transform Functionality Tests:**
- ✅ Transform composition verification
- ✅ ToTensor transform inclusion
- ✅ Identical transforms for train/test splits

### Model Module (`framework/model.py` and `framework/resnet.py`)

**`build_model` Function Tests:**
- ✅ ResNet18 model building route selection
- ✅ ResNet50 model building route selection
- ✅ Case insensitive architecture name matching
- ✅ Unsupported architecture error handling
- ✅ Config validation and error handling

**`_build_resnet` Function Tests:**
- ✅ ResNet18 creation with correct parameters
- ✅ ResNet50 creation with correct parameters
- ✅ Different number of classes configuration
- ✅ Checkpoint loading functionality
- ✅ Unsupported ResNet architecture error handling

**`ResNet18` Class Tests:**
- ✅ Proper initialization (loss function, feature size, output classes)
- ✅ Forward pass without targets (returns logits, encoding, loss=None)
- ✅ Forward pass with targets (computes and returns loss)
- ✅ Weight initialization validation (Kaiming normal, batch norm constants)

**`ResNet50` Class Tests:**
- ✅ Proper initialization (loss function, feature size, output classes)
- ✅ Forward pass without targets (returns logits, encoding, loss=None)
- ✅ Forward pass with targets (computes and returns loss)
- ✅ Different input batch sizes handling

**Model Integration Tests:**
- ✅ End-to-end model building workflow
- ✅ Consistency across different ResNet architectures
- ✅ Checkpoint loading integration with real files
- ✅ Model interface standardization (logits, loss, encoding output format)

## Test Features

**Isolation & Cleanup:**
- All tests use temporary directories for file operations
- Proper cleanup using try/finally blocks
- No side effects between tests

**Mocking & Patching:**
- Mock external dependencies (`namegenerator.gen`, `random.seed`, `torch.manual_seed`)
- Patch system arguments for command line testing
- Isolated testing without affecting global state

**Edge Cases:**
- Missing configuration keys
- Path object handling
- Seed value edge cases (-1 vs positive integers)
- Directory creation with existing paths

## Running the Tests

To run all tests:
```bash
/home/jakobj/.micromamba/envs/edo/bin/python -m pytest tests/ -v
```

To run specific test files:
```bash
/home/jakobj/.micromamba/envs/edo/bin/python -m pytest tests/test_config.py -v
/home/jakobj/.micromamba/envs/edo/bin/python -m pytest tests/test_parse_configs.py -v
```

## Test Results

- **Total Tests:** 62
- **Config Tests:** 14 (11 + 3)
- **Data Tests:** 24 (18 dataset + 6 dataloader)
- **Model Tests:** 24 (5 build_model + 5 _build_resnet + 4 ResNet18 + 4 ResNet50 + 3 integration + 3 error handling)
- **Status:** All passing ✅
- **Coverage:** Config, Data, and Model modules comprehensively tested
- **Execution Time:** ~8.5 seconds for full suite

## Test Categories

### Unit Tests
- Individual function testing with mocked dependencies
- Parameter validation and error handling
- Return value structure verification

### Integration Tests  
- End-to-end workflow testing
- Cross-function compatibility
- Real filesystem operations (with cleanup)

### Error Handling Tests
- Invalid input validation
- Missing configuration handling
- Filesystem permission errors
- Unsupported dataset types

## Notable Test Features

**Comprehensive Mocking:**
- Mock torchvision datasets to avoid actual downloads
- Mock filesystem operations for controlled testing
- Mock external dependencies (namegenerator, torch functions)

**Edge Case Coverage:**
- Case sensitivity in dataset names
- Various root path formats and invalid paths
- Missing configuration attributes
- Transform composition and validation

**Isolation & Safety:**
- All tests use temporary directories
- Proper cleanup with try/finally blocks
- No side effects between tests
- No actual dataset downloads during testing

**Real-world Scenarios:**
- Command line argument parsing with real sys.argv mocking
- Config file loading and validation
- Directory creation and JSON serialization
- Transform pipeline verification
- Model forward passes with realistic input sizes
- Checkpoint loading and state dict management
- PyTorch model initialization and weight verification