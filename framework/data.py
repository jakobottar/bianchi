"""
datasets and dataset building stuff
handles loading datasets, applying transforms, building dataloaders

includes standard datasets and allows for adding more
"""

from collections.abc import Callable

import torch
import torchvision.transforms.v2 as transforms
from jsonargparse import Namespace
from torchvision import datasets


def build_datasets(configs: Namespace) -> dict:
    """Builds datasets and returns them in a dict"""

    match configs.data.name.lower():
        case "mnist":
            return _build_mnist_datasets(configs)
        case "cifar10":
            return _build_cifar10_datasets(configs)
        case _:
            raise ValueError(f"Dataset {configs.data.name} not supported")

        # ========================================
        # TEMPLATE: Adding new datasets
        # ========================================
        # To add a new dataset:
        #
        # 1. Create a builder function following the template patterns below
        #    - Use _build_template_datasets as a starting point
        #    - For custom datasets, use _build_custom_dataset_template
        #    - Follow the return format: {"train": dataset, "test": dataset, "num_classes": int}
        #
        # 2. Add a new case here:
        #    case "imagenet":
        #        return _build_imagenet_datasets(configs)
        #    case "coco":
        #        return _build_coco_datasets(configs)
        #    case "custom_dataset":
        #        return _build_custom_dataset(configs)
        #
        # Example cases (uncomment and implement as needed):
        # case "imagenet":
        #     return _build_imagenet_datasets(configs)
        # case "coco":
        #     return _build_coco_datasets(configs)
        # case "places365":
        #     return _build_places365_datasets(configs)
        # case "food101":
        #     return _build_food101_datasets(configs)


def build_dataloaders(configs: Namespace, dataset_map: dict | None) -> dict:
    """Builds dataloaders from datasets"""

    if dataset_map is None:
        dataset_map = build_datasets(configs)

    train_loader = torch.utils.data.DataLoader(
        dataset_map["train"],
        batch_size=configs.data.batch_size,
        shuffle=True,
        num_workers=configs.data.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_map["test"],
        batch_size=configs.data.batch_size,
        shuffle=False,
        num_workers=configs.data.workers,
    )

    return {"train": train_loader, "test": test_loader}


def _build_mnist_datasets(configs: Namespace) -> dict:
    """Builds MNIST datasets"""

    transform = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )

    train_dataset = datasets.MNIST(
        root=configs.data.root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=configs.data.root, train=False, download=True, transform=transform
    )

    return {"train": train_dataset, "test": test_dataset, "num_classes": 10}


def _build_cifar10_datasets(configs: Namespace) -> dict:
    """Builds CIFAR-10 datasets"""

    transform = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )

    train_dataset = datasets.CIFAR10(
        root=configs.data.root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=configs.data.root, train=False, download=True, transform=transform
    )

    return {"train": train_dataset, "test": test_dataset, "num_classes": 10}


# ========================================
# TEMPLATE FUNCTIONS FOR NEW DATASETS
# ========================================
# Copy and modify these templates when adding new datasets
# Make sure to follow the same pattern and return structure


def _build_template_datasets(configs: Namespace) -> dict:
    """
    TEMPLATE: Builds [DATASET_NAME] datasets

    Replace [DATASET_NAME] with your actual dataset name (e.g., ImageNet, COCO, etc.)

    Args:
        configs: Configuration namespace containing:
            - configs.data.root: str - Root directory for dataset storage
            - Any additional dataset-specific configs you need

    Returns:
        dict: Dictionary with keys:
            - "train": torch.utils.data.Dataset - Training dataset
            - "test": torch.utils.data.Dataset - Test/validation dataset
            - "num_classes": int - Number of classes in the dataset

    Usage:
        1. Copy this function and rename it (e.g., _build_imagenet_datasets)
        2. Update the transforms for your dataset's requirements
        3. Replace the dataset class and parameters
        4. Update num_classes to match your dataset
        5. Add your case to the build_datasets() match statement
    """

    # Define transforms for your dataset
    # Modify these based on your dataset's requirements:
    # - Image size (resize, crop)
    # - Normalization values
    # - Data augmentation for training
    # - Any dataset-specific preprocessing
    train_transform = transforms.Compose(
        [
            # Example transforms - modify for your dataset:
            transforms.ToImage(),  # Convert to tensor format
            transforms.ToDtype(
                torch.float32, scale=True
            ),  # Convert to float32 and scale [0,1]
            # transforms.Resize((224, 224)),                # Resize if needed
            # transforms.RandomHorizontalFlip(),            # Data augmentation
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet normalization
            #                     std=[0.229, 0.224, 0.225])
        ]
    )

    # Test transform (usually no augmentation)
    test_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            # Include same preprocessing as train but no augmentation
            # transforms.Resize((224, 224)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ]
    )

    # Create training dataset
    # Replace with your actual dataset class and parameters
    train_dataset = None  # Replace with: datasets.YourDataset(
    # root=configs.data.root,
    # train=True,
    # download=True,  # Set to False if manual download required
    # transform=train_transform,
    # Any other dataset-specific parameters
    # )

    # Create test/validation dataset
    test_dataset = None  # Replace with: datasets.YourDataset(
    # root=configs.data.root,
    # train=False,  # or split='val'/'test' depending on dataset
    # download=True,
    # transform=test_transform,
    # Any other dataset-specific parameters
    # )

    # Update this number to match your dataset's class count
    num_classes = 1000  # Replace with actual number of classes

    return {"train": train_dataset, "test": test_dataset, "num_classes": num_classes}


def _build_custom_dataset_template(configs: Namespace) -> dict:
    """
    TEMPLATE: For custom datasets not available in torchvision.datasets

    Use this template when you need to create a custom dataset class
    or load data from custom file formats.

    Args:
        configs: Configuration namespace, may include:
            - configs.data.root: str - Root directory for dataset
            - configs.data.train_dir: str - Training data directory
            - configs.data.test_dir: str - Test data directory
            - configs.data.annotations_file: str - Path to annotations
            - Any other custom parameters your dataset needs

    Returns:
        dict: Same structure as other dataset builders
    """

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            # Add dataset-specific transforms
        ]
    )

    # Option 1: Use ImageFolder for directory-based datasets
    # train_dataset = datasets.ImageFolder(
    #     root=os.path.join(configs.data.root, "train"),
    #     transform=transform
    # )
    # test_dataset = datasets.ImageFolder(
    #     root=os.path.join(configs.data.root, "test"),
    #     transform=transform
    # )

    # Option 2: Create custom dataset class (see template below)
    # train_dataset = CustomDatasetTemplate(
    #     data_dir=configs.data.train_dir,
    #     annotations=configs.data.annotations_file,
    #     transform=transform,
    #     split='train'
    # )

    # For now, return None to avoid errors
    return {"train": None, "test": None, "num_classes": 0}  # Update with actual number


# ========================================
# TEMPLATE CUSTOM DATASET CLASS
# ========================================


class CustomDatasetTemplate(torch.utils.data.Dataset):
    """
    TEMPLATE: Custom dataset class for non-standard data formats

    Use this template when torchvision.datasets doesn't have your dataset
    or when you need custom data loading logic.

    This class should be moved to a separate file if it becomes complex.

    Args:
        data_dir (str): Directory containing the data files
        annotations (str): Path to annotations file (labels, bboxes, etc.)
        transform (callable, optional): Transform to apply to samples
        split (str): Data split ('train', 'val', 'test')
    """

    def __init__(
        self,
        data_dir: str,
        annotations: str | None = None,
        transform: Callable | None = None,
        split: str = "train",
    ):
        """
        Initialize the custom dataset

        Modify this constructor based on your data format:
        - CSV files with image paths and labels
        - JSON annotations
        - Directory structure parsing
        - Database connections
        - etc.
        """
        self.data_dir = data_dir
        self.annotations = annotations
        self.transform = transform
        self.split = split

        # Load and parse your data here
        # Example for CSV-based dataset:
        # self.data_list = self._load_data_list()
        # self.classes = self._get_class_names()
        # self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Placeholder - replace with actual data loading
        self.data_list = []
        self.classes = []

    def __len__(self) -> int:
        """Return the total number of samples in the dataset"""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample from the dataset

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (image, label) where image is the processed sample
                   and label is the corresponding class index
        """
        # Load your data sample here
        # Example:
        # sample_info = self.data_list[idx]
        # image_path = os.path.join(self.data_dir, sample_info['image'])
        # image = Image.open(image_path).convert('RGB')
        # label = sample_info['label']

        # Apply transforms if specified
        # if self.transform:
        #     image = self.transform(image)

        # For template, return dummy data
        image = torch.zeros((3, 224, 224))  # Replace with actual image loading
        label = 0  # Replace with actual label

        return image, label

    def _load_data_list(self) -> list:
        """
        Private method to load and parse the data file list

        Modify this method based on your data format:
        - Parse CSV files
        - Read JSON annotations
        - Scan directory structures
        - Query databases
        - etc.

        Returns:
            list: List of dictionaries containing sample information
                  e.g., [{'image': 'path/to/img.jpg', 'label': 0}, ...]
        """
        # Implement your data loading logic here
        return []

    def _get_class_names(self) -> list:
        """
        Get the list of class names for this dataset

        Returns:
            list: List of class names in order of their indices
        """
        # Implement class name extraction
        return []
