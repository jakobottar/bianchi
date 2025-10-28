"""
train and test loops
"""

import torch
from jsonargparse import Namespace
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    configs: Namespace,
) -> dict:
    """Trains the model for one epoch"""

    # set model to train mode
    model.train()
    # TODO: figure out how to handle this when we move to multi-GPU training (w/ DDP)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracy = MulticlassAccuracy(num_classes=configs.data.num_classes).to(device)

    num_batches = len(dataloader)
    train_loss = 0.0

    tbar_loader = tqdm(
        dataloader,
        desc="train ",
        total=num_batches,
        dynamic_ncols=True,
        disable=configs.no_tqdm,
    )

    for images, labels in tbar_loader:
        # move images to GPU if needed
        images, labels = images.to(device), labels.to(device)

        # zero gradients from previous step
        optimizer.zero_grad()

        # compute prediction and loss
        out = model(images, labels)
        train_loss += out["loss"].item()

        # backpropagation
        out["loss"].backward()
        optimizer.step()
        scheduler.step()

        # update metrics
        accuracy.update(out["logits"], labels)

    return {
        "train_acc": float(accuracy.compute()),
        "train_loss": train_loss / num_batches,
        "learning_rate": scheduler.get_last_lr()[0],
    }


def val_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    configs: Namespace,
) -> dict:
    """
    model validation loop
    """

    # set model to eval mode
    model.eval()
    # TODO: figure out how to handle this when we move to multi-GPU training (w/ DDP)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracy = MulticlassAccuracy(num_classes=configs.data.num_classes).to(device)

    num_batches = len(dataloader)
    val_loss = 0.0

    tbar_loader = tqdm(
        dataloader,
        desc="val ",
        total=num_batches,
        dynamic_ncols=True,
        disable=configs.no_tqdm,
    )

    with torch.no_grad():
        for images, labels in tbar_loader:
            # move images to GPU if needed
            images, labels = images.to(device), labels.to(device)

            # compute prediction and loss
            out = model(images, labels)
            val_loss += out["loss"].item()

            # update metrics
            accuracy.update(out["logits"], labels)

    return {
        "val_acc": float(accuracy.compute()),
        "val_loss": val_loss / num_batches,
    }
