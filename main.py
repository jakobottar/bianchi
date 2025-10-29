"""
Sample main file
"""

import os

import torch

import framework as fwk

if __name__ == "__main__":

    # parse configs
    configs = fwk.parse_configs()
    print(f"run name: {configs.name}")

    # build datasets
    datasets = fwk.build_datasets(configs)
    configs.data.num_classes = datasets["num_classes"]

    print("datasets:")
    for split, dataset in datasets.items():
        if split != "num_classes":
            print(f"\t{split}: {dataset}")

    dataloaders = fwk.build_dataloaders(configs, datasets)

    # build model
    model = fwk.build_model(configs)
    print(f"model: {model}")

    # set up optimizer and lr decay
    model.to(device="cuda" if torch.cuda.is_available() else "cpu")

    # initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=configs.opt.lr, weight_decay=configs.opt.weight_decay
    )

    # initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=configs.epochs * len(dataloaders["train"]),
        eta_min=1e-6 / configs.opt.lr,
    )

    # storage for metrics
    best_val_acc = 0.0
    start_epoch = 0

    # recover from shutdown if it happened
    if os.path.exists(os.path.join(configs.root, "partial.pth")):
        best_val_acc, start_epoch = fwk.resume(model, optimizer, scheduler, configs)

    # training loop
    if not configs.skip_train:
        for epoch in range(start_epoch, configs.epochs):
            if not fwk.utils.interrupted:
                print(f"Epoch {epoch + 1}/{configs.epochs}")

                train_metrics = fwk.train_one_epoch(
                    model, dataloaders["train"], optimizer, scheduler, configs
                )
                # acc numbers should always follow the baseball format of .XXX
                print(
                    f"\ttrain loss: {train_metrics['train_loss']:.3f}, "
                    f"train acc: {train_metrics['train_acc']:.3f}, "
                    f"lr: {train_metrics['learning_rate']:.6f}"
                )

            if not fwk.utils.interrupted:  # interrupt might change during train epoch
                val_metrics = fwk.val_one_epoch(model, dataloaders["test"], configs)
                print(
                    f"\tval loss: {val_metrics['val_loss']:.3f}, "
                    f"val acc: {val_metrics['val_acc']:.3f}"
                )

                # save checkpoint last.pth
                torch.save(model.state_dict(), os.path.join(configs.root, "last.pth"))
                if val_metrics["val_acc"] > best_val_acc:
                    best_val_acc = val_metrics["val_acc"]
                    torch.save(
                        model.state_dict(), os.path.join(configs.root, "best.pth")
                    )

            if fwk.utils.interrupted:
                fwk.shutdown(model, optimizer, scheduler, configs, best_val_acc, epoch)

    # final val loop with best model
    model.load_state_dict(torch.load(os.path.join(configs.root, "best.pth")))
    val_metrics = fwk.val_one_epoch(model, dataloaders["test"], configs)

    print("done!")
