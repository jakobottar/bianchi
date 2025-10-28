"""
Sample main file
"""

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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs.opt.lr, weight_decay=configs.opt.weight_decay
    )

    # initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=configs.epochs * len(dataloaders["train"]),
        eta_min=1e-6 / configs.opt.lr,
    )

    # training loop would go here

    # validation loop would go here

    # save final checkpoint would go here

    print("done!")
