"""
Sample main file
"""

import framework as fwk

if __name__ == "__main__":

    # parse configs
    configs = fwk.parse_configs()
    print(f"run name: {configs.name}")

    # build datasets
    datasets = fwk.build_datasets(configs)
    configs.num_classes = datasets["num_classes"]

    print("datasets:")
    for split, dataset in datasets.items():
        if split != "num_classes":
            print(f"\t{split}: {dataset}")

    dataloaders = fwk.build_dataloaders(configs, datasets)
