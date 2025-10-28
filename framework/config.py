"""
config parsing and such

opens a yaml config file and provides access to its parameters
"""

import json
import os
import random

import jsonargparse
import namegenerator
import torch


def parse_configs() -> jsonargparse.Namespace:
    """Parses command line arguments and config files"""

    # define parser and commands
    parser = jsonargparse.ArgumentParser(default_config_files=["./config.json"])
    # fmt: off
    parser.add_argument("-c", "--config", action="config", help="config file location")
    parser.add_argument("--data.batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-r", "--data.root", type=str, default="./data/", help="dataset filepath")
    parser.add_argument("--data.workers", type=int, default=2, help="dataloader worker threads")
    parser.add_argument("-E", "--epochs", type=int, default=1, help="number of epochs to train for")
    parser.add_argument("--model.arch", type=str, default="resnet18", help="model architecture")
    parser.add_argument("--model.checkpoint", type=str, default=None, help="checkpoint file, omit for no checkpoint")
    parser.add_argument("--name", type=str, default="random", help="run name")
    parser.add_argument("--no-tqdm", action="store_true", help="disable tqdm progress bar")
    parser.add_argument("--opt.lr", type=float, default=1.0, help="learning rate")
    parser.add_argument("--opt.weight-decay", type=float, default=1e-9, help="optimizer weight decay")
    parser.add_argument("--root", type=str, default="runs", help="root of folder to save runs in")
    parser.add_argument("-S", "--seed", type=int, default=-1, help="random seed, -1 for random")
    parser.add_argument("--skip-train", action="store_true", help="skip training")
    # fmt: on

    configs = parser.parse_args()

    # set up seeds and pre-training files
    configs = set_up_configs(configs)

    return configs


def set_up_configs(configs: jsonargparse.Namespace) -> jsonargparse.Namespace:
    """Sets up configs after parsing"""

    # set name
    if configs.name == "random":
        configs.name = namegenerator.gen()
    else:
        configs.name = configs.name

    # set seed
    if configs.seed != -1:
        random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True

    # create run folder
    new_root = os.path.join(configs.root, configs.name)
    os.makedirs(new_root, exist_ok=True)
    configs.root = new_root

    # save configs object as yaml
    configs_out = configs.as_dict()
    if "config" in configs_out:
        del configs_out["config"]

    # convert every Path to str
    for key, value in configs_out.items():
        if isinstance(value, os.PathLike):
            configs_out[key] = str(value)

    with open(os.path.join(configs.root, "config.json"), "w", encoding="utf-8") as file:
        json.dump(
            configs_out,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return configs
