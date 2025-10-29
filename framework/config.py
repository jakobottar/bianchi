"""
config parsing and such

opens a yaml config file and provides access to its parameters
"""

import json
import os
import random
import signal

import jsonargparse
import namegenerator
import torch

from .utils import set_current_logger, signal_handler


def _dict_to_namespace(d):
    """Recursively convert dictionary to jsonargparse.Namespace"""
    if isinstance(d, dict):
        # Convert nested dictionaries recursively
        converted = {}
        for key, value in d.items():
            converted[key] = _dict_to_namespace(value)
        return jsonargparse.Namespace(**converted)
    elif isinstance(d, list):
        # Convert list elements recursively
        return [_dict_to_namespace(item) for item in d]
    else:
        # Return primitive values as-is
        return d


def parse_configs() -> jsonargparse.Namespace:
    """Parses command line arguments and config files"""

    # define parser and commands
    parser = jsonargparse.ArgumentParser(default_config_files=["./config.json"])
    # fmt: off
    parser.add_argument("-c", "--config", action="config", help="config file location")
    parser.add_argument("--data.batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--data.name", type=str, default="cifar10", help="dataset name")
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
    parser.add_argument("--seed", type=int, default=-1, help="random seed, -1 for random")
    parser.add_argument("--skip-train", action="store_true", help="skip training")
    parser.add_argument("-S", "--slurm_job_id", type=int, default=-1, help="slurm job id for run tracking")
    # fmt: on

    configs = parser.parse_args()

    # set up seeds and pre-training files
    configs = _set_up_configs(configs)

    return configs


def _set_up_configs(configs: jsonargparse.Namespace) -> jsonargparse.Namespace:
    """Sets up configs after parsing"""

    resume_message = None  # Store resume message to log later

    # see if there's already a run with this slurm job id
    # and load it if so
    if configs.slurm_job_id != -1:
        for name in os.listdir(configs.root):
            if name.startswith(f"{configs.slurm_job_id}_"):
                resume_message = f"found existing run with slurm job id {configs.slurm_job_id}, resuming"
                configs.name = name
                # Store the current root and full path before loading
                base_root = configs.root
                run_path = os.path.join(base_root, name)

                # load previous configs
                with open(
                    os.path.join(base_root, name, "config.json"),
                    "r",
                    encoding="utf-8",
                ) as file:
                    loaded_configs = json.load(file)

                # Convert loaded dict to Namespace for compatibility with jsonargparse
                loaded_namespace = _dict_to_namespace(loaded_configs)
                configs.update(loaded_namespace)

                # Restore the correct root path
                configs.root = run_path

                # Set up logger for resumed run
                logger, close_logger = _create_logger(
                    os.path.join(configs.root, "job.log")
                )
                configs.logger = logger
                configs.close_logger = close_logger
                set_current_logger(logger)

                # Register the signal handler
                signal.signal(signal.SIGTERM, signal_handler)

                # Now log the resume message
                configs.logger(resume_message)

                return configs

    # set name
    if configs.name == "random":
        configs.name = f"{configs.slurm_job_id}_{namegenerator.gen()}"
    else:
        configs.name = f"{configs.slurm_job_id}_{configs.name}"

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

    # Register the signal handler
    signal.signal(signal.SIGTERM, signal_handler)

    # set up logger
    logger, close_logger = _create_logger(os.path.join(configs.root, "job.log"))
    configs.logger = logger
    configs.close_logger = close_logger

    # Set current logger for global access
    set_current_logger(logger)

    return configs


def _create_logger(log_filename, display=True):
    f = open(log_filename, "a", encoding="utf-8")
    counter = [0]

    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text)
        f.write(str(text) + "\n")
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())

    return logger, f.close
