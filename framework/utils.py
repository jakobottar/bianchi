"""
utils functions and stuff
"""

import os
import sys

import torch

# global variable to indicate if an interrupt signal has been received
interrupted = False


# Definition of the signal handler. All it does is flip the 'interrupted' variable
def signal_handler(signum, frame):
    print("received shutdown signal, gracefully closing up shop.")
    global interrupted
    interrupted = True


def shutdown(model, optimizer, scheduler, configs, best_val_acc, curr_epoch):
    """Handles graceful shutdown on interrupt signal"""

    print("interrupted, saving run state")
    # save partially trained model
    torch.save(model.state_dict(), os.path.join(configs.root, "partial.pth"))
    # save optimizer and scheduler state too
    torch.save(
        optimizer.state_dict(),
        os.path.join(configs.root, "partial_opt.pth"),
    )
    torch.save(
        scheduler.state_dict(),
        os.path.join(configs.root, "partial_sched.pth"),
    )
    # save best metric and current epoch
    with open(
        os.path.join(configs.root, "partial_stats.txt"), "w", encoding="utf-8"
    ) as file:
        file.write(f"{best_val_acc}\n")
        file.write(f"{curr_epoch}\n")

    sys.exit(42)
