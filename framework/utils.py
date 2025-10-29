"""
utils functions and stuff
"""

import os
import sys

import torch

# global variable to indicate if an interrupt signal has been received
interrupted = False
# current logger reference for global access
_current_logger = None


def get_current_logger():
    """Get the current logger, with fallback to print"""
    return _current_logger if _current_logger else print


def set_current_logger(logger):
    """Set the current logger for global access"""
    global _current_logger
    _current_logger = logger


def log(message):
    """Convenience function to log a message using the current logger"""
    get_current_logger()(message)


# Definition of the signal handler. All it does is flip the 'interrupted' variable
def signal_handler(signum, frame):
    log("received shutdown signal, gracefully closing up shop.")
    global interrupted
    interrupted = True


def shutdown(model, optimizer, scheduler, configs, best_val_acc, curr_epoch):
    """Handles graceful shutdown on interrupt signal"""

    configs.logger("interrupted, saving run state")
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

    sys.exit(99)  # signal to requeue job


def resume(model, optimizer, scheduler, configs):
    """Resumes training from a partial shutdown"""

    # load partially trained model and optimizer/scheduler states
    configs.logger("resuming from partial shutdown...")
    # TODO: double-check these actually update the model, optimizer, scheduler in place
    model.load_state_dict(torch.load(os.path.join(configs.root, "partial.pth")))
    optimizer.load_state_dict(torch.load(os.path.join(configs.root, "partial_opt.pth")))
    scheduler.load_state_dict(
        torch.load(os.path.join(configs.root, "partial_sched.pth"))
    )
    # load best metric and current epoch
    with open(
        os.path.join(configs.root, "partial_stats.txt"), "r", encoding="utf-8"
    ) as file:
        best_val_acc = float(file.readline().strip())
        start_epoch = int(file.readline().strip())
    configs.logger(
        f"resuming from epoch {start_epoch+1}, recovered best val acc: {best_val_acc:.3f}"
    )

    # remove partial files
    os.remove(os.path.join(configs.root, "partial.pth"))
    os.remove(os.path.join(configs.root, "partial_opt.pth"))
    os.remove(os.path.join(configs.root, "partial_sched.pth"))
    os.remove(os.path.join(configs.root, "partial_stats.txt"))

    return best_val_acc, start_epoch
