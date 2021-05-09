import torch
import math

from torch.optim.lr_scheduler import _LRScheduler
from collections import Counter
import warnings

class PolyLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_iter, power, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch/self.max_iter) ** self.power
                for base_lr in self.base_lrs]


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr, max_lr, milestones, gamma=0.1, last_epoch=-1):
        assert warmup_epochs < milestones[0]
        self.warmup_epochs = warmup_epochs
        self.milestones = Counter(milestones)
        self.gamma = gamma

        initial_lrs = self._format_param("initial_lr", optimizer, initial_lr)
        max_lrs = self._format_param("max_lr", optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(optimizer.param_groups):
                group["initial_lr"] = initial_lrs[idx]
                group["max_lr"] = max_lrs[idx]

        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_epochs:
            pct = self.last_epoch / self.warmup_epochs
            return [
                (group["max_lr"] - group["initial_lr"]) * pct + group["initial_lr"]
                for group in self.optimizer.param_groups]

    @staticmethod
    def _format_param(name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)