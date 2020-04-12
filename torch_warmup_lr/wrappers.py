from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import sin, pi
import numpy as np

class WarmupLR(_LRScheduler):
    def __init__(self, scheduler, min_lr=1e-3, num_warmup=1, warmup_strategy='linear'):
        if warmup_strategy not in ['linear', 'sin', 'constant']:
            raise ValueError("Expect warmup_strategy to be one of ['linear', 'sin', 'constant'] but got {}".format(warmup_strategy))
        self._scheduler = scheduler
        self._min_lr = min_lr
        self._num_warmup = num_warmup
        self._step_count = 0
        # Define the strategy to warm up learning rate 
        self._warmup_strategy = warmup_strategy
        if warmup_strategy == 'sin':
            self._warmup_func = self._warmup_sin
        elif warmup_strategy == 'linear':
            self._warmup_func = self._warmup_linear
        else:
            self._warmup_func = self._warmup_const

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def _warmup_sin(self):
        pass 
    
    def _warmup_const(self):
        pass 

    def _warmup_linear(self):
        pass 

    def get_lr(self):
        lrs = []
        step_num = self._step_count
        # warm up learning rate 
        if step_num <= self._num_warmup:
            for group in self._scheduler.optimizer.param_groups:
                computed_lr = self._warmup_func()
        else:
            lrs = self._scheduler.get_lr()
        return lrs

    def step(self, *args):
        if 