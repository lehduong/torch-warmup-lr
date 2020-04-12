import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.optim import SGD
import numpy as np 
import sys
sys.path.append('..')
from torch_warmup_lr import WarmupLR


if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(1, 1, requires_grad=True))]
    optim = SGD(model, 0.1)

    # Choose different scheduler to test
    scheduler = StepLR(optim, step_size=10, gamma=0.1)
    scheduler = MultiStepLR(optim, milestones=[3,6,9], gamma=0.1)
    scheduler = ReduceLROnPlateau(optim, threshold=0.99, mode='min', patience=2, cooldown=5)
    scheduler = WarmupLR(scheduler, 0.01, 3, 'cos')

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()

    for epoch in range(1, 20):
        # reduce learning rate 
        if isinstance(scheduler._scheduler, ReduceLROnPlateau):
            pseudo_loss = 20-epoch
            scheduler.step(pseudo_loss)
            print('Epoch: {} LR: {:.3f} pseudo loss: {:.2f}'.format(epoch, optim.param_groups[0]['lr'], pseudo_loss))
        else:
            scheduler.step()
            print(epoch, optim.param_groups[0]['lr'])
        optim.step()    # backward pass (update network)