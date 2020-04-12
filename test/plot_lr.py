import torch
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('..')
from torch_warmup_lr import WarmupLR
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR, ReduceLROnPlateau


if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(1, 1, requires_grad=True))]
    optim = SGD(model, 0.1)

    # Choose different scheduler to test
    scheduler = MultiStepLR(optim, milestones=[30,60], gamma=0.1)
    scheduler = WarmupLR(scheduler, init_lr=0.001, num_warmup=20, warmup_strategy='cos')

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()

    # store lr
    lrs = list()
    x = list()
    # The wrapper doesn't affect old scheduler api
    # Simply plug and play
    for epoch in range(1, 90):
        scheduler.step()
        lr = optim.param_groups[0]['lr']
        print(epoch, lr)
        optim.step()    # backward pass (update network)
        lrs.append(lr)
        x.append(epoch)
    data = pd.DataFrame({'lr': lrs, 'epoch': x})
    sns_plot = sns.lineplot(x='epoch', y='lr', data=data)
    sns_plot.set(xlabel='epoch', ylabel='learning rate')
    sns_plot.set_title('learning rate warmup')
    sns_plot.get_figure().savefig('output.png')
    

    