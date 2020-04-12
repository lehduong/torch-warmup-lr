# Pytorch Scheduler wrapper for learning rate warmup

A lightweight wrapper around the DeepMind Control Suite that provides the standard OpenAI Gym interface. The wrapper allows to specify the following:
* Reliable random seed initialization that will ensure deterministic behaviour.
* Setting ```from_pixels=True``` converts proprioceptive observations into image-based. In additional, you can choose the image dimensions, by setting ```height``` and ```width```.
* Action space normalization bound each action's coordinate into the ```[-1, 1]``` range.
* Setting ```frame_skip``` argument lets to perform action repeat.


### Instalation
```
pip install git+git://github.com/lehduong/torch-warmup-lr.git
```

### Usage
```python
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD

from torch_warmup_lr import WarmupLR


if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    scheduler = StepLR(optim, step_size=10, gamma=0.1)
    scheduler = WarmupLR(scheduler)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()

    for epoch in range(1, 20):
        scheduler.step()
        print(epoch, optim.param_groups[0]['lr'])
        optim.step()    # backward pass (update network)
```
