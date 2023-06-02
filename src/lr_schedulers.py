"""
Summarized from Bing AI

Here is a more comprehensive list of learning rate schedulers available in PyTorch's `torch.optim.lr_scheduler` package:

- `LambdaLR`: Sets the learning rate of each parameter group to the initial lr times a given function.
- `MultiplicativeLR`: Multiplies the learning rate of each parameter group by the factor given in the specified function.
- `StepLR`: Decays the learning rate of each parameter group by `gamma` every `step_size` epochs.
- `MultiStepLR`: Decays the learning rate of each parameter group by `gamma` once the number of epoch reaches one of the milestones.
- `ExponentialLR`: Decays the learning rate of each parameter group by `gamma` every epoch.
- `CosineAnnealingLR`: Sets the learning rate of each parameter group using a cosine annealing schedule.
- `ReduceLROnPlateau`: Reduces the learning rate when a metric has stopped improving.
- `CyclicLR`: Sets the learning rate according to a cyclical schedule.
- `OneCycleLR`: Sets the learning rate according to the 1cycle policy.
- `CosineAnnealingWarmRestarts`: Sets the learning rate of each parameter group using a cosine annealing schedule with warm restarts.

You can find more information about these and other schedulers in :
https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

- Step decay:
```python
def lr_lambda(step, step_size, gamma):
    return gamma ** (step // step_size)
```

- Exponential decay:
```python
def lr_lambda(step, gamma):
    return gamma ** step
```

- Cosine annealing:
```python
def lr_lambda(step, T_max, eta_min):
    return eta_min + (0.5 * (1 + math.cos(math.pi * (step % T_max) / T_max)))
```

- Cosine annealing with warm restarts:
```python
def lr_lambda(step, T_max, eta_min, T_mult):
    T_max *= T_mult
    return eta_min + (0.5 * (1 + math.cos(math.pi * (step % T_max) / T_max)))
```
"""

import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_scheduler(optimizer, T_max, eta_min=0, warmup_steps=500, T_mult=1, last_epoch=-1):
    """
    Cosine annealing with warm restarts with optional warmup.
    """
    def lr_lambda(step, warmup_steps, T_max, eta_min, T_mult, last_epoch):
        step += last_epoch + 1
        T_max -= warmup_steps
        if step < warmup_steps:
            return step / warmup_steps
        else:
            step -= warmup_steps
            # warm restarts if resulting T_max < step
            T_max *= T_mult
            # TODO: check if eta_min is enforced correctly
            multiplier = eta_min + (0.5 * (1 + math.cos(math.pi * (step % T_max) / T_max)))
            return multiplier

    scheduler = LambdaLR(
        optimizer, lambda step: lr_lambda(step, warmup_steps, T_max, eta_min, T_mult, last_epoch)
    )
    return scheduler


def step_decay_scheduler(optimizer, warmup_steps, step_size, gamma, last_epoch=-1):
    """
    Step decay learning rate scheduler with optional warmup.
    """
    def lr_lambda(step, warmup_steps, step_size, gamma, last_epoch):
        step += last_epoch + 1
        if step < warmup_steps:
            return step / warmup_steps
        else:
            step -= warmup_steps
            return gamma ** (step // step_size)

    scheduler = LambdaLR(
        optimizer, lambda step: lr_lambda(step, warmup_steps, step_size, gamma, last_epoch)
    )
    return scheduler


def exp_decay_scheduler(optimizer, warmup_steps, gamma, last_epoch=-1):
    """
    Exponential decay learning rate scheduler with optional warmup.
    """
    def lr_lambda(step, warmup_steps, gamma, last_epoch):
        step += last_epoch + 1
        if step < warmup_steps:
            return step / warmup_steps
        else:
            step -= warmup_steps
            return  gamma ** step

    scheduler = LambdaLR(
        optimizer, lambda step: lr_lambda(step, warmup_steps, gamma, last_epoch)
    )
    return scheduler


def get_lr_scheduler(
        optimizer, 
        scheduler_name, 
        min_lr, 
        max_steps, 
        warmup_factor,
        step_size=None,
        gamma=None,
        last_epoch=-1, 
        T_mult=1
    ):
    if scheduler_name == 'cosine':
        sch = cosine_scheduler(
            optimizer, 
            T_max=max_steps, 
            eta_min=min_lr, 
            warmup_steps=int(warmup_factor * max_steps),
            T_mult=T_mult, 
            last_epoch=last_epoch
        )
        pass
    elif scheduler_name == 'step':
        assert None not in [step_size, gamma]
        sch = step_decay_scheduler(
            optimizer, 
            warmup_steps=int(warmup_factor * max_steps),
            step_size=step_size, 
            gamma=gamma, 
            last_epoch=last_epoch
        )
        pass
    else:
        raise ValueError(f'{scheduler_name} not in {{step, cosine}}')
    return sch
