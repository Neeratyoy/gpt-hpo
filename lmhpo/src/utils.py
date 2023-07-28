from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
from scipy.signal import savgol_filter
import time
import torch
import torch.nn as nn
from torch import optim as opt  # import SGD, Adam, Adafactor
from tqdm import tqdm
from typing import List, Tuple, Callable, Dict
import yaml

from lmhpo.src.lr_schedulers import get_lr_scheduler


class Swish(nn.Module):
    # https://arxiv.org/abs/1710.05941v2
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation_function(name: str):
    acquisition_map = dict(
        relu=nn.ReLU,
        gelu=nn.GELU,
        sigmoid=nn.Sigmoid,
        tanh=nn.Tanh,
        softplus=nn.Softplus,
        swish=Swish,
        silu=nn.SiLU,
    )
    assert name in acquisition_map.keys(), f"{name} not in {list(acquisition_map.keys())}"
    return acquisition_map[name]


def get_optimizer(optimizer_name, model_params, lr):
    # could refer to https://github.com/jettify/pytorch-optimizer for more optimizers
    if optimizer_name == 'sgd':
        return opt.SGD(model_params, lr=lr)
    elif optimizer_name == 'adam':
        return opt.Adam(model_params, lr=lr)
    elif optimizer_name == 'adamw':
        return opt.AdamW(model_params, lr=lr)
    else:
        raise ValueError(f'{optimizer_name} not in {{sgd, adam, adamw}}')
    

def plot_losses(losses, filepath, val_losses=None, lrs=None, smooth=True):
    plt.clf()
    plt.plot(losses, label="train");
    if smooth:
        smoothed_losses = savgol_filter(losses, 51, 3)
        plt.plot(smoothed_losses, label="train (smoothed)");
    if val_losses is not None:
        plt.plot(val_losses, label="valid");
    plt.ylabel("Loss")
    plt.xlabel(f"Num steps")
    plt.xlim(0, len(losses))
    plt.legend(loc="upper right");
    if lrs is not None:
        ax2 = plt.gca().twinx()
        # plot the second line on the secondary y-axis
        ax2.plot(lrs, label="lr", color="red");
        ax2.set_ylabel('Learning rate')
        # ax2.set_yscale('log')
    ax2.legend(loc="lower left");
    plt.tight_layout()
    plt.savefig(filepath, dpi=300);


def count_trainable_params(model) -> float:
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model) -> float:
    """Get the size of a PyTorch model in GBs."""
    # Get the total number of elements in the model's parameters
    num_elements = sum(p.numel() for p in model.parameters())

    # Get the size of each element in bytes
    element_size = model.parameters().__next__().element_size()

    # Calculate the size of the model in bytes
    model_size = num_elements * element_size
    model_size_gb = model_size / (1024 ** 3)

    return model_size_gb


# TODO: verify this ChatGPT solution
def count_flops(model, input_shape):
    """Count the number of FLOPs required to perform a forward pass through a PyTorch model."""
    # Create a dummy input tensor with the desired shape
    _input = torch.randn(*input_shape)

    # Define a hook function to count FLOPs
    flops = [0]
    def count_flops_hook(module, _input, output):
        if hasattr(module, "weight"):
            flops[0] += torch.prod(torch.tensor(output.shape)) * torch.prod(torch.tensor(module.weight.shape))

    # Register the hook on all convolutions and linear layers in the model
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
            module.register_forward_hook(count_flops_hook)

    # Perform a forward pass through the model to trigger the hook and count FLOPs
    with torch.no_grad():
        model(_input)

    return flops[0].item()


@torch.no_grad()
def estimate_loss(
        model: nn.Module,
        X: torch.Tensor,
        Y: torch.Tensor,
        # eval_iters: int,
) -> Dict[str, float]:
    """ Function to evaluate the model on train & valid splits.
    """
    _, loss = model(X, Y)
    return loss.item()


def train_and_evaluate_model(
    model: nn.Module,
    batch_size: int,
    dataloader: Callable,
    optimizer: torch.optim = None,
    scheduler: torch.optim.lr_scheduler = None,
    max_steps: int = 10000,
    training_steps: int = None,
    verbosity_len: int = 1000,
    plot_loss: str = True,
    info: dict = None,
    **kwargs
) -> Dict[str, List[float]]:
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=kwargs["learning_rate"]
        )

    if info is None:
        train_losses = [np.inf]
        valid_losses = [np.inf]
        lrs = []
    else:
        train_losses = info["train_losses"]
        valid_losses = info["valid_losses"]
        lrs = info["lrs"]

    # preparing logger
    # wandb_logger = None if "wandb" not in kwargs else wandb

    # setting iteration state
    curr_step = kwargs["curr_step"] + 1 if "curr_step" in kwargs else 0
    training_steps = max_steps if training_steps is None else min(training_steps, max_steps)

    training_time = 0
    validation_time = 0

    # training loop
    for iter in tqdm(range(curr_step, training_steps)):

        # sample a batch of data
        split = "train"
        xb, yb = dataloader(split, batch_size)
    
        training_time_start = time.time()

        # evaluate loss on the batch
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        # gradient update
        loss.backward()
        optimizer.step()

        # scheduler step
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

        training_time += time.time() - training_time_start

        train_losses.append(loss.item())
        valid_losses.append(valid_losses[-1])

        # evaluate the loss on train and val sets every `verbosity_len` steps
        if (iter + 1) % verbosity_len == 0 or iter == training_steps - 1:
            model.eval()    
            
            validation_time_start = time.time()
            valid_loss = evaluate_model(model, dataloader)
            validation_time += time.time() - validation_time_start
            
            valid_losses[-1] = valid_loss
            _train = np.mean(train_losses[-1])
            _valid = np.mean(valid_losses[-1])
            print(
                f"step {iter}: train loss={_train:.4f}, val loss={_valid:.4f}"
            )
            if "save_path" in kwargs and kwargs["save_path"] is not None:
                save_checkpoint(
                    path=Path(kwargs["save_path"]),
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    steps=iter,
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    lrs=lrs,
                )
            model.train()

        # logging
        if "wandb_logger" in kwargs and kwargs["wandb_logger"]is not None:
            wb = kwargs["wandb_logger"]
            wb.log({
                "train_loss": train_losses[-1],
                "valid_loss": valid_losses[-1],
                "lr": lrs[-1],
                "step": iter
            })

    if plot_loss:
        plot_losses(
            train_losses, 
            os.path.join(os.path.dirname(kwargs["save_path"]), "summary.png"), 
            valid_losses, 
            lrs, 
            smooth=True
        )
    _losses = {
        "train": train_losses,
        "valid": valid_losses,
        "lrs": lrs,
        "training_time": training_time,
        "validation_time": validation_time
    }
    return _losses


def evaluate_model(
    model: nn.Module,
    dataloader: Callable,
) -> Dict[str, List[float]]:
    """ Evaluates the model on the validation set. 
    """
    model.eval()
    X, Y = dataloader(split="valid", batch_size=None)
    loss = estimate_loss(model, X, Y)
    model.train()

    return loss


def generate_from_model(
    model: nn.Module, batch_num: int, sentence_len: int, start_str: str = None
):
    # sampling a start token and generating a batch of it as context
    if start_str is None:
        start_token = np.random.randint(VOCAB_SIZE)
        print(f"Start token: {decode([start_token])}")
        context = torch.zeros((batch_num, 1), dtype=torch.long, device=DEVICE)
        # setting the first token of the batch to the sampled start token
        context[:, 0] = start_token
    else:
        start_token = encode(start_str)
        print(f"Start token: {decode(start_token)}")
        # generating batch of sentences with the start token
        context = torch.tensor(start_token, dtype=torch.long, device=DEVICE)
        context = context.repeat(batch_num, 1)
    # will generate the next sentence_len characters for each of the start token
    out = model.generate(
        context, max_new_tokens=sentence_len, block_size=BLOCK_SIZE
    )
    print(out.shape)
    return out


def decode_and_print_batch(batch: torch.Tensor) -> None:
    for b in range(batch.shape[0]):
        print(f"\nBatch ID: {b}")
        print(decode(batch[b].tolist()))
    print()
    return 1


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    return lrs


def load_config(filename):
    if ".yml" not in filename and ".yaml" not in filename:
        filename = filename + ".yaml"
    if "/" not in filename:
        filename = os.path.dirname(__file__) + "/../configs/" + filename 
    print("loading from", filename)
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """ Function to set all relevant random seed states.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        np.random.seed(seed)
    except NameError:
        # was not imported
        import numpy as np
        np.random.seed(seed)
    try:
        random.seed(seed)
    except NameError:
        # was not imported
        import random
        random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_training(
    model: nn.Module, load_path: str=None, **kwargs
) -> tuple([torch.optim.Optimizer, torch.optim.lr_scheduler, int, dict]):
    # initialize the optimizer
    optimizer = get_optimizer(
        kwargs["optimizer_name"], model.parameters(), kwargs["learning_rate"]
    )
    # setup the LR scheduler
    #TODO: account for different scheduler settings, right now, only Cosine Annealing
    scheduler_args = dict(
        scheduler_name=None if "lr_schedule" not in kwargs else kwargs["lr_schedule"],
        min_lr=None if "min_learning_rate" not in kwargs else kwargs["min_learning_rate"],
        max_steps=None if "max_steps" not in kwargs else kwargs["max_steps"],
        warmup_factor=None if "warmup_factor" not in kwargs else kwargs["warmup_factor"],
        step_size=None if "step_size" not in kwargs else kwargs["step_size"],
        gamma=None if "gamma" not in kwargs else kwargs["gamma"],
        last_epoch=-1,
        T_mult=1 if "T_mult" not in kwargs else kwargs["T_mult"]
    )
    scheduler = get_lr_scheduler(
        optimizer,
        **scheduler_args
    )

    # loading checkpoints if available
    info = None
    current_step = 0
    if load_path is not None and isinstance(load_path, (str, Path)):
        current_step, info = load_checkpoint(load_path, model, optimizer, scheduler)
        print(f"Loaded checkpoint at {current_step}")

    return optimizer, scheduler, current_step, info


def log_weight_statistics(model):
    for i, (name, param) in enumerate(model.named_parameters()):
        print(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
        if i == 5:
            print()
            break


def save_checkpoint(
        path: Path, 
        model: nn.Module, 
        optimizer: torch.optim, 
        lr_scheduler: torch.optim.lr_scheduler, 
        steps: int,
        train_losses: list,
        valid_losses: list,
        lrs: list,
    ) -> None:
    """ Saves the model weights and state of the training pipeline.
    """
    path = Path(path) / "checkpoints.pt" if ".pt" not in str(path) else path
    checkpoint = {
        'steps': steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'lrs': lrs,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
        'random_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
        path: str, 
        model: nn.Module = None, 
        optimizer: torch.optim = None, 
        lr_scheduler: torch.optim.lr_scheduler = None
    ) -> Tuple[int, dict]:
    """ Loads the model weights and state of the training pipeline.
    """
    path = Path(path) / "checkpoints.pt" if ".pt" not in str(path) else path
    if not os.path.isfile(path):
        print(f"Checkpoint file not found at {path}")
        return 0, None
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    steps = checkpoint['steps']
    info = {
        'train_losses': checkpoint["train_losses"],
        'valid_losses': checkpoint["valid_losses"],
        'lrs': checkpoint["lrs"],
    }

    # IMPORTANT: set the random seed states
    torch.set_rng_state(checkpoint['torch_rng_state'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    random.setstate(checkpoint['random_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    return steps, info


def exp_setup(setup_args: str=None):
    if setup_args is None:
        setup_args = load_config("setup_charLM-default")
    if isinstance(setup_args, str):
        setup_args = load_config(setup_args)
    if "device" not in setup_args or setup_args["device"] is None:
        setup_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return setup_args
