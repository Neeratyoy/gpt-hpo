from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch import optim as opt  # import SGD, Adam, Adafactor
from tqdm import tqdm
from typing import List, Tuple, Callable, Dict
import yaml


class Swish(nn.Module):
    # https://arxiv.org/abs/1710.05941v2
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_acquisition_function(name: str):
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
    

def plot_losses(losses, verbosity, filepath, val_losses=None, lrs=None):
    plt.clf()
    plt.plot(losses, label="train");
    if val_losses is not None:
        plt.plot(val_losses, label="valid");
    plt.ylabel("Loss")
    plt.xlabel(f"Num steps (~{verbosity}x)")
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


def count_trainable_params(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    verbosity_len: int = 1000,
    eval_iters: int = 500,
    plot_loss: str = True,
    device: str = "cpu",
    **kwargs
) -> Dict[str, List[float]]:
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=kwargs["learning_rate"]
        )

    train_losses = [np.inf]
    valid_losses = [np.inf]
    lrs = []

    # preparing logger
    wandb_logger = None if "wandb" not  in kwargs else wandb

    for iter in tqdm(range(max_steps)):

        # sample a batch of data
        split = "train"
        xb, yb = dataloader(split, batch_size)

        # evaluate loss on the batch
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        # gradient update
        loss.backward()
        optimizer.step()

        # scheduler step
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

        train_losses.append(loss.item())
        valid_losses.append(valid_losses[-1])

        # every once in a while evaluate the loss on train and val sets
        if (iter + 1) % verbosity_len == 0 or iter == max_steps - 1:
            model.eval()
            losses = []
            for _ in range(eval_iters):
                split = "valid"
                X, Y = dataloader(split, batch_size)
                losses.append(estimate_loss(model, X, Y))
            valid_losses[-1] = np.mean(losses)
            model.train()
            _train = np.mean(train_losses[-1])
            _valid = np.mean(valid_losses[-1])
            print(
                f"step {iter}: train loss={_train:.4f}, val loss={_valid:.4f}"
            )
        
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
        plot_losses(train_losses, verbosity_len, "temp.png", valid_losses, lrs)
    _losses = {
        "train": train_losses,
        "valid": valid_losses,
        "lrs": lrs,
    }
    return _losses


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
        filename =  os.getcwd() + "/configs/" + filename 
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config



# TODO: verify this ChatGPT solution
# def measure_throughput(model, input_data, batch_size=1, num_runs=10):
#     """Measure the throughput (in samples per second) of a PyTorch model on input data."""
#     # Set the model to evaluation mode
#     model.eval()

#     # Create a dataloader for the input data
#     input_loader = torch.utils.data.DataLoader(input_data, batch_size=batch_size)

#     # Warm up the GPU by running the model once on a small batch
#     with torch.no_grad():
#         input = next(iter(input_loader))
#         model(input.cuda())

#     # Measure the time required to run the model on the input data
#     start_time = time.time()
#     for i in range(num_runs):
#         for input in input_loader:
#             input = input.cuda()
#             with torch.no_grad():
#                 model(input)
#     end_time = time.time()
#     elapsed_time = end_time - start_time

#     # Calculate the throughput in samples per second
#     num_samples = len(input_data)
#     throughput = num_samples * num_runs / elapsed_time

#     return throughput


# TODO: verify this Bing solution
# import torch
# import torch.nn as nn
# from torchprofile import profile_macs

# model = nn.Sequential(
#     nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.Linear(64 * 8 * 8, 10)
# )

# flops = profile_macs(model, (1, 3, 32, 32))
# print(f"The model has approximately {flops / 1e6:.2f} million FLOPs.")
