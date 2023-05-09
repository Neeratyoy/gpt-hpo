from matplotlib import pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import tqdm
from typing import List, Tuple, Callable, Dict


def plot_losses(losses, verbosity, filepath, val_losses=None):
    # plt.clf()
    plt.plot(losses, label="train");
    if val_losses is not None:
        plt.plot(val_losses, label="valid");
        plt.legend();
    plt.ylabel("Loss")
    plt.xlabel(f"Num steps (~{verbosity}x)")
    plt.xlim(0, len(losses))
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


# TODO: verify this ChatGPT solution
def measure_throughput(model, input_data, batch_size=1, num_runs=10):
    """Measure the throughput (in samples per second) of a PyTorch model on input data."""
    # Set the model to evaluation mode
    model.eval()

    # Create a dataloader for the input data
    input_loader = torch.utils.data.DataLoader(input_data, batch_size=batch_size)

    # Warm up the GPU by running the model once on a small batch
    with torch.no_grad():
        input = next(iter(input_loader))
        model(input.cuda())

    # Measure the time required to run the model on the input data
    start_time = time.time()
    for i in range(num_runs):
        for input in input_loader:
            input = input.cuda()
            with torch.no_grad():
                model(input)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate the throughput in samples per second
    num_samples = len(input_data)
    throughput = num_samples * num_runs / elapsed_time

    return throughput


# TODO: verify this Bing solution
"""
import torch
import torch.nn as nn
from torchprofile import profile_macs

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 10)
)

flops = profile_macs(model, (1, 3, 32, 32))
print(f"The model has approximately {flops / 1e6:.2f} million FLOPs.")
"""


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters: int, get_batch: Callable) -> Dict[str, float]:
    """ Function to evaluate the model on train & valid splits.
    """
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_and_evaluate_model(
    model: nn.Module,
    block_size: int,
    batch_size: int,
    get_batch: Callable,
    optimizer: torch.optim = None,
    num_train_steps: int = 10000,
    verbosity_len: int = 1000,
    eval_iters: int = 500,
    plot_loss: str = True,
    device: str = "cpu",
    **kwargs
):
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=kwargs["learning_rate"]
        )

    train_losses = [np.inf]
    valid_losses = [np.inf]

    for iter in tqdm(range(num_train_steps)):

        # sample a batch of data
        xb, yb = get_batch('train', block_size, batch_size, device)

        # evaluate loss on the batch
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        # gradient update
        loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if iter % verbosity_len == 0 or iter == num_train_steps - 1:
            _losses = estimate_loss(model, eval_iters)
            train_losses.append(_losses['train'])
            valid_losses.append(_losses['valid'])
            print()
            print(
                f"step {iter}: train loss {_losses['train']:.4f}, "\
                f"val loss {_losses['valid']:.4f}"
            )

    if plot_loss:
        plot_losses(train_losses, verbosity_len, valid_losses)
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


def decode_and_print_batch(batch):
    for b in range(batch.shape[0]):
        print(f"\nBatch ID: {b}")
        print(decode(batch[b].tolist()))
    print()
