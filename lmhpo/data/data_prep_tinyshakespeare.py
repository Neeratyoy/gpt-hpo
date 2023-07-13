"""
Data preparation script for Shakespeare
---------------------------------------

Can be downloaded via:
`wget -O [destination] https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

Check with:
`ls -lh [destination]`
"""

import math
import numpy as np
import pickle
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


input_path = "data/tinyshakespeare/input.txt"


def extract_vocab_and_data(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = sorted(list(set(text)))
    return vocab, text


def create_text_encoder_decoder(vocab) -> Tuple[Callable, Callable]:
    vocab = sorted(vocab)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    def encode(s: str, mapping: Dict[str, int]=stoi) -> List[int]:
        return [mapping[c] for c in s]

    # Decoder: take a list of integers, output a string
    def decode(l: List[int], mapping: Dict[int, str]=itos) -> str:
        return ''.join([mapping[i] for i in l])

    return encode, decode


def create_data_splits(
        data, train_size: float=0.9, encode: Callable=None, decode: Callable=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if encode is not None and decode is not None:
        data = torch.tensor(encode(data), dtype=torch.long)
    n = int(train_size * len(data))
    train_data = data[:n]
    valid_data = data[n:]
    return data, train_data, valid_data


def get_batch(
        split: str,
        train_data: torch.Tensor,
        valid_data: torch.Tensor,
        block_size: int = 8,
        batch_size: int = 4,
        device: str = None
):
    """ Gets a randomized batch from the split of data chosen.

    Arguments
    ---------
    split : str, {"train", "valid"}
    train_data: torch.Tensor
        The training data
    valid_data: torch.Tensor
        The validation data
    block_size : int
        The context length for predictions, that is, a sentence length
    batch_size : int
        The batch size, that is, the number of sentences
    device: str, optional
        The device to use for the tensors ("cuda" or "cpu")
        If None, will use "cuda" if GPU available, else "cpu"
    """
    # generate a small batch of data of inputs x and targets y
    assert split in ["train", "valid"]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if split == 'train':
        data = train_data
        # generating random indices as markers in the full text document
        # such that they are a starting point to the sentence of length
        # `block_size` that will be a data point in the batch
        ix = torch.randint(
            low=0, high=len(data) - block_size, size=(batch_size,)
        )
    elif split == 'valid':
        # dropping the last incomplete batch 
        data = valid_data[:(len(valid_data) // block_size) * block_size]
        # generating indices as markers in the full text document
        ix = torch.arange(start=0, end=len(data) - block_size, step=block_size)
    else:
        raise ValueError(f"Invalid split choice: {split}")
    
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix`
    x = torch.stack([data[i:i+block_size] for i in ix])
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix` + 1 (shifted to right)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
   
    x, y = x.to(device), y.to(device)
    return x, y


def prepare_shakespeare(train_size=0.9, input_path="lmhpo/data/tinyshakespeare/input.txt"):
    vocab, text = extract_vocab_and_data(input_path)
    vocab_size = len(vocab)
    encode, decode = create_text_encoder_decoder(vocab)
    data, train_data, valid_data = create_data_splits(text, train_size, encode, decode)

    shakespeare = dict(
        vocab=vocab,
        vocab_size=len(vocab),
        train_data=train_data,
        valid_data=valid_data,
        encode=encode,
        decode=decode,
        get_batch=get_batch,
    )
    return shakespeare


if __name__ == "__main__":
    # Demonstrates how the dataset can be loaded and verified for use.
    # Provides an example use of the batching function for training and validation.

    # Reading dataset as a list of characters
    vocab, text = extract_vocab_and_data(input_path)

    # Sampling the text
    print("Sampling the text...")
    idx = np.random.randint(0, 1000)
    print(text[idx: idx + 500])

    # Checking all the unique characters that occur in this text
    print("Checking all the unique characters that occur in this text...")
    vocab_size = len(vocab)
    vocab_set = "".join(vocab)

    print(f"Vocab set: {vocab_set}")
    print(f"Vocab size: {vocab_size}")

    # Create a mapping from characters to integers
    encode, decode = create_text_encoder_decoder(vocab)

    # example usage
    print(f"encode('Hello World'): {encode('Hello World')}")
    print(f"decode(encode('Hello World')): {decode(encode('Hello World'))}")
    print(f"len(string): {len('Hello World')}")
    print(f"len(encode(string)): {len(encode('Hello World'))}")

    # Encoding all characters in the text to integers and storing it as a tensor
    print("Encoding all characters in the text to integers and storing it as a tensor...")
    print("Creating train and test splits...")

    train_size = 0.9  # 90% train, 10% valid
    data, train_data, valid_data = create_data_splits(
        text, train_size=0.9, encode=encode, decode=decode
    )
    print(f"{data.dtype}, {data.shape}, {len(text)}")
    print("Data:", data)

    # Decoding to verify the earlier sampled text
    print("Decoding to verify the earlier sampled text...")
    print(decode([i.item() for i in data[idx:idx + 500]]))

    # Making the data available globally
    # global data, train_data, valid_data

    print(f"Training data: {train_data.shape}")
    print(f"Validation data: {valid_data.shape}")

    BLOCK_SIZE = 8
    BATCH_SIZE = 4

    # Example sample
    print("Example sample...")
    x, y = get_batch("train", train_data, valid_data, BLOCK_SIZE, BATCH_SIZE, device="cpu")
    print(x.shape, y.shape)
    print("Shape of x and y:", x.shape, y.shape)

    # Demonstrating how the future elements are not part of the input context
    print("Demonstrating how the future elements are not part of the input context...")

    print(f"Examples from batch: \n{'-' * 20}")
    for b in range(2):  # batch
        print(f"Sample {b}: {x[b]}")
        for i in range(BLOCK_SIZE):  # time
            context = x[b, :i + 1]
            target = y[b, i]
            print(f"t{i} - Input: {context}; Target: {target}")
        print()
