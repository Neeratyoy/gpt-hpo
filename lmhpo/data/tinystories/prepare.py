import os
import datasets
from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple


datasets.config.DEFAULT_HF_CACHE_HOME = os.environ["DEFAULT_HF_CACHE_HOME"]
datasets.config.DEFAULT_HF_DATASETS_CACHE = os.environ["DEFAULT_HF_DATASETS_CACHE"]


def download_and_tokenize(save: bool=True):
    # Load a tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
    # Set the end-of-sequence token as the padding token
    tokenizer.pad_token = tokenizer.eos_token
    # Extract the vocabulary size
    vocab_size = tokenizer.vocab_size

    try:
        tokenized_dataset = load_from_disk(
            os.path.join(datasets.config.DEFAULT_HF_CACHE_HOME, "tinystories-tokenized")
        )
        return tokenized_dataset, vocab_size
    except Exception as e:
        print(e)
        tokenized_dataset = None

    # Load the dataset
    dataset = load_dataset('roneneldan/TinyStories')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True
        )
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    if save:
        # Save the tokenized dataset
        tokenized_dataset.save_to_disk(
            os.path.join(datasets.config.DEFAULT_HF_CACHE_HOME, "tinystories-tokenized")
        )

    return tokenized_dataset, vocab_size


def get_batch(
    dataloader: torch.utils.data.DataLoader, 
    batch_size: int,
    block_size: int,
    device: str="cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        batch = next(dataloader)
    except StopIteration:
        dataloader = iter(dataloader)
        batch = next(dataloader)
    except Exception as e:
        raise e

    # get a batch from the dataloader
    batch = next(dataloader)  # loads data of shape (2048, effective_batch_size)
    # processing
    _x = torch.Tensor([_x.tolist() for _x in batch["input_ids"]])
    _x = _x.to(torch.int64)  # TODO: enforce type across models?

    # flatten it
    _x = _x.transpose(0,1).reshape(-1)
    
    # generate starting indexes for each sentence in the batch
    idxs = torch.randint(low=0, high=len(_x) - block_size, size=(batch_size,))
    # extract the sentences
    X = torch.stack([_x[i:i+block_size] for i in idxs]).to(device)
    y = torch.stack([_x[i+1:i+1+block_size] for i in idxs]).to(device)

    return X, y


def get_dataloaders(
    dataset: datasets.DatasetDict, 
    batch_size: int=64,
    block_size: int=256
):
    train = dataset["train"]
    valid = dataset["validation"]

    # calculate effective batch size
    max_length = 2048  # fixed for GPT2Tokenizer
    effective_batch_size = (batch_size * block_size) // max_length  # total tokens seen per batch remains same

    # Create PyTorch DataLoaders for the training and validation sets
    train_dataloader = iter(DataLoader(train, batch_size=effective_batch_size, shuffle=True))
    val_dataloader = iter(DataLoader(valid, batch_size=effective_batch_size))

    return train_dataloader, val_dataloader


if __name__ == "__main__":

    BATCH_SIZE = 64
    BLOCK_SIZE = 256

    dataset = download_and_tokenize()
    train_dataloader, val_dataloader = get_dataloaders(dataset, BATCH_SIZE)

    X, y = get_batch(train_dataloader, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)

    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")

    print(X[0])
    print(y[0])
