import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import get_acquisition_function


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, n_embed: int, head_size: int):
        """
        Arguments
        ---------
        block_size: int
            The sentence length allowed
        n_embed: int
            The size (dimension) of the token embeddings
        head_size: int
            The size (dimension) of the output of an attention head
        """
        super().__init__()

        self.block_size = block_size  # equivalent to T
        self.n_embed = n_embed  # equivalent to C
        self.head_size = head_size

        self.key = nn.Linear(self.n_embed, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embed, self.head_size, bias=False)
        self.value = nn.Linear(self.n_embed, self.head_size, bias=False)

        self.register_buffer(
            'tril', torch.tril(torch.ones(self.block_size, self.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape  # B: batch size; T: block size; C: embedding size
        k = self.key(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)
        q = self.query(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        # performing `scaled` attention
        wei *= self.head_size ** -(1 / 2)  # scaling by `1/sqrt(head size)`

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(
        self, block_size: int, n_embed: int, head_size: int, num_heads: int
    ):
        """
        Arguments
        ---------
        block_size: int
            The sentence length allowed
        n_embed: int
            The size (dimension) of the token embeddings
        head_size: int
            The size (dimension) of the output of an attention head
        num_heads: int
            The number of single attention heads that together form
            one multi-headed attention layer
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(block_size, n_embed, head_size) for _ in range(num_heads)]
        )
        # linear FC layer
        self.proj = nn.Linear(head_size * num_heads, n_embed)

    def forward(self, x):
        # simply stack multiple heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # B: batch size; T: block size; C: embedding size; H: head_size * num_heads
        out = self.proj(out)  # (B, T, H) @ (H, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(
            self,
            n_embed: int,
            wide_factor: int = 4,
            activation: str = "relu",
            dropout: float = 0.0
        ):
        super().__init__()
        self.activation = get_acquisition_function(activation)
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(n_embed, wide_factor * n_embed),
            self.activation(),
            nn.Linear(wide_factor * n_embed, n_embed),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(
        self,
        block_size: int,
        n_embed: int,
        num_heads: int,
        wide_factor: int = 4,
        activation: str = "relu",  # could also be "gelu"
        dropout: float = 0.0,
        prenormalize: bool = False
    ):
        super().__init__()
        # setting head_size to be a factor of other dimensions
        head_size = n_embed // num_heads
        # the multi-headed self-attention (msa)
        self.msa = MultiHeadAttention(block_size, n_embed, head_size, num_heads)
        self.ffwd = FeedForward(n_embed, wide_factor, activation, dropout)

        self.prenormalize = prenormalize
        if prenormalize:
            self.pre_ln = nn.LayerNorm(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        if self.prenormalize:
            # normalizes inputs before passing it through the attention block
            x = x + self.msa( self.pre_ln(x) )
        else:
            x = x + self.msa(x)
        # norm after attention
        x = self.ln1(x)
        # feed-forward
        x = x + self.ffwd(x)
        # norm after feed-forward
        x = self.ln2(x)
        return x


if __name__ == "__main__":

    # Hyperparameter to self-attention
    B, T, C = 4, 8, 32  # batch, time, channels (vocab size)
    head_size = 16

    # Sampling a random tensor of the shape of a batch of data
    print("Sampling a random tensor of the shape of a batch of data...")
    x = torch.randn(B, T, C)
    print(x.shape)

    # Example pass through single-head-attention
    print("Example pass through single-head-attention...")
    verbose = True
    model = Head(T, C, 1)  # single-headed
    out = model(x)
    print(out.shape)

    # Hyperparameters for Multi-headed Self-Attention
    batch_size = 4
    block_size = 8
    n_embed = 64
    head_size = 16
    # having it as 4 would help align with current n_embed but for a general representation choosing 3
    num_heads = 3

    args = {
        "block_size": block_size,
        "n_embed": n_embed,
        "head_size": head_size,
        "num_heads": num_heads
    }

    B, T, C = batch_size, block_size, head_size  # batch, time, channels (vocab size)

    # Initializing a multi-headed attention layer
    msa = MultiHeadAttention(**args)

    # Sampling a random tensor of the shape of a batch of data
    print("Sampling a random tensor of the shape of a batch of data...")
    x = torch.randn(batch_size, block_size, n_embed)
    print(x.shape)