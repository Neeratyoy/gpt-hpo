import torch
import torch.nn as nn
from torch.nn import functional as F

from attention import Block
from utils import train_and_evaluate_model, estimate_loss, plot_losses


class CharLM(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            n_layers: int,
            block_size: int,
            n_embed: int,
            num_heads: int,
            wide_factor: int = 4,
            activation: str = "relu",  # could also be "gelu"
            dropout: float = 0.0,
            prenormalize: bool = False,
            device: str = None
    ):
        super().__init__()
        # each token directly reads off the logits for the next
        # token from a lookup table
        # Note attention does not have any notion of colocation
        # of characters/words and this is important for lms
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(
                block_size=block_size,
                n_embed=n_embed,
                num_heads=num_heads,
                wide_factor=wide_factor,
                activation=activation,
                dropout=dropout,
                prenormalize=prenormalize,
            ) for _ in range(n_layers)]  # stacks the layers of Transformer blocks
        )
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm (has bias)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.device = device
        if self.device is None:
            self.device = "gpu" if torch.cuda.is_available() else "cpu"

    def forward(self, idx, targets=None):
        # B: batch_size, T: block_size, C: embedding_size
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        # fixing positional inputs and learning an embedding over it
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        # adding the positional embeddings across the token embedding batch
        x = tok_emb + pos_emb  # (B,T,C)
        # forward pass through the Transformer layers
        x = self.blocks(x)  # (B,T,C)
        # final layernorm
        x = self.ln_f(x)  # (B,T,C)
        # projecting to the vocabulary
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size, verbose: bool = False):
        # B: batch_size, T: block_size, C:
        # idx is (B, T) array of indices in the current context

        model.eval()

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        model.train()
        return idx


if __name__ == "__main__":
    # Model HPs
    BLOCK_SIZE = 16
    EMBED_SIZE = 64
    NUM_HEADS = 3
    ACTIVATION = "relu"
    PRENORM = False
    WIDE_FACTOR = 4
    DROPOUT = 0.0
    LAYERS = 3

    # Training HPs
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3

    # Experiment HPs
    VOCAB_SIZE = vocab_size
    NUM_TRAIN_STEPS = 10000
    VERBOSTIY_LEN = 1000
    EVAL_ITERS = 500
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initializing model

    model = CharLM(
        vocab_size=VOCAB_SIZE,
        n_layers=LAYERS,
        block_size=BLOCK_SIZE,
        n_embed=EMBED_SIZE,
        num_heads=NUM_HEADS,
        wide_factor=WIDE_FACTOR,
        activation=ACTIVATION,
        dropout=DROPOUT,
        prenormalize=PRENORM,
        device=DEVICE,
    )
    model = model.to(DEVICE)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    # Training model

    train_and_evaluate_model(
        model=model,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        optimizer=None,  # internally uses AdamW, pass an optimizer to override
        num_train_steps=NUM_TRAIN_STEPS,
        verbosity_len=VERBOSTIY_LEN,
        eval_iters=EVAL_ITERS,
        plot_loss=True,
        device=DEVICE,
        learning_rate=LEARNING_RATE  # part of kwargs
    );