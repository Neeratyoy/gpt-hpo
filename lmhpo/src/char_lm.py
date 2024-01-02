from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union

from lmhpo.src.attention import Block
from lmhpo.src.utils import (
    train_and_evaluate_model, 
    estimate_loss, 
    evaluate_model,
    plot_losses, 
    count_trainable_params, 
    load_checkpoint,
    load_config,
    get_optimizer
)


class CharLM(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            n_layers: int,
            block_size: int,
            embed_size: int,
            num_heads: int,
            wide_factor: int = 4,
            activation: str = "relu",  # could also be "gelu"
            dropout: float = 0.0,
            prenormalize: bool = False,
            device: str = None,
            **kwargs
    ):
        super().__init__()
        self.device = device
        if self.device is None:
            self.device = "gpu" if torch.cuda.is_available() else "cpu"
        # each token directly reads off the logits for the next
        # token from a lookup table
        # Note attention does not have any notion of colocation
        # of characters/words and this is important for LMs
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size, device=self.device)
        self.position_embedding_table = nn.Embedding(block_size, embed_size, device=self.device)
        self.blocks = nn.Sequential(
            *[Block(
                block_size=block_size,
                embed_size=embed_size,
                num_heads=num_heads,
                wide_factor=wide_factor,
                activation=activation,
                dropout=dropout,
                prenormalize=prenormalize,
            ) for _ in range(n_layers)]  # stacks the layers of Transformer blocks
        )
        self.blocks = self.blocks.to(self.device)
        self.ln_f = nn.LayerNorm(embed_size, device=self.device)  # final layer norm (has bias)
        self.lm_head = nn.Linear(embed_size, vocab_size, device=self.device)


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
            loss = F.cross_entropy(logits, targets)  # loss function, could be an argument

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


def setup_model(
    setting: dict
    # config: Union[str, dict]=None, fixed_config: dict=None, **kwargs
) -> tuple([nn.Module, dict]):
    """ Instantiates the model as per arguments passed.
    """
    model = CharLM(**setting)
    if "device" not in setting or setting["device"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        setting["device"] = device
    model = model.to(setting["device"])

    return model


if __name__ == "__main__":

    import wandb
    wandb.init(project='char-lm')

    d = prepare_shakespeare()

    name = "charLM-test.yaml"
    fixed_setting = exp_setup(f"setup_{name}")

    # adding dataloader as part of experiment setup
    fixed_setting.update(dict(
        vocab_size=d["vocab_size"], 
        dataloader=lambda split, batch_size: get_batch(
            split=split, batch_size=batch_size, block_size=fixed_setting["block_size"],
            train_data=d["train_data"], valid_data=d["valid_data"], 
        )
    ))

    config = load_config(name)
    fixed_setting["log_name"] = name

    setting = dict()
    setting.update(dict(
        config=config.copy(),  # get_dictionary().copy(),
        fixed_config=fixed_setting,   # important step 
        
    ))
    print("Running an evaluation...")

    # Load defaults
    model, setting = setup_model(
        vocab_size=VOCAB_SIZE, fixed_config=fixed_config, checkpoint=None
    )
    # Print the number of parameters in the model
    print(count_trainable_params(model)/1e6, 'M parameters')

    # Training setup
    optimizer, scheduler, steps, _ = setup_training(model, setting)

    # Training model
    losses = train_and_evaluate_model(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        get_batch=get_batch,
        optimizer=optimizer,
        scheduler=scheduler,
        plot_loss=False,
        wandb_logger=wandb,
        **setting
    )
    wandb.finish()
    plot_losses(
        losses["train"], plot_path, losses["valid"], losses["lrs"], smooth=True
    )
