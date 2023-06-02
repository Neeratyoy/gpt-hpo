import torch
import torch.nn as nn
from torch.nn import functional as F

from src.attention import Block
from src.lr_schedulers import get_lr_scheduler
from src.utils import (
    train_and_evaluate_model, 
    estimate_loss, 
    plot_losses, 
    count_trainable_params, 
    load_config,
    get_optimizer
)

import wandb


wandb.init(project='char-lm')


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
        # each token directly reads off the logits for the next
        # token from a lookup table
        # Note attention does not have any notion of colocation
        # of characters/words and this is important for lms
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
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
        self.ln_f = nn.LayerNorm(embed_size)  # final layer norm (has bias)
        self.lm_head = nn.Linear(embed_size, vocab_size)

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


def setup_model(vocab_size, fixed_config: dict=None, checkpoint=None) -> tuple([nn.Module, dict]):
    default_setting = load_config("charLM-default")
    if fixed_config is not None:
        default_setting.update(fixed_config)

    model = CharLM(vocab_size=vocab_size, **default_setting)
    return model, default_setting


def setup_training(model: nn.Module, setting: dict, checkpoint=None):
    # initialize the optimizer
    optimizer = get_optimizer(
        setting["optimizer_name"], model.parameters(), setting["learning_rate"]
    )
    # setup the LR scheduler
    #TODO: account for different scheduler settings, right now, only Cosine Annealing
    scheduler_args = dict(
        scheduler_name=setting["lr_schedule"],
        min_lr=setting["min_learning_rate"],
        max_steps=setting["max_steps"],
        warmup_factor=setting["warmup_factor"],
        step_size=setting["step_size"],
        gamma=setting["gamma"],
        last_epoch=-1,  # TODO: load from checkpoint 
        T_mult=setting["T_mult"]
    )
    scheduler = get_lr_scheduler(
        optimizer,
        **scheduler_args
    )
    return optimizer, scheduler


if __name__ == "__main__":
    # Preparing data
    from data.data_prep_tinyshakespeare import (
        extract_vocab_and_data, create_text_encoder_decoder, create_data_splits, get_batch
    )

    train_size = 0.9
    input_path = "data/tinyshakespeare/input.txt"
    vocab, text = extract_vocab_and_data(input_path)
    vocab_size = len(vocab)
    encode, decode = create_text_encoder_decoder(vocab)
    data, train_data, valid_data = create_data_splits(text, train_size, encode, decode)

    # Experiment HPs
    VOCAB_SIZE = vocab_size
    NUM_TRAIN_STEPS = 100
    VERBOSTIY_LEN = 5
    EVAL_ITERS = NUM_TRAIN_STEPS // 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    fixed_config = dict(
        max_steps = NUM_TRAIN_STEPS,
        verbosity_len = VERBOSTIY_LEN,
        eval_iters = EVAL_ITERS,
        device = DEVICE,
        # scheduler defaults
        T_mult = 1,
        gamma = None,
        step_size = None
    )

    # Load defaults
    model, setting = setup_model(
        vocab_size=VOCAB_SIZE, fixed_config=fixed_config, checkpoint=None
    )

    # Training setup
    optimizer, scheduler = setup_training(model, setting)

    # Print the number of parameters in the model
    print(count_trainable_params(model)/1e6, 'M parameters')

    # Training model
    print(
        "Number of unique batches: "\
        f"{(train_data.shape[0] / setting['block_size']) // setting['batch_size']}"
    )
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
    plot_losses(losses["train"], setting["verbosity_len"], "temp.png", losses["valid"], losses["lrs"])
