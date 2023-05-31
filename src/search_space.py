from ConfigSpace import Categorical, ConfigurationSpace, Constant, Float, Integer


def charLM_space_CS(seed: int=1234) -> ConfigurationSpace: 
    # HPs that instantiate a model architecture primarily
    model_hps = {
        "block_size": Integer("block_size", bounds=(32, 512), default=64),
        "embed_size": Integer("embed_size", bounds=(32, 1024), default=128),
        "num_heads": Integer("num_heads", bounds=(1, 16), default=4),      
        "n_layers": Integer("n_layers", bounds=(1, 50), default=3),                            
        "wide_factor": Integer("wide_factor", bounds=(1, 5), default=2),                   
        "activation": Categorical(
            "activation", 
            ["gelu", "relu", "tanh", "sigmoid", "softplus", "swish", "silu"],
            default="relu"
        ),                
        "prenorm": Categorical(
            "prenorm", 
            [True, False],
            default=True
        ),
        # TODO: add regularizers
    }
    # HPs that define the training pipeline for the model
    training_hps = {
        "batch_size": Integer("batch_size", bounds=(32, 2048), log=True, default=64),
        "learning_rate": Float("learning_rate", bounds=(1e-6, 1), log=True, default=1e-3),
        # TODO: enforce min LR to be 0 if greater than LR
        "min_learning_rate": Float("min_learning_rate", bounds=(1e-20, 1e-2), log=True, default=1e-6),
        "optimizer": Categorical(
            "optimizer", ["sgd", "adam", "adafactor"], default="adam"
        ),
        "lr_schedule": Categorical(
            "lr_schedule", ["step", "cosine", "constant"], default="cosine"
        ),
        "warmup_factor": Float("warmup_steps", bounds=(0, 0.5), default=0.1),
    }

    cs = ConfigurationSpace(seed=seed, space={**model_hps, **training_hps, **exp_hps})
    return cs


def get_experiment_params(vocab_size):
    # these HPs are task or pipeline dependent but affect the model and its training
    # at runtime, these HPs can be fixed with a preset dict before actual training
    exp_hps = {
        "vocab_size": Constant("vocab_size", vocab_size),
        "train_steps": Integer("train_steps", bounds=(1, 1000000)),
        "valid_steps": Integer("valid_steps", bounds=(1, 1000000)),
        "eval_freq": Integer("eval_step_factor", bounds=(0, 1)),
        "device": Categorical("device", ["cpu", "cuda"], default="cpu")
    }
    return ConfigurationSpace(exp_hps)
