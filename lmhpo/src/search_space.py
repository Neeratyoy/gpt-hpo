from ConfigSpace import Categorical, ConfigurationSpace, Constant, Float, Integer


def get_charLM_space_small(seed: int=1234, defaults: dict=None) -> ConfigurationSpace:
    """Returns a ConfigurationSpace object for the charLM-small benchmark."""
    # HPs that define the training pipeline for the model
    training_hps = {
        "learning_rate": Float(
            "learning_rate", 
            bounds=(1e-6, 1e-2), 
            log=True, 
            default=None if (defaults is None or "learning_rate" not in defaults) else defaults["learning_rate"]
        ),
        "warmup_factor": Integer(
            "warmup_factor", 
            bounds=(0, 5),  # means 0% to 50% of training steps
            default=None if (defaults is None or "warmup_factor" not in defaults) else defaults["warmup_factor"]
        ),
        "dropout": Float(
            "dropout", 
            bounds=(0.01, 0.35),
            default=None if (defaults is None or "dropout" not in defaults) else defaults["dropout"]
        ),
    }
    cs = ConfigurationSpace(seed=seed, space=training_hps)
    return cs


def get_charLM_space_large(seed: int=1234, defaults: dict=None) -> ConfigurationSpace:
    """Returns a ConfigurationSpace object for the charLM-large benchmark."""
    # HPs that instantiate a model architecture primarily
    model_hps = {             
        "embed_size": Integer(
            "embed_size", 
            bounds=(64, 512), 
            default=None if (defaults is None or "embed_size" not in defaults) else defaults["embed_size"], 
            log=True
        ),
        "wide_factor": Integer(
            "wide_factor", 
            bounds=(1, 5), 
            default=None if (defaults is None or "wide_factor" not in defaults) else defaults["wide_factor"]
        ),
        "num_heads": Integer(
            "num_heads", 
            bounds=(4, 16), 
            default=None if (defaults is None or "num_heads" not in defaults) else defaults["num_heads"]
        ),      
        "activation": Categorical(
            "activation", 
            ["gelu", "relu",  "swish"],  # "tanh", "sigmoid", "softplus", "silu"
            default=None if (defaults is None or "activation" not in defaults) else defaults["activation"]
        )
    }         
    # HPs that define the training pipeline for the model
    training_hps = {
        "learning_rate": Float(
            "learning_rate", 
            bounds=(1e-6, 1e-2), 
            log=True, 
            default=None if (defaults is None or "learning_rate" not in defaults) else defaults["learning_rate"]
        ),
        "warmup_factor": Integer(
            "warmup_factor", 
            bounds=(0, 5),  # means 0% to 50% of training steps
            default=None if (defaults is None or "warmup_factor" not in defaults) else defaults["warmup_factor"]
        ),
        "dropout": Float(
            "dropout", 
            bounds=(0.01, 0.35),
            default=None if (defaults is None or "dropout" not in defaults) else defaults["dropout"]
        ),
    }
    cs = ConfigurationSpace(seed=seed, space={**model_hps, **training_hps})
    return cs
