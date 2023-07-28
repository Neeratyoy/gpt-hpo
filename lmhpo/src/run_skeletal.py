"""
Defines a training function to take a configuration, train, and return results.
"""
import os
from pathlib import Path
import time
import torch
# import wandb

from lmhpo.data.data_prep_tinyshakespeare import get_batch, prepare_shakespeare
from lmhpo.src.char_lm import setup_model
from lmhpo.src.utils import (
    count_trainable_params, 
    get_model_size,
    load_config, 
    set_seed, 
    setup_training,
    train_and_evaluate_model, 
    evaluate_model, 
    exp_setup,
    log_weight_statistics
)
    

def run(setting, verbose: str=True):
    """ The DL pipeline executor.

    Arguments
    ---------
    setting : dict of dicts
        A hierarchy of dicts that are the arguments for the `setup_model` function.
        It contains a `config` dict for mainly the search space hyperparameters, the 
        subset that changes every run.
        The `fixed_config` dict houses the task related hyperparameters that are less 
        likely to change per run.
        The optional `checkpoint` dict contains necessary information for reloading an 
        existing DL pipeline state saved to disk.
        Note that the `setup_model` modifies and flattens these dicts. Since `config` and
        `fixed_config` could contain the same hyperparameter, the flattening happens with 
        precedence over the `fixed_config` dict values.
    """
    # Setup logger
    # wandb_args = dict(project="lm-hpo")
    # if "log_name" in setting:
    #     wandb_args.update(dict(name=setting["log_name"]))
    # wandb.init(**wandb_args, config=setting["config"].copy())

    # Set the seed
    try:
        set_seed(setting["seed"])
    except KeyError:
        raise Exception("Cannot find `seed` in setting.")

    try:
        # the try-block exists to not make the code block throw an error if the training 
        # pipeline created does not hold in GOU memory and to catch it appropriately to 
        # indicate a failed run

        # Load defaults
        model = setup_model(setting)  # setting is now flattened
        
        if verbose:
            # Print the number of parameters in the model
            print(setting)
            print(count_trainable_params(model)/1e6, 'M parameters')
            print(get_model_size(model), 'GB in memory')

        # Training setup
        optimizer, scheduler, curr_step, info = setup_training(model, **setting)

        # Training model
        start_time = time.time()
        losses = train_and_evaluate_model(
            model=model,
            **setting,
            optimizer=optimizer,
            scheduler=scheduler,
            curr_step=curr_step,
            plot_loss=True,
            info=info,
            # wandb_logger=wandb,
        )
        end_time = time.time()
        runtime = end_time - start_time

        # Preparing result
        result = dict(
            loss=losses["valid"][-1],
            cost=runtime,
            start_time=start_time,
            end_time=end_time,
        )
        result.update(losses)
        result.update(dict(
            process_id=os.getpid(),
            fidelity=setting["training_steps"],
        ))

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print('CUDA out of memory error:', e)
            result = dict(
                loss=float("inf"),  # maximal loss 
                cost=0,       # minimal runtime cost
                info=str(e)
            )
        else:
            # Handle other runtime errors
            raise e

    # Kill logger
    # wandb.finish()

    
    return result


if __name__ == "__main__":

    setting = dict()

    base_path = Path(__file__).resolve().parent
    # preprocessing data
    d = prepare_shakespeare(input_path=base_path / "../data/tinyshakespeare/input.txt")
    setting.update(dict(vocab_size=d["vocab_size"]))

    name = "charLM-test.yaml"
    training_setting = load_config(f"setup_{name}")
    default_setting = load_config(name)

    # flattening setting dict
    setting.update(training_setting)
    setting.update(default_setting)
    setting.update(dict(device="cuda" if torch.cuda.is_available() else "cpu"))

    # adding dataloader as part of experiment setup
    setting.update(dict(
        dataloader=lambda split, batch_size: get_batch(
            split=split, batch_size=batch_size, block_size=setting["block_size"],
            train_data=d["train_data"], valid_data=d["valid_data"], 
            device=setting["device"]
        )
    ))

    # adding log name
    setting.update(dict(log_name=name))

    # checkpoints
    setting.update({"load_path": base_path / "../../debug"})
    setting.update({"save_path": base_path / "../../debug"})
    
    print("Running an evaluation...")
    result = run(setting, verbose=True)

    print("Result:\n", result)
# end of file