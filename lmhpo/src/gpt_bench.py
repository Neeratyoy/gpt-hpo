import os
from pathlib import Path
import torch
from typing import Union

from lmhpo.data.tinystories.prepare import (
    download_and_tokenize, get_batch, get_dataloaders
)
from lmhpo.data.data_prep_tinyshakespeare import extract_vocab_and_data, prepare_shakespeare
from lmhpo.data.openwebtext.prepare import (
    load_openwebtext_dataloader, load_tokenized_dataset, GPT2_VOCAB_SIZE
)
from lmhpo.src.search_space import get_charLM_space_large, get_charLM_space_small
from lmhpo.src.utils import load_config
from lmhpo.src.run_skeletal import run


class CharLMBench:
    def __init__(
            self, 
            bench_size: str="small",  # or can be "large"
            bench_name: str="charLM-default",
            data_name: str="tinyshakespeare",
            data_path: str=None,
            train_size: int=0.9,
            set_defaults: bool=False,
            seed: int=None
        ):
        self.bench_name = bench_name
        self.bench_size = bench_size
        self.train_size = train_size
        self.data_name = data_name
        self.data_path = self.check_input_path(data_path)
        self.set_defaults = set_defaults
        self.seed = seed

        # load experiment configs
        training_setting = load_config(f"setup_{bench_name}")
        default_setting = load_config(bench_name)
    
        self.setting = training_setting.copy()
        self.setting.update(default_setting)
        
        # update seed correctly
        if self.seed is None:
            # if benchmark not instantiated with seed, use the default seed defined
            self.seed = self.setting["seed"]
        else:
            # if benchmark instantiated with seed, update the seed in the setting config
            self.setting.update({"seed": self.seed})

        # load benchmark data
        self.dataset = None
        self.vocab_size = None
        self.get_batch = None
        self._prepare_dataset()

        # load default search space
        self.search_space = self.get_search_space(
            seed=self.seed, 
            bench_size=self.bench_size, 
            set_defaults=self.set_defaults
        )

        # setting device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setting.update({"device": self.device})

    def _prepare_dataset(self):
        assert self.data_name in ["tinyshakespeare", "tinystories", "openwebtext"]
        assert self.train_size > 0 and self.train_size <= 1

        if self.data_name == "tinystories":
            # download and tokenize the dataset
            self.dataset, self.vocab_size = download_and_tokenize()
        elif self.data_name == "tinyshakespeare":
            self.dataset = prepare_shakespeare(self.train_size, self.data_path)
            self.vocab_size=self.dataset["vocab_size"]
        elif self.data_name == "openwebtext":
            self.dataset = load_tokenized_dataset(self.data_path)
            self.vocab_size = GPT2_VOCAB_SIZE

    def _prepare_batch_dataloaders(self, setting: dict):
        if self.data_name == "tinystories":
            train_dataloader, val_dataloader = get_dataloaders(
                self.dataset, setting["batch_size"], setting["block_size"]
            )
            def _get_batch(split, batch_size):
                dataloader = train_dataloader if split == "train" else val_dataloader
                return get_batch(dataloader, batch_size, setting["block_size"], setting["device"])
            dataloader = _get_batch
        elif self.data_name == "tinyshakespeare":
            dataloader = lambda split, batch_size: self.dataset["get_batch"](
                split=split, 
                batch_size=batch_size, 
                block_size=setting["block_size"],
                train_data=self.dataset["train_data"], 
                valid_data=self.dataset["valid_data"], 
                device=self.device
            )
        elif self.data_name == "openwebtext":
            dataloader = load_openwebtext_dataloader(
                data_path=self.data_path, 
                batch_size=setting["batch_size"], 
                block_size=setting["block_size"], 
                dataset=self.dataset, 
                device=self.device
            )
        else:
            raise ValueError("Unknown dataset, not in {tinyshakespeare, tinystories}")
        return dataloader

    def check_input_path(self, data_path: Union[Path, str]) -> Path:
        if self.data_name == "tinyshakespeare":
            if data_path is None:
                data_path = Path(__file__).resolve().parent.parent / "data/tinyshakespeare/input.txt"
            else:
                data_path = Path(data_path).resolve()
            if os.path.isfile(data_path):
                pass
            elif os.path.isdir(data_path): 
                if os.path.isfile(data_path / "input.txt"):
                    data_path = os.path.isfile(data_path / "input.txt")
                else:
                    raise ValueError("Passed a directory with no `input.txt` to load!")
            else:
                raise ValueError("Not a file or directory!")
        elif self.data_name == "openwebtext":
            if data_path is None:
                data_path = Path(__file__).resolve().parent.parent / "data/openwebtext"
            else:
                data_path = Path(data_path).resolve()
        data_path = Path(data_path).resolve()
        return data_path

    def get_search_space(
            self, 
            bench_size: str=None, 
            seed: int=None, 
            set_defaults: bool=False
        ) -> dict:
        # retrieve the search space as per the bench_name
        search_space = {
            "small": get_charLM_space_small(seed, self.setting if set_defaults else None),
            # "large": get_charLM_space_large(seed, self.setting if set_defaults else None) 
        }
        bench_size = bench_size if bench_size is not None else self.bench_size
        ss = search_space[bench_size]
        return ss

    def query(
        self, 
        config, 
        load_path: Union[str, Path]=None, 
        save_path: Union[str, Path]=None, 
        setting: dict=None
    ) -> dict:
        # update the setting with the current config, overwriting the defaults
        # this allows an option to update other aspects of the training other than the 
        # config passed (i.e., the search space aside all parameters can be updated)
        # fidelity parameters that are NOT part of the search space can be updated here
        # can thus update multiple parameters including architecture, batch size, etc.
        _setting = self.setting.copy() if setting is None else setting.copy()
        _setting.update(config)
        
        # create dataloader 
        dataloader = self._prepare_batch_dataloaders(_setting)
        # dataloader = lambda split, batch_size: self.shakespeare["get_batch"](
        #     split=split, 
        #     batch_size=batch_size, 
        #     block_size=_setting["block_size"],
        #     train_data=self.shakespeare["train_data"], 
        #     valid_data=self.shakespeare["valid_data"], 
        #     device=self.device
        # )
        _setting.update({"dataloader": dataloader})
        _setting.update({"vocab_size": self.vocab_size})  # shakespeare["vocab_size"]

        # updating possible checkpoint
        if load_path is not None:
            _setting.update({"load_path": load_path})
        if save_path is not None:
            _setting.update({"save_path": save_path})
        # run the evaluation
        result = run(_setting, verbose=True)

        return result


if __name__ == "__main__":
    # test
    bench = CharLMBench(
        bench_name="charLM-default", 
        bench_size="small",
        set_defaults=True,
        seed=42,
        data_name="openwebtext",
    )
    # reduntant passing of arguments to check if the seeding works as this sets the 
    # config space to be 24 while the setting config remains 42
    # search_space = bench.get_search_space(seed=24, bench_size="large", set_defaults=True)
    search_space = bench.get_search_space()
    print(f"Search space:\n{search_space}\n")
    print(f"Default configuration:\n{search_space.get_default_configuration()}\n")
    config = search_space.sample_configuration()
    print(config)
    print()

    # demonstrating how settinig can be overriden during a call to the benchmark
    _setting = None
    _setting = bench.setting.copy()
    _setting.update(dict(
        load_path="./debug",
        save_path="./debug"
    ))

    res = bench.query(
        config, 
        setting=_setting
    )
    print(res.keys())
    print(f"Loss: {res['loss']:.4f}; Cost: {res['cost']:.4f}")
