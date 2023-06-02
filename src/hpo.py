"""
Defines a training function to take a configuration, train, and return results.
"""


def prepare_shakespeare(train_size=0.9, input_path="data/tinyshakespeare/input.txt"):

    from data.data_prep_tinyshakespeare import (
        extract_vocab_and_data, create_text_encoder_decoder, create_data_splits, get_batch
    )
    vocab, text = extract_vocab_and_data(input_path)
    vocab_size = len(vocab)
    encode, decode = create_text_encoder_decoder(vocab)
    data, train_data, valid_data = create_data_splits(text, train_size, encode, decode)

    shakespeare = dict(
        vocab=vocab,
        vocab_size=len(vocab),
        train_data=train_data,
        valid_data=valid_data
    )
    return shakespeare


def exp_setup(setup_args=None):
    if setup_args is None:
        # TODO: load yaml
        pass 
    setup = dict(

    )
    return setup
    

def training_run(setting):
    pass 


if __name__ == "__main__":
    d = prepare_shakespeare()
    pass
