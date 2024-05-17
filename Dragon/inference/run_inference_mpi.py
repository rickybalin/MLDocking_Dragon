import os
import sys
from collections import OrderedDict
from typing import List
import numpy as np
import psutil

from dragon.native.process import current as current_process

from inference.utils_transformer import ParamsJson, ModelArchitecture, pad
from inference.utils_encoder import SMILES_SPE_Tokenizer


def split_dict_keys(keys: List[str], size: int, proc: int) -> List[str]:
    """Read the keys containing inference data from the Dragon Dictionary 
    and split equally among the procs

    :param keys: list of keys in the dictionary 
    :type keys: List[str] 
    :param size: Number of total procs 
    :type size: int
    :param proc: Local proc ID
    :type proc: int
    :return: list of strings containing the split keys
    :rtype: List[str]
    """
    num_keys = len(keys)
    num_keys_per_proc = num_keys//size
    start_ind = proc*num_keys_per_proc
    end_ind = (proc+1)*num_keys_per_proc
    if proc!=(size-1):
        split_keys = keys[start_ind:end_ind]
    else:
        split_keys = keys[start_ind:-1]
    return split_keys

def process_inference_data(hyper_params: dict, tokenizer, smiles_raw: List[str]):
    """Preprosess the raw SMILES strings to generate the model input data

    :param hyper_params: dictionary with the model hyperparameters
    :type hyper_params: dict
    :param tokenizer: tokenizer to be used for preprocessing
    :type tokenizer: ...
    :param smiles_raw: list of the raw smiles read from file or dict
    :type smiles_raw: list
    :return: model input data
    :rtype: ...
    """
    maxlen = hyper_params['tokenization']['maxlen']
    x_inference = np.array([list(pad(tokenizer(smi)['input_ids'], maxlen, 0)) for smi in smiles_raw])
    return x_inference

def infer(dd, num_procs, proc):
    """Run inference reading from and writing data to the Dragon Dictionary
    """
    # !!! DEBUG !!!
    debug = False
    if debug:
        myp = current_process()
        p = psutil.Process()
        core_list = p.cpu_affinity()
        with open(f"ws_worker_{myp.ident}.log",'a') as f:
            f.write(f"\n\n\n\nNew run\n")
            f.write(f"Hello from process {proc} on core {core_list}\n")
    

    # Read HyperParameters 
    json_file = 'inference/config.json'
    hyper_params = ParamsJson(json_file)

    # Load model and weights
    try:
        model = ModelArchitecture(hyper_params).call()
        model.load_weights(f'inference/smile_regress.autosave.model.h5')
    except Exception as e:
        if debug:
            with open(f"ws_worker_{myp.ident}.log",'a') as f:
                f.write(f"{proc}: Loading model exception {e}\n")

    # Split keys in Dragon Dict
    keys = dd.keys()
    if num_procs>1:
        split_keys = split_dict_keys(keys, num_procs, proc)
    else:
        split_keys = keys

    # Set up tokenizer
    if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
        vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
        spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
        tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

    # Iterate over keys in Dragon Dict
    BATCH = hyper_params['general']['batch_size']
    cutoff = 9
    try:
        for key in split_keys:
            smiles_raw = dd[key]
            x_inference = process_inference_data(hyper_params, tokenizer, smiles_raw)
            output = model.predict(x_inference, batch_size = BATCH)
        
            smiles_ds = np.vstack((smiles_raw, output.flatten())).T
            smiles_ds = sorted(smiles_ds, key=lambda x: x[1], reverse=True)
            filtered_data = list(OrderedDict((item[0], item) for item in smiles_ds if float(item[1]) >= cutoff).values())
            if debug:
                with open(f"ws_worker_{myp.ident}.log",'a') as f:
                    f.write(f"created filtered list\n")
                    f.write(f"{filtered_data}\n")

            if filtered_data:
                dd[f'inf_{key}'] = filtered_data

    except Exception as e:
        if debug:
            with open(f"ws_worker_{myp.ident}.log",'a') as f:
                f.write(f'{e}\n')
        pass

## Run main
if __name__ == "__main__":
    print('Cannot be run as a script at this time', flush=True)


