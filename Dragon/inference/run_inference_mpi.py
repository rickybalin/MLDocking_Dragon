############# Module Loading ##############
import os
import sys
from collections import OrderedDict
from typing import List

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import backend as K
#from tensorflow.keras import layers
#from tensorflow.keras.callbacks import (
#    CSVLogger,
#    EarlyStopping,
#    ModelCheckpoint,
#    ReduceLROnPlateau,
#)
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.preprocessing import sequence, text
#from tensorflow.python.client import device_lib

from inference.utils_transformer import ParamsJson, ModelArchitecture
from inference.utils_encoder import SMILES_SPE_Tokenizer

def split_dict_keys(keys: List[str], size: int, rank: int) -> List[str]:
    """Read the keys containing inference data from the Dragon Dictionary 
    and split equally among the MPI ranks

    :param keys: list of keys in the dictionary 
    :type keys: List[str] 
    :param size: Number of total ranks (MPI comm size)
    :type size: int
    :param rank: Local rank ID
    :type rank: int
    :return: list of strings containing the split keys
    :rtype: List[str]
    """
    num_keys = len(keys)
    num_keys_per_rank = num_keys//size
    start_ind = rank*num_keys_per_rank
    end_ind = (rank+1)*num_keys_per_rank
    if rank!=(size-1):
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
    #x_inference = preprocess_smiles_pair_encoding(smiles_raw, tokenizer, maxlen)
    x_inference = np.array([list(pad(tokenizer(smi)['input_ids'], maxlen, 0)) for smi in smiles_raw])
    return x_inference

def infer(dd):
    """Run inference with MPI reading from and writing data to the Dragon Dictionary
    """
    # Initialize MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
    if rank==0:
        print("Initialized MPI", flush=True)

    if rank==0:
        print(dd["H04M000"], flush=True)

    # Read HyperParameters 
    json_file = 'inference/config.json'
    hyper_params = ParamsJson(json_file)
    comm.Barrier()
    if rank==0:
        print("Loaded hyperparameters", flush=True)

    # Load model and weights
    #model = ModelArchitecture(hyper_params).call()
    #model.load_weights(f'inference/smile_regress.autosave.model.h5')
    comm.Barrier()
    if rank==0:
        print("Loaded model", flush=True)

    # Split keys in Dragon Dict
    keys = dd.keys()
    split_keys = split_dict_keys(keys, size, rank)
    comm.Barrier()
    if rank==0:
        print("Split keys across ranks", flush=True)

    # Set up tokenizer
    if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
        vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
        spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
        print(f'loading file {vocab_file} from {os.getcwd()}',flush=True)
        #tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
    comm.Barrier()
    if rank==0:
        print("Set up tokenizer", flush=True)

    # Iterate over keys in Dragon Dict
    BATCH = hyper_params['general']['batch_size']
    cutoff = 9
    for key in split_keys:
        if rank==0:
            print(f"reading key {key}", flush=True)
        smiles_raw = dd[key]
        #x_inference = process_inference_data(hyper_params, tokenizer, smiles_raw)
        
        #Output = model.predict(x_inference, batch_size = BATCH)
        
        #sort the output so the inference data per key is pre-sorted

        #dd[f'inf_{key}'] = 

        #SMILES_DS = np.vstack((Data_smiles_inf, np.array(Output).flatten())).T
        #SMILES_DS = sorted(SMILES_DS, key=lambda x: x[1], reverse=True)

        #filtered_data = list(OrderedDict((item[0], item) for item in SMILES_DS if item[1] >= cutoff).values())

    MPI.Finalize()

## Run main
if __name__ == "__main__":
    print('Cannot be run as a script at this time', flush=True)


