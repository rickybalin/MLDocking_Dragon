############# Module Loading ##############
import sys
import pickle
from collections import OrderedDict
from typing import List
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

from utils_transformer import ParamsJson, ModelArchitecture
from utils_encoder import SMILES_SPE_Tokenizer

from dragon.utils import B64

def initialize_mpi():
    """Initialize MPI
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    return comm, size, rank

def split_dict_keys(_dict, size: int, rank: int) -> List[str]:
    """Read the keys containing inference data from the Dragon Dictionary 
    and split equally among the MPI ranks

    :param _dict: Dragon Distributed dictionary
    :type _dict: ...
    :param size: Number of total ranks (MPI comm size)
    :type size: int
    :param rank: Local rank ID
    :type rank: int
    :return: list of strings containing the split keys
    :rtype: List[str]
    """
    keys = _dict.keys()
    num_keys = len(keys)
    num_keys_per_rank = num_keys/size
    start_ind = rank*num_keys_per_rank
    end_ind = (rank+1)*num_keys_per_rank
    split_keys = keys[start_ind:end_ind]
    return split_keys


def main():
    """Run inference with MPI reading from and writing data to the Dragon Dictionary
    """
    # Initialize MPI
    comm, size, rank = initialize_mpi()
    comm.Barrier()
    if rank==0:
        print("initialized MPI", flush=True)

    # Parse Dragon Dict
    args = sys.argv[1:]
    serial_dict = B64.str_to_bytes(args[0])
    if rank==0:
        print(f"serial_dict is {serial_dict}", flush=True)
        print("")
    
    comm.Barrier()
    if rank==0:
        print("Next I will unpickle the dict")
    _dict = pickle.loads(serial_dict)
    comm.Barrier()
    if rank==0:
        print("Unpickled")
    if rank==0:
        print(_dict["H04M000"], flush=True)

    _dict.close()

    """
    # Read HyperParameters 
    json_file = 'config.json'
    hyper_params = ParamsJson(json_file)

    # Load model and weights
    model = ModelArchitecture(hyper_params).call()
    model.load_weights(f'smile_regress.autosave.model.h5')

    # Split keys in Dragon Dict
    split_keys = split_dict_keys(hyper_params, size, rank)

    # Set up tokenizer
    if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
        vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
        spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
        tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

    # Iterate over keys in Dragon Dict
    BATCH = hyper_params['general']['batch_size']
    cutoff = 9
    for key in split_keys:
        print(f"[rank]: reading key {key}", flush=True)
        #Data_smiles_inf, x_inference = large_inference_data_gen(hyper_params, tokenizer, dirs, fil, rank)
        #Output = model.predict(x_inference, batch_size = BATCH)

        '''
        Combine SMILES and predicted docking score.
        Sort the data based on the docking score,
        remove data below cutoff score.
        write data to file in output directory
        '''
        #SMILES_DS = np.vstack((Data_smiles_inf, np.array(Output).flatten())).T
        #SMILES_DS = sorted(SMILES_DS, key=lambda x: x[1], reverse=True)

        #filtered_data = list(OrderedDict((item[0], item) for item in SMILES_DS if item[1] >= cutoff).values())

        #del (Data_smiles_inf)
        #del(Output)
        #del(x_inference)
        #del(SMILES_DS)
        #del(filtered_data)
    """

## Run main
if __name__ == "__main__":
    main()


