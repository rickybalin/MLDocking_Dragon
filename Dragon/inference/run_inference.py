import os
import sys
from typing import List
import numpy as np
import psutil
import os
from time import perf_counter
import random
import gc
import socket
from tqdm import tqdm
from dragon.utils import host_id
from collections import OrderedDict
import csv

#from inference.utils_transformer import ParamsJson, ModelArchitecture, pad
from inference.utils_transformer import pad, ParamsJson, ModelArchitecture, large_scale_split, large_inference_data_gen
from inference.utils_encoder import SMILES_SPE_Tokenizer
#from training.ST_funcs.clr_callback import *
#from training.ST_funcs.smiles_regress_transformer_funcs import *
from data_loader.model_loader import retrieve_model_from_dict

import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

driver_path = os.getenv("DRIVER_PATH")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
    try:
        keys_per_proc = num_keys // size
        remainder = num_keys % size

        next_start_index = 0
        for i in range(proc+1):
            start_index = next_start_index
            end_index = min(start_index + keys_per_proc + (1 if i < remainder else 0),
                            num_keys)
            next_start_index = end_index
        split_keys = keys[start_index:end_index]
    except Exception as e:
        with open("error.out",'a') as f:
            f.write(f"Exception {e}\n")

    #if num_keys / size - num_keys // size > 0:
    #    num_keys_per_proc = num_keys // size + 1
    #else:
    #    num_keys_per_proc = num_keys // size
    #start_ind = proc * num_keys_per_proc
    #end_ind = (proc + 1) * num_keys_per_proc
    #if proc != (size - 1):
    #    split_keys = keys[start_ind:end_ind]
    #else:
    #    split_keys = keys[start_ind:]

    random.shuffle(split_keys)
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
    maxlen = hyper_params["tokenization"]["maxlen"]
    x_inference = np.array(
        [list(pad(tokenizer(smi)["input_ids"], maxlen, 0)) for smi in smiles_raw]
    )
    return x_inference


def check_model_iter(continue_event):
    return True


def infer(file_path,
          split_files, 
          num_procs, 
          proc, 
          limit=None, 
          debug=False):
    """Run inference reading from and writing data to the Dragon Dictionary"""
    tic = perf_counter()
    gc.collect()
    # !!! DEBUG !!!
    if debug:
        p = psutil.Process()
        core_list = p.cpu_affinity()
        log_file_name = f"infer_worker_{proc}.log" 
        print(f"Opening inference worker log {log_file_name}", flush=True)
        with open(log_file_name,'w') as f:
            f.write(f"\n\nNew run\n")
            f.write(f"Hello from process {p} on core {core_list}\n")
            f.flush()
        cuda_device = os.getenv("CUDA_VISIBLE_DEVICES")
        pvc_device = os.getenv("ZE_AFFINITY_MASK")
        device = None
        if cuda_device:
            device = cuda_device
        if pvc_device:
            device = pvc_device
        hostname = socket.gethostname()
        with open(log_file_name,'a') as f:
            f.write(f"Launching infer for worker {proc} from process {p} on core {core_list} on device {hostname}:{device}\n")
    
    #######HyperParamSetting#############
    driver_path = os.getenv("DRIVER_PATH")
    json_file = os.path.join(driver_path, "inference/config.json")
    hyper_params = ParamsJson(json_file)

    ######## Load model #############
    model = ModelArchitecture(hyper_params).call()
    model.load_weights(os.path.join(driver_path,"inference/smile_regress.autosave.model.h5"))

    ####### Oranize data files #########
    if debug: print(f"Inference process {proc}/{num_procs} has {len(split_files)} files",flush=True)
    
    # Set up tokenizer
    # if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = driver_path + "inference/VocabFiles/vocab_spe.txt"
    spe_file = driver_path + "inference/VocabFiles/SPE_ChEMBL.txt"
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file=spe_file)

    total_model_time = 0
    total_io_time = 0
    num_smiles = 0

    # Iterate over files
    BATCH = hyper_params["general"]["batch_size"]
    cutoff = 9
    output_dir = os.path.join(driver_path, "predicted_data")
    if proc == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for fil in split_files:
            
        # read files and procedd data
        tic_read = perf_counter()
        smiles_raw, x_inference, header = large_inference_data_gen(hyper_params, 
                                                           tokenizer, 
                                                           file_path, 
                                                           fil)
        toc_read = perf_counter()
        
        # run model
        tic_fp = perf_counter()
        output = model.predict(x_inference, batch_size=BATCH, verbose=0).flatten()
        toc_fp = perf_counter()

        SMILES_DS = np.vstack((smiles_raw, np.array(output).flatten())).T
        SMILES_DS = sorted(SMILES_DS, key=lambda x: x[1], reverse=True)
        num_smiles += len(SMILES_DS)

        filtered_data = list(OrderedDict((item[0], item) for item in SMILES_DS if float(item[1]) >= cutoff).values())

        tic_write = perf_counter()
        filename = f'{output_dir}/{header}.csv'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['smiles', 'score'])
            writer.writerows(filtered_data)
        toc_write = perf_counter()
        
        
        model_time = toc_fp - tic_fp
        total_model_time += model_time
        io_time = (toc_read-tic_read) + (toc_write-tic_write)
        total_io_time += io_time

        if debug:
            with open(log_file_name, "a") as f:
                f.write(
                    f"Performed inference on file {fil}: {io_time=} {model_time=} {len(SMILES_DS)=}\n"
                )

    toc = perf_counter()

    metrics = {
        "num_smiles": num_smiles,
        "total_time": toc - tic,
        "data_move_time": total_io_time,
    }
    if debug: print(f"worker {proc} is all DONE in {toc - tic} seconds!! :)", flush=True)
    print(f"Performed inference on {len(split_files)} files: total={toc - tic}, IO={total_io_time}, model={total_model_time}",flush=True)
    return metrics


## Run main
if __name__ == "__main__":
    
    import gzip
    import glob
    
    num_procs = 1
    proc = 0
    continue_event = None
    dd = {}


    file_dir = os.getenv("DATA_PATH")
    all_files = glob.glob(file_dir+"*.gz")
    files = all_files[0:1]
    num_files = len(files)
    file_tuples = [(i,fpath,i) for i,fpath in enumerate(files)]


    for file_tuple in file_tuples:
        file_index = file_tuple[0]
        manager_index = file_tuple[2]
        file_path = file_tuple[1]
        
        smiles = []
        f_name = str(file_path).split("/")[-1]
        f_extension = str(file_path).split("/")[-1].split(".")[-1]
        if f_extension=="smi":
            with file_path.open() as f:
                for line in f:
                    smile = line.split("\t")[0]
                    smiles.append(smile)
        elif f_extension=="gz":
            with gzip.open(str(file_path), 'rt') as f:
                for line in f:
                    smile = line.split("\t")[0]
                    smiles.append(smile)

        inf_results = [0.0 for i in range(len(smiles))]
        key = f"{manager_index}_{file_index}"
        f_name_list = f_name.split('.gz')
        logname =  f_name_list[0].split(".")[0]+f_name_list[1]
        dd[key] = {"f_name": f_name, 
                   "smiles": smiles,
                   "inf": inf_results}
    
    infer(dd, num_procs, proc, continue_event, limit=None)
