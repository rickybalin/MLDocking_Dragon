import os
import sys
from collections import OrderedDict
from typing import List
import numpy as np
import psutil
import os
from time import perf_counter
import random
import datetime
import gc
import socket

from inference.utils_transformer import ParamsJson, ModelArchitecture, pad
from inference.utils_encoder import SMILES_SPE_Tokenizer
from training.ST_funcs.clr_callback import *
from training.ST_funcs.smiles_regress_transformer_funcs import *

import keras
import tensorflow as tf

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
    
    if num_keys/size - num_keys//size > 0:
        num_keys_per_proc = num_keys//size + 1
    else:
        num_keys_per_proc = num_keys//size
    start_ind = proc*num_keys_per_proc
    end_ind = (proc+1)*num_keys_per_proc
    if proc!=(size-1):
        split_keys = keys[start_ind:end_ind]
    else:
        split_keys = keys[start_ind:]
    
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
    maxlen = hyper_params['tokenization']['maxlen']
    x_inference = np.array([list(pad(tokenizer(smi)['input_ids'], maxlen, 0)) for smi in smiles_raw])
    return x_inference

def check_model_iter(dd, model_iter, continue_event):
    test_match = True
    if continue_event is not None:
        if "model_iter" in dd.keys():
            if model_iter != dd["model_iter"]:
                test_match = False
    return test_match


def infer(dd, num_procs, proc, continue_event, limit=None):
    """Run inference reading from and writing data to the Dragon Dictionary
    """
    gc.collect()
    # !!! DEBUG !!!
    debug = True
    if debug:
        #myp = current_process()
        p = psutil.Process()
        core_list = p.cpu_affinity()
        log_file_name = f"infer_worker_{proc}.log"
        print(f"Opening inference worker log {log_file_name}", flush=True)
        with open(log_file_name,'a') as f:
            f.write(f"\n\n\n\nNew run\n")
            f.write(f"Hello from process {p} on core {core_list}\n")
        cuda_device = os.getenv("CUDA_VISIBLE_DEVICES")
        hostname = socket.gethostname()
        print(f"Launching infer for worker {proc} from process {p} on core {core_list} on device {hostname}:{cuda_device}", flush=True)
    try:
        keys = dd.keys()
    except Exception as e:
        print(f"Client raised exception on DDict assignment: {e}", flush=True)
        print(f"could not get keys in inference worker")
        raise(e)

    # If there is no fine-tuned model, load pre-trained model
    if "model" not in keys:
        model_iter = 0

        # Read HyperParameters 
        json_file = driver_path+'inference/config.json'
        hyper_params = ParamsJson(json_file)

        # Load model and weights
        try:
            model = ModelArchitecture(hyper_params).call()
            model.load_weights(driver_path+f'inference/smile_regress.autosave.model.h5')
        except Exception as e:
            #eprint(e, flush=True)
            with open(log_file_name,'a') as f:
                f.write(f"{e}\n")
        with open(log_file_name,'a') as f:
            f.write("Loaded pretrained model\n")
            print("Loaded pretrained model", flush=True)
    # If there is a fine-tuned model, load weights 
    else:
        try:
            with open(log_file_name,'a') as f:
                f.write(f"Loading fine tuned model\n")
            
            model_iter = dd["model_iter"]
            weights_dict = dd["model"]
            hyper_params = dd["model_hyper_params"]
            model = ModelArchitecture(hyper_params).call()
            # Assign the weights back to the model
            for layer_idx, layer in enumerate(model.layers):
                weights = [weights_dict[f'layer_{layer_idx}_weight_{weight_idx}'] 
                            for weight_idx in range(len(layer.get_weights()))]
                layer.set_weights(weights)

            if debug:
                with open(log_file_name,'a') as f:
                    f.write(f"Loaded model {model_iter}\n")
                    print("Loaded model")
        except Exception as e:
            with open(log_file_name,'a') as f:
                f.write(f"{e}\n")

    # Split keys in Dragon Dict
    keys = [k for k in keys if "iter" not in k and "model" not in k]
    keys.sort()
    if num_procs>1:
        split_keys = split_dict_keys(keys, num_procs, proc)
    else:
        split_keys = keys
    if debug:
        with open(log_file_name,'a') as f:
            f.write(f"Running inference on {len(split_keys)} keys\n")


    # Set up tokenizer
    #if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = driver_path+"inference/VocabFiles/vocab_spe.txt"
    spe_file = driver_path+"inference/VocabFiles/SPE_ChEMBL.txt"
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
    tic = perf_counter()
    num_smiles = 0
    dictionary_time = 0
    data_moved_size = 0
    num_run = len(split_keys)
    if limit is not None:
        num_run = limit
    # Iterate over keys in Dragon Dict
    BATCH = hyper_params['general']['batch_size']
    cutoff = 9
    try:
        #for key in split_keys:
        for ikey in range(num_run):
            if debug:
                print(f"worker {proc} on key iter {ikey}", flush=True)
            if check_model_iter(dd, model_iter, continue_event): # this check is to stop inference in async wf when model is retrained
                ktic = perf_counter()
                key = split_keys[ikey]
                dict_tic = perf_counter()
                try:
                    print(f"worker {proc}: getting val from dd",flush=True)
                    val = dd[key]
                    print(f"worker {proc}: finished getting val from dd",flush=True)
                except:
                    print(f"Client raised exception on pulling from DDict: {e}", flush=True)
                dict_toc = perf_counter()
                key_dictionary_time = dict_toc - dict_tic
                if debug:
                    print(f"worker {proc} pulled key {key} in {key_dictionary_time}s", flush=True)

                for kkey in val.keys():
                    key_data_moved_size = sys.getsizeof(kkey)
                    key_data_moved_size += sum([sys.getsizeof(v) for v in val[kkey]])
                
                smiles_raw = val['smiles']
                x_inference = process_inference_data(hyper_params, tokenizer, smiles_raw)
                output = model.predict(x_inference, batch_size = BATCH, verbose=0).flatten()
                if debug:
                    print(f"worker {proc} inference on key {key}", flush=True)

                sort_index = np.flip(np.argsort(output)).tolist()
                smiles_sorted = [smiles_raw[i] for i in sort_index]
                pred_sorted = [output[sort_index[i]].item() if output[sort_index[i]]>cutoff else 0.0 for i in range(len(sort_index))]

                val['smiles'] = smiles_sorted
                val['inf'] = pred_sorted
                val['model_iter'] = [model_iter for i in range(len(smiles_sorted))]

                dict_tic = perf_counter()
                try:
                    dd[key] = val
                except:
                    print(f"Client raised exception on DDict assignment: {e}", flush=True)
                dict_toc = perf_counter()
                key_dictionary_time += dict_toc -dict_tic
                if debug:
                    print(f"worker {proc} put key {key} in {key_dictionary_time}s", flush=True)

                for kkey in val.keys():
                    key_data_moved_size += sys.getsizeof(kkey)
                    key_data_moved_size += sum([sys.getsizeof(v) for v in val[kkey]])
                
                num_smiles += len(smiles_sorted)

                ktoc = perf_counter()
                key_time = ktoc - ktic
                dictionary_time += key_dictionary_time
                data_moved_size += key_data_moved_size

                if debug:
                    with open(log_file_name,'a') as f:
                        f.write(f"Performed inference on key {key} {key_time=} {len(smiles_sorted)=} {key_data_moved_size=} {key_dictionary_time=}\n")
                    print(f"Performed inference on key {key} {key_time=} {len(smiles_sorted)=} {key_data_moved_size=} {key_dictionary_time=}", flush=True)
            else:
                break

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        with open(log_file_name,'a') as f:
            f.write(f"{exc_type=}, {exc_tb.tb_lineno=}\n")
            f.write(f"{e}\n")

    toc = perf_counter()
   
    metrics = {"num_smiles": num_smiles, 
                "total_time": toc-tic, 
                "data_move_time":dictionary_time, 
                "data_move_size":data_moved_size}
    return metrics
## Run main
if __name__ == "__main__":
    print('Cannot be run as a script at this time', flush=True)
    import pathlib
    import gzip
    import glob
    
    num_procs = 1
    proc = 0
    continue_event = None
    dd = {}

    file_dir = os.getenv("DATA_PATH")
    #file_dir = "/gila/Aurora_deployment/dragon/data/tiny/"
    all_files = glob.glob(file_dir+"*.gz")
    files = all_files[0:1]
    #files = [file_dir+"H28P470.smi.gzch.gz"]
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

