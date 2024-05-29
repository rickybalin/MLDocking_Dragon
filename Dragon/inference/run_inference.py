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
from dragon.native.process import current as current_process

from inference.utils_transformer import ParamsJson, ModelArchitecture, pad
from inference.utils_encoder import SMILES_SPE_Tokenizer
from training.ST_funcs.clr_callback import *
from training.ST_funcs.smiles_regress_transformer_funcs import *
import keras

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
    num_keys_per_proc = num_keys//size
    start_ind = proc*num_keys_per_proc
    end_ind = (proc+1)*num_keys_per_proc
    if proc!=(size-1):
        split_keys = keys[start_ind:end_ind]
    else:
        split_keys = keys[start_ind:-1]
    #random.shuffle(split_keys)
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

def infer_switch(dd, num_procs, proc, continue_event, inf_num_limit):
    
    switch_log = "infer_switch.log"
    iter = 0
    if proc == 0:
        with open(switch_log,'w') as f:
            f.write(f"{datetime.datetime.now()}: Starting inference\n")
            f.write(f"{datetime.datetime.now()}: Limiting number of keys per worker to {inf_num_limit}\n")
    last_model_iter = -1
    while continue_event.is_set():
    #if True:
        #gc.collect()
        # Only run inference if there is a new model
        if "model_iter" in dd.keys():
            current_model_iter = dd["model_iter"]
        else:
            current_model_iter = 0
        if current_model_iter > last_model_iter:
            tic = perf_counter()
            if proc == 0:
                with open(switch_log,"a") as f:
                    f.write(f"{datetime.datetime.now()}: Inference on iter {iter} with model iter {current_model_iter}\n")
            metrics = infer(dd, num_procs, proc, limit=inf_num_limit)
            if proc == 0:
                dd["inf_iter"] = iter
            # with open(switch_log,'a') as f:
            #     f.write(f"iter {iter}: proc {proc}: time per smiles {time_per_smiles} s \n")
            last_model_iter = current_model_iter
            toc = perf_counter()
            with open(switch_log, 'a') as f:
                line = f"{datetime.datetime.now()}: proc {proc}: "
                for mkey in metrics.keys():
                    line += f"{mkey}={metrics[mkey]} "
                line += "\n"
                f.write(line)
                
            if proc == 0:
                with open(switch_log,'a') as f:
                    f.write(f"{datetime.datetime.now()}: iter {iter}: run time {toc - tic} s\n")
            iter += 1


def check_model_iter(dd, model_iter):
    test_match = True
    if "model_iter" in dd.keys():
        if model_iter != dd["model_iter"]:
            test_match = False
    return test_match


def infer(dd, num_procs, proc, limit=None):
    """Run inference reading from and writing data to the Dragon Dictionary
    """
    # !!! DEBUG !!!
    debug = True
    if debug:
        myp = current_process()
        p = psutil.Process()
        core_list = p.cpu_affinity()
        with open(f"ws_worker_{myp.ident}.log",'a') as f:
            f.write(f"\n\n\n\nNew run\n")
            f.write(f"Hello from process {proc} on core {core_list}\n")
    
    keys = dd.keys()

    # Read HyperParameters 
    json_file = driver_path+'inference/config.json'
    hyper_params = ParamsJson(json_file)
    if "model" not in keys:
        model_iter = 0
        # Load model and weights
        try:
            model = ModelArchitecture(hyper_params).call()
            model.load_weights(driver_path+f'inference/smile_regress.autosave.model.h5')
        except Exception as e:
            #eprint(e, flush=True)
            with open(f"ws_worker_{myp.ident}.log",'a') as f:
                f.write(f"{e}\n")
    else:
        try:
            with open(f"ws_worker_{myp.ident}.log",'a') as f:
                f.write(f"Loading retrained model\n")
            #model_path = dd["model"]
            #model = keras.saving.load_model(model_path)
            model_iter = dd["model_iter"]
            model = dd["model"]
            if debug:
                with open(f"ws_worker_{myp.ident}.log",'a') as f:
                    f.write(f"Loaded model {model_iter}\n")
        except Exception as e:
            with open(f"ws_worker_{myp.ident}.log",'a') as f:
                f.write(f"{e}\n")

    # Split keys in Dragon Dict
    
    if num_procs>1:
        split_keys = split_dict_keys(keys, num_procs, proc)
    else:
        split_keys = keys
    if debug:
        with open(f"ws_worker_{myp.ident}.log",'a') as f:
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
            if check_model_iter(dd, model_iter):
                key = split_keys[ikey]
                dict_tic = perf_counter()
                val = dd[key]
                dict_toc = perf_counter()
                
                dictionary_time += dict_toc - dict_tic

                for kkey in val.keys():
                    data_moved_size += sys.getsizeof(kkey)
                    data_moved_size += sum([sys.getsizeof(v) for v in val[kkey]])
                
                smiles_raw = val['smiles']
                x_inference = process_inference_data(hyper_params, tokenizer, smiles_raw)
                output = model.predict(x_inference, batch_size = BATCH, verbose=0).flatten()

                sort_index = np.flip(np.argsort(output)).tolist()
                smiles_sorted = [smiles_raw[i] for i in sort_index]
                pred_sorted = [output[sort_index[i]].item() if output[sort_index[i]]>cutoff else 0.0 for i in range(len(sort_index))]

                # cutoff_check = [p for p in pred_sorted if p < cutoff and p > 0.0]
                # if debug:
                #     with open(f"ws_worker_{myp.ident}.log",'a') as f:
                #         f.write(f"Cutoff check: {len(cutoff_check)} below cutoff \n")

                val['smiles'] = smiles_sorted
                val['inf'] = pred_sorted
                val['model_iter'] = [model_iter for i in range(len(smiles_sorted))]
                dict_tic = perf_counter()
                dd[key] = val
                dict_toc = perf_counter()
                dictionary_time += dict_toc -dict_tic
                for kkey in val.keys():
                    data_moved_size += sys.getsizeof(kkey)
                    data_moved_size += sum([sys.getsizeof(v) for v in val[kkey]])
                num_smiles += len(smiles_sorted)
                if debug:
                    with open(f"ws_worker_{myp.ident}.log",'a') as f:
                        f.write(f"Performed inference on key {key}\n")
            else:
                break

    except Exception as e:
        #eprint(e, flush=True)
        with open(f"ws_worker_{myp.ident}.log",'a') as f:
            f.write(f"{e}")
    toc = perf_counter()
    time_per_smiles = (toc-tic)/num_smiles
    data_move_time_per_smiles = dictionary_time/num_smiles
    data_move_size_per_sec = data_moved_size/dictionary_time
    metrics = {"num_smiles": num_smiles, "time_per_smiles": time_per_smiles, "data_move_time_per_smiles":data_move_time_per_smiles, "data_move_size_per_sec":data_move_size_per_sec}
    return metrics
## Run main
if __name__ == "__main__":
    print('Cannot be run as a script at this time', flush=True)


