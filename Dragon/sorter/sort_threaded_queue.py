import math
from time import perf_counter
import dragon
import multiprocessing as mp
from functools import partial
from .sort_mpi import merge, save_list
from dragon.globalservices.api_setup import connect_to_infrastructure
connect_to_infrastructure()


def parallel_merge_sort(_dict, keys, num_return_sorted, nkey_cutoff,sorted_chunk_queue):

    if len(keys) <= nkey_cutoff:
        result = sort(_dict,keys,num_return_sorted)
        sorted_chunk_queue.put(result)
    else:
        midpoint = len(keys) // 2
        left_keys = keys[:midpoint]
        right_keys = keys[midpoint:]

        result_queue = mp.Queue()
        
        left_proc = mp.Process(target=parallel_merge_sort,
                               args=(_dict, left_keys, num_return_sorted, nkey_cutoff, result_queue))
        right_proc = mp.Process(target=parallel_merge_sort,
                                args=(_dict, right_keys, num_return_sorted, nkey_cutoff, result_queue))
        left_proc.start()
        right_proc.start()
 
        result_a = result_queue.get(timeout=None)  # blocking
        right_b = result_queue.get(timeout=None)
 
        result = merge(result_a, right_b, num_return_sorted)
 
        sorted_chunk_queue.put(result)


def merge_sort(_dict, num_return_sorted, candidate_dict, num_procs):

    print(f"Starting merge sort")
    tic = perf_counter()
    
    keys = _dict.keys()
    keys = [key for key in keys if "iter" not in key and "model" not in key]
    num_keys = len(keys)

    nkey_cutoff = num_keys//num_procs

    result_queue = mp.Queue()
    parallel_merge_sort(_dict, keys, num_return_sorted, nkey_cutoff, result_queue)
    results = result_queue.get()
    
    print(f"Finished merging results in {perf_counter() - tic} seconds",flush=True)
    # put data in candidate_dict                                                                                                 
    top_candidates = results
    num_top_candidates = len(top_candidates)
    with open("sort_controller.log", "a") as f:
        f.write(f"Collected {num_top_candidates=}:\n")
        for tc in top_candidates:
            f.write(f"{tc}\n")
    print(f"Collected {num_top_candidates=}",flush=True)
    if num_top_candidates > 0:
        last_list_key = candidate_dict["max_sort_iter"]
        ckey = str(int(last_list_key) + 1)
        candidate_inf,candidate_smiles,candidate_model_iter = zip(*top_candidates)
        non_zero_infs = len([cinf for cinf in candidate_inf if cinf != 0])
        print(f"Sorted list contains {non_zero_infs} non-zero inference results out of {len(candidate_inf)}")
        sort_val = {"inf": list(candidate_inf), "smiles": list(candidate_smiles), "model_iter": list(candidate_model_iter)}
        save_list(candidate_dict, ckey, sort_val)

    
def sort(_dict, my_key_list, num_return_sorted):
    tic = perf_counter()
    # Direct sort keys assigned to this rank
    my_results = []
    for key in my_key_list:
        try:
            val = _dict[key]
        except Exception as e:
            print(f"Failed to pull {key} from dict", flush=True)
            print(f"Exception {e}",flush=True)
            raise(e)
        if any(val["inf"]):
            this_value = list(zip(val["inf"],val["smiles"],val["model_iter"]))
            this_value.sort(key=lambda tup: tup[0])
            my_results = merge(this_value, my_results, num_return_sorted)
    toc = perf_counter()
    print(f"Sort of {len(my_key_list)} keys in {toc-tic} seconds",flush=True)
    return my_results
    


