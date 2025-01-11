import math
from time import perf_counter
import dragon
import multiprocessing as mp
from functools import partial
from .sort_mpi import merge, save_list
from dragon.globalservices.api_setup import connect_to_infrastructure
connect_to_infrastructure()


#def merge(left: list, right: list, num_return_sorted: int) -> list:
#    """This function merges two lists.
#
#    :param left: First list of tuples containing data
#    :type left: list
#    :param right: Second list of tuples containing data
#    :type right: list
#    :return: Merged data
#    :rtype: list
#    """
#    
#    # Merge by 0th element of tuples
#    # i.e. [(9.4, "asdfasd"), (3.5, "oisdjfosa"), ...]
#
#    merged_list = [None] * (len(left) + len(right))
#
#    i = 0
#    j = 0
#    k = 0
#
#    while i < len(left) and j < len(right):
#        if left[i][0] < right[j][0]:
#            merged_list[k] = left[i]
#            i = i + 1
#        else:
#            merged_list[k] = right[j]
#            j = j + 1
#        k = k + 1
#
#    # When we are done with the while loop above
#    # it is either the case that i > midpoint or
#    # that j > end but not both.
#
#    # finish up copying over the 1st list if needed
#    while i < len(left):
#        merged_list[k] = left[i]
#        i = i + 1
#        k = k + 1
#
#    # finish up copying over the 2nd list if needed
#    while j < len(right):
#        merged_list[k] = right[j]
#        j = j + 1
#        k = k + 1
#
#    # only return the last num_return_sorted elements
#    #print(f"Merged list returned {merged_list[-num_return_sorted:]}",flush=True)
#    return merged_list[-num_return_sorted:]


def merge_results_list(results, num_return_sorted, pool):
    num_results = len(results)
    if num_results > 1:
        merged_results = merge(merge_result_list(results[0:num_results//2]),
                               merge_result_list(results[num_results//2:]),
                               num_return_sorted)
    else:
        return results[0]
    return merged_results[0]



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
    top_candidates = results[0]
    print(f"{top_candidates=}")
    num_top_candidates = len(results)
    with open("sort_controller.log", "a") as f:
        f.write(f"Collected {num_top_candidates=}\n")
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
    
            
#def save_list(candidate_dict, ckey, sort_val):
#    candidate_dict[ckey] = sort_val
#    candidate_dict["sort_iter"] = int(ckey)
#    candidate_dict["max_sort_iter"] = ckey
#    print(f"candidate dictionary on iter {int(ckey)}",flush=True)

