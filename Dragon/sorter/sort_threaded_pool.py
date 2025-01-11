from multiprocessing import Pool
import math
from time import perf_counter
import dragon
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

    
def pool_sort(_dict, num_return_sorted, candidate_dict, num_procs):
    tic = perf_counter()
    with Pool(processes=num_procs) as pool:
        print(f"Starting key sort and merge",flush=True)
        # First, every thread sorts and merges a set of keys
        results = [r for r in pool.imap_unordered(partial(sort,
                                                          _dict=_dict,
                                                          size=num_procs,
                                                          num_return_sorted=num_return_sorted),
                                      range(num_procs))
                   if len(r) > 0]
        print(f"Distributed sort done on {num_procs} threads in {perf_counter()-tic} seconds",flush=True)
        
        # Merge results
        tic = perf_counter()
        merged_results = merge_results(results, pool, num_return_sorted)

        
        
        print(f"Finished merging results in {perf_counter() - tic} seconds",flush=True)
    # put data in candidate_dict
    top_candidates = merged_results
    num_top_candidates = len(top_candidates)
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


def merge_results(results, pool, num_return_sorted):

    num_results = len(results)

    if num_results > 1:
        res_left = merge_results(results[0:num_results//2],
                                 pool,
                                 num_return_sorted)
        res_right = merge_results(results[num_results//2:num_results],
                                  pool,
                                  num_return_sorted)

        print("Creating merge process",flush=True)
        merged_result = pool.apply_async(merge, args=[res_left,
	                                              res_right,
                                                      num_return_sorted])
        
        return merged_result.get()
    
    elif num_results == 1:
        return results[0]

    else:
        return []

        
        
def sort(rank, _dict, size, num_return_sorted):
 
    key_list = _dict.keys()
    key_list = [key for key in key_list if "iter" not in key and "model" not in key]
    key_list.sort()

    num_keys = len(key_list)
    direct_sort_num = max(len(key_list)//size+1,1)

    my_key_list = key_list
    if rank*direct_sort_num < num_keys:
        my_key_list = key_list[rank*direct_sort_num:min((rank+1)*direct_sort_num,num_keys)]
        
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
    
    return my_results
    
            
#def save_list(candidate_dict, ckey, sort_val):
#    candidate_dict[ckey] = sort_val
#    candidate_dict["sort_iter"] = int(ckey)
#    candidate_dict["max_sort_iter"] = ckey
#    print(f"candidate dictionary on iter {int(ckey)}",flush=True)

