from time import perf_counter
from typing import Tuple
import argparse
import sys
import random

import dragon
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict


def merge(left: list, right: list, num_return_sorted: int) -> list:
    """This function merges two lists.

    :param left: First list containing data
    :type left: list
    :param right: Second list containing data
    :type right: list
    :return: Merged data
    :rtype: list
    """

    merged_list = [None] * (len(left) + len(right))

    i = 0
    j = 0
    k = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged_list[k] = left[i]
            i = i + 1
        else:
            merged_list[k] = right[j]
            j = j + 1
        k = k + 1

    # When we are done with the while loop above
    # it is either the case that i > midpoint or
    # that j > end but not both.

    # finish up copying over the 1st list if needed
    while i < len(left):
        merged_list[k] = left[i]
        i = i + 1
        k = k + 1

    # finish up copying over the 2nd list if needed
    while j < len(right):
        merged_list[k] = right[j]
        j = j + 1
        k = k + 1

    # only return the last num_return_sorted elements
    return merged_list[-num_return_sorted:]
    
def direct_key_sort(_dict,
                    key_list,
                    num_return_sorted,
                    sorted_dict_queue):

    my_results = []
    for key in key_list:
        this_value = _dict[key]
        this_value.sort()
        my_results = merge(this_value, my_results, num_return_sorted)
    sorted_dict_queue.put(my_results)


def parallel_dictionary_sort(_dict, 
                             key_list,
                             direct_sort_num, 
                             num_return_sorted,
                             sorted_dict_queue):
    """ This function sorts the values of a dictionary
    :param _dict: DDict
    :param key_list: list of keys that need sorting
    :param num_to_sort: number of key values to direct sort
    :param num_return_sorted: max lenght of sorted return list
    :param sorted_dict_queue: queue for sorted results
    """

    my_result_queue = mp.Queue()
    my_keys = []
    for i in range(direct_sort_num):
        if len(key_list) > 0:
            this_key = key_list[-1]
            my_keys.append(this_key)
            key_list.pop()

    my_keys_proc = mp.Process(target=direct_key_sort, 
                                     args=(_dict, my_keys, num_return_sorted, my_result_queue))
    my_keys_proc.start()        

    if len(key_list) > 0:
        other_result_queue = mp.Queue()
        other_keys_proc = mp.Process(target=parallel_dictionary_sort, 
                                     args=(_dict, key_list, direct_sort_num, num_return_sorted, other_result_queue))

        other_keys_proc.start()
        other_keys_result = other_result_queue.get(timeout=None)
        my_result = my_result_queue.get(timeout=None)
        result = merge(my_result,other_keys_result,num_return_sorted)
    else:
        result = my_result_queue.get(timeout=None)

    sorted_dict_queue.put(result)
    

    
def sort_dictionary(_dict, num_return_sorted: str, max_procs: int, key_list: list, candidate_queue):
    """Sort dictionary and return top num_return_sorted smiles strings

    :param _dict: Dragon distributed dictionary
    :type _dict: ...
    """

    direct_sort_num = int(round(len(key_list)/max_procs))

    result_queue = mp.Queue()
    parallel_dictionary_sort(_dict, key_list, direct_sort_num, num_return_sorted, result_queue)
    result = result_queue.get()
    top_candidates = result[-num_return_sorted:]
    print(f"{top_candidates=}",flush=True)
    candidate_queue.put(top_candidates)


def create_dummy_data(_dict,num_managers):

    NUMKEYS = 100

    for i in range(NUMKEYS):
        key=f"{i%num_managers}_{i}"
        _dict[key] = [random.randrange(0,1000,1) for j in range(10)]



if __name__ == "__main__":
    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes the dictionary distributed across')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=1,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--dictionary_timeout', type=int, default=10,
                        help='Timeout for Dictionary in seconds')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    args = parser.parse_args()

    # Start distributed dictionary
    mp.set_start_method("dragon")
    total_mem_size = args.mem_per_node * args.num_nodes * (1024*1024*1024)
    dd = DDict(args.managers_per_node, args.num_nodes, total_mem_size)
    print("Launched Dragon Dictionary \n", flush=True)

    loader_proc = mp.Process(target=create_dummy_data, 
                             args=(dd,
                                   args.num_nodes*args.managers_per_node), 
                             )#ignore_exit_on_error=True)
    loader_proc.start()
    print("Process started",flush=True)

    loader_proc.join()
    print("Process ended",flush=True)

    print(f"Number of keys in dictionary is {len(dd.keys())}", flush=True)

    candidate_queue = mp.Queue()
    top_candidate_number = 10
    sorter_proc = mp.Process(target=sort_dictionary, 
                             args=(dd,
                                   top_candidate_number,
                                   args.max_procs_per_node*args.num_nodes,
                                   dd.keys(),
                                    candidate_queue), 
                             )#ignore_exit_on_error=True)
    sorter_proc.start()
    print("Process started",flush=True)

    sorter_proc.join()
    print("Process ended",flush=True)
    top_candidates = candidate_queue.get(timeout=None)
    print(f"{top_candidates=}")

    # # Launch the data loader
    # print("Loading inference data into Dragon Dictionary ...", flush=True)
    # tic = perf_counter()
    # load_inference_data(dd, args.data_path, args.max_procs)
    # toc = perf_counter()
    # load_time = toc - tic
    # print(f"Loaded inference data in {load_time:.3f} seconds \n", flush=True)

    # Close the dictionary
    print("Done, closing the Dragon Dictionary", flush=True)
    dd.destroy()
