import pathlib
import gzip
from time import perf_counter
from typing import Tuple
import argparse
import os
import sys
import time
import socket
import uuid

sys.path.append("..")
from key_decode import MyKey

import dragon
import multiprocessing as mp
#from dragon.data.distdictionary.dragon_dict import DragonDict
from dragon.data.ddict.ddict import DDict


class WorkerStopException(Exception):
    pass


global data_dict 
data_dict = None

def init_worker(q):
    global data_dict
    data_dict = q.get()
    return


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
    

    
def sort_dictionary(_dict, num_return_sorted: str, max_procs: int, key_list: list):
    """Sort dictionary and return top num_return_sorted smiles strings

    :param _dict: Dragon distributed dictionary
    :type _dict: ...
    """

    direct_sort_num = int(round(len(key_list)/max_procs))

    result_queue = mp.Queue()
    parallel_dictionary_sort(_dict, key_list, direct_sort_num, num_return_sorted, result_queue)
    result = result_queue.get()
    top_candidates = result[-num_return_sorted:]
    
    return top_candidates


if __name__ == "__main__":
    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes the dictionary distributed across')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--total_mem_size', type=int, default=8,
                        help='total managed memory size for dictionary in GB')
    parser.add_argument('--max_procs', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    args = parser.parse_args()

    # Start distributed dictionary
    mp.set_start_method("dragon")
    total_mem_size = args.total_mem_size * (1024*1024*1024)
    dd = DDict(args.managers_per_node, args.num_nodes, total_mem_size)
    print("Launched Dragon Dictionary \n", flush=True)


    
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
