from time import perf_counter
from typing import Tuple
import argparse
import os
import random

import dragon
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict
from dragon.native.process_group import ProcessGroup
from dragon.infrastructure.policy import Policy, GS_DEFAULT_POLICY
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.native.machine import cpu_count, current, System, Node
from .sort_mpi import mpi_sort
import datetime

global data_dict 
data_dict = None

def init_worker(q):
    global data_dict
    data_dict = q.get()
    return

def merge(left: list, right: list, num_return_sorted: int) -> list:
    """This function merges two lists.

    :param left: First list of tuples containing data
    :type left: list
    :param right: Second list of tuples containing data
    :type right: list
    :return: Merged data
    :rtype: list
    """
    
    # Merge by 0th element of tuples
    # i.e. [(9.4, "asdfasd"), (3.5, "oisdjfosa"), ...]

    merged_list = [None] * (len(left) + len(right))

    i = 0
    j = 0
    k = 0

    while i < len(left) and j < len(right):
        if left[i][0] < right[j][0]:
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
    #print(f"Merged list returned {merged_list[-num_return_sorted:]}",flush=True)
    return merged_list[-num_return_sorted:]

def filter_candidate_keys(ckeys: list):
    ckeys = [key for key in ckeys if "iter" not in key and key[0] != "d" and key[0] != "l"]
    return ckeys


def direct_key_sort(_dict,
                    key_list,
                    num_return_sorted,
                    sorted_dict_queue):

    tic = perf_counter()
    my_results = []
    for key in key_list:
        val = _dict[key]
        if any(val["inf"]):
            this_value = list(zip(val["inf"],val["smiles"],val["model_iter"]))
            this_value.sort(key=lambda tup: tup[0])
            my_results = merge(this_value, my_results, num_return_sorted)
        #print(f"my_results {my_results}")
    sorted_dict_queue.put(my_results)
    toc = perf_counter()
    load_time = toc - tic
    print(f"sorted {len(key_list)} keys in {load_time} s", flush=True)

def direct_key_sort_pool(args):

    global data_dict

    key_list, kwargs = args
    num_return_sorted = kwargs['num_return_sorted']
    tic = perf_counter()
    my_results = []
    for key in key_list:
        val = data_dict[key]
        #print(f"{key=}",flush=True)
        #print(f"inf is {val['inf']}")
        if any(val["inf"]):
            this_value = list(zip(val["inf"],val["smiles"],val["model_iter"]))
            this_value.sort(key=lambda tup: tup[0])
            my_results = merge(this_value, my_results, num_return_sorted)
        #print(f"my_results {my_results}")
    toc = perf_counter()
    load_time = toc - tic
    print(f"sorted {len(key_list)} keys in {load_time} s", flush=True)
    return my_results


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
        
        print(f"Merging results with {len(key_list)} keys left", flush=True)
        my_result = my_result_queue.get(timeout=None)
        other_keys_result = other_result_queue.get(timeout=None)
        result = merge(my_result,other_keys_result,num_return_sorted)
    else:
        print(f"Finished Launching sorting procs", flush=True)
        result = my_result_queue.get(timeout=None)

    sorted_dict_queue.put(result)
    
def compare_candidate_results(candidate_dict, continue_event, num_return_sorted, ncompare = 3, max_iter = 100, purge=True):
    
    print(f"Comparing Candidate Lists")
    end_workflow = False
    candidate_keys = candidate_dict.keys()
    sort_iter = 0
    if "sort_iter" in candidate_keys:
        sort_iter = candidate_dict["sort_iter"]
        #candidate_keys.remove('iter')
    candidate_keys = filter_candidate_keys(candidate_keys)
    #print(f"{candidate_keys=}")
    #ncompare = min(ncompare,len(candidate_keys))
    candidate_keys.sort(reverse=True)
    #print(f"{candidate_keys=}")
    num_top_candidates = 0
    if len(candidate_keys) > 0:
        num_top_candidates = len(candidate_dict[candidate_keys[0]]["smiles"])
    

    if sort_iter > max_iter:
        # end if maximum number of iterations reached
        end_workflow = True
        print(f"Ending workflow: sort_iter {sort_iter} exceeded max_iter {max_iter}", flush=True)
    elif len(candidate_keys) > ncompare and num_top_candidates == num_return_sorted:
        # look for unique entries in most recent ncompare lists
        # only do this if there are enough lists to compare and if enough candidates have been identified
        
        
        not_in_common = []
        model_iters = []
        print(f"{candidate_dict=}",flush=True)
        for i in range(ncompare):
            for j in range(ncompare):
                if i < j:
                    ckey_i = candidate_keys[i]
                    ckey_j = candidate_keys[j]
                    not_in_common+=list(set(candidate_dict[ckey_i]["smiles"]) ^ set(candidate_dict[ckey_j]["smiles"]))
                    model_iter_i = list(set(candidate_dict[ckey_i]["model_iter"]))
                    model_iter_j = list(set(candidate_dict[ckey_j]["model_iter"]))
                    model_iters+=model_iter_i
                    model_iters+=model_iter_j
        print(f"Number not in common {len(not_in_common)}")
        # ncompare consecutive lists are identical, end workflow
        print(f"{model_iters=}")

        # End workflow if ncompare lists with unique model_iters are idententical
        if len(not_in_common) == 0 and len(model_iters) == 2*ncompare:
            if len(set(model_iters)) == ncompare:
                print(f"Ending workflow: {ncompare} lists identical", flush=True)
                end_workflow = True

        # If purge, only keep ncompare sorting lists
        if purge:
            if len(candidate_keys) > ncompare:
                del candidate_dict[candidate_keys[-1]]

    print(f"{end_workflow=}")
    if end_workflow:
        continue_event.clear()
        

    
def sort_dictionary_queue(_dict, num_return_sorted: str, max_procs: int, key_list: list, candidate_dict):
    """Sort dictionary and return top num_return_sorted smiles strings

    :param _dict: Dragon distributed dictionary
    :type _dict: ...
    """
    key_list = _dict.keys()
    if "inf_iter" in key_list:
        key_list.remove("inf_iter")
    direct_sort_num = max(int(round(len(key_list)/max_procs))+1,1024)
    print(f"Direct sorting {direct_sort_num} keys per process",flush=True)
    tic = perf_counter()
    result_queue = mp.Queue()
    parallel_dictionary_sort(_dict, key_list, direct_sort_num, num_return_sorted, result_queue)
    result = result_queue.get()
    top_candidates = result[-num_return_sorted:]
    #print(f"{top_candidates=}",flush=True)
    print(f"Found {len(top_candidates)} top candidates")
    
    num_top_candidates = len(top_candidates)
    if num_top_candidates > 0:
        candidate_keys = candidate_dict.keys()
        candidate_keys = filter_candidate_keys(candidate_keys)
        print(f"candidate keys {candidate_keys}")
        ckey = "0"
        if len(candidate_keys) > 0:
            ckey = str(int(max(candidate_keys))+1)
        candidate_inf,candidate_smiles,candidate_model_iter = zip(*top_candidates)
        candidate_dict[ckey] = {"inf": candidate_inf, "smiles": candidate_smiles, "model_iter": candidate_model_iter}
        candidate_dict["iter"] = int(ckey)
        print(f"candidate dictionary on iter {int(ckey)}",flush=True)
        toc = perf_counter()
        print(f"Queue total time {toc - tic} s")


def sort_dictionary_pool(_dict, num_return_sorted: str, max_procs: int, key_list: list, candidate_dict):
    """Sort dictionary and return top num_return_sorted smiles strings

    :param _dict: Dragon distributed dictionary
    :type _dict: ...
    """
    
    key_list = _dict.keys()
    if "inf_iter" in key_list:
        key_list.remove("inf_iter")
    num_keys = len(key_list)
    direct_sort_num = max(num_keys//max_procs+1,1)
    print(f"Direct sorting {direct_sort_num} keys per process on {max_procs} processes",flush=True)

    # Launch Pool
    tic = perf_counter()
    initq = mp.Queue(maxsize=max_procs)
    for _ in range(max_procs):
        initq.put(_dict)
        
    pool = mp.Pool(max_procs, initializer=init_worker, initargs=(initq,))
    print(f"Pool initialized", flush=True)
    kwargs = {"num_return_sorted": num_return_sorted}
    key_lists = [(key_list[st*direct_sort_num:min((st+1)*direct_sort_num,num_keys)], kwargs) 
                for st in range(max_procs) if st*direct_sort_num < num_keys]
    
    print(f"Created {len(key_lists)} key lists from {num_keys} keys")
    
    sort_results = pool.map(direct_key_sort_pool, key_lists)
    sort_results = [sr for sr in sort_results if len(sr) > 0]
    print(f"number of sort_results returned {len(sort_results)}", flush=True)
    pool.close()
    pool.join()
    toc = perf_counter()
    print(f"Pool sort time {toc - tic} s")
    
    if len(sort_results) > 1:
        pool = mp.Pool(max_procs)
        while len(sort_results) > 1:
            merge_pairs = [(sort_results[i],sort_results[i+1],num_return_sorted) 
                        for i in range(0,int(len(sort_results)),2) 
                        if i + 1 < len(sort_results)
                        ]
            print(f"merge_pairs: {len(merge_pairs)} merge pairs for {len(sort_results)} sort results", flush=True)
            merge_results = pool.starmap(merge, merge_pairs)
            print(f"merge pool returned",flush=True)
            if len(sort_results)%2 == 0:
                sort_results = merge_results
            else:
                sort_results = merge_results+[sort_results[-1]]
        print(f"Finished merge", flush=True)
        pool.close()
        print(f"Merge pool closed",flush=True)
        pool.join()
        print(f"Merge pool joined",flush=True)
    
    if len(sort_results) == 0:
        result = []
    else:
        result = sort_results[0]
   
    toc = perf_counter()
    print(f"Pool merge time {toc - tic} s")

    top_candidates = result[-num_return_sorted:]
    #print(f"{top_candidates=}",flush=True)
    print(f"Found {len(top_candidates)} top candidates")
    num_top_candidates = len(top_candidates)
    if num_top_candidates > 0:
        candidate_keys = candidate_dict.keys()
        if "iter" in candidate_keys:
            candidate_keys.remove("iter")
        print(f"candidate keys {candidate_keys}")
        ckey = "0"
        if len(candidate_keys) > 0:
            ckey = str(int(max(candidate_keys))+1)
        candidate_inf,candidate_smiles,candidate_model_iter = zip(*top_candidates)
        candidate_dict[ckey] = {"inf": candidate_inf, "smiles": candidate_smiles, "model_iter": candidate_model_iter}
        candidate_dict["iter"] = int(ckey)
        print(f"candidate dictionary on iter {int(ckey)}",flush=True)
        toc = perf_counter()
        print(f"Pool total time {toc - tic} s",flush=True)

def save_top_candidates_list(candidate_dict):
    ckeys = filter_candidate_keys(candidate_dict.keys())
    if len(ckeys) > 0:
        max_ckey = max(ckeys)
        top_candidates = candidate_dict[max_ckey]
        top_smiles = top_candidates["smiles"]
        lines = [sm+"\n" for sm in top_smiles]

        with open(f"top_candidates.out", 'w') as f:
            f.writelines(lines)


def sort_controller(dd, 
                    num_return_sorted: str, 
                    max_procs: int, 
                    nodelist: list, 
                    candidate_dict, 
                    continue_event, 
                    checkpoint_interval_min=10):

    iter = 0
    with open("sort_controller.log", "w") as f:
        f.write(f"{datetime.datetime.now()}: Starting Sort Controller\n")
        f.write(f"{datetime.datetime.now()}: Sorting for {num_return_sorted} candidates\n")

    candidate_dict["max_sort_iter"] = "-1"
    check_time = perf_counter()
    while continue_event.is_set():
    #if True:
        with open("sort_controller.log", "a") as f:
            f.write(f"{datetime.datetime.now()}: Starting iter {iter}\n")
        tic = perf_counter()
        print(f"Sort iter {iter}",flush=True)
        #sort_dictionary_queue(_dict, num_return_sorted, max_procs, key_list, candidate_dict)
        #sort_dictionary_pool(_dict, num_return_sorted, max_procs, key_list, candidate_dict)
        sort_dictionary_pg(dd, num_return_sorted, max_procs, nodelist, candidate_dict)
        #dd["sort_iter"] = iter
        # max_ckey = candidate_dict["max_sort_iter"]
        # inf_results = candidate_dict[max_ckey]["inf"]
        # cutoff_check = [p for p in inf_results if p < 9 and p > 0]
        # print(f"Cutoff check: {len(cutoff_check)} inf vals below cutoff")
        compare_candidate_results(candidate_dict, continue_event, num_return_sorted, max_iter=50)
        
        
        if (check_time-perf_counter())/60. > checkpoint_interval_min:
            save_top_candidates_list(candidate_dict)
            check_time = perf_counter()
        toc = perf_counter()
        with open("sort_controller.log", "a") as f:
            f.write(f"{datetime.datetime.now()}: iter {iter}: sort time {toc-tic} s\n")
        iter += 1
    ckeys = candidate_dict.keys()
    print(f"final {ckeys=}")
    save_top_candidates_list(candidate_dict)
    # ckeys = filter_candidate_keys(ckeys)
    # if len(ckeys) > 0:
    #     ckey_max = max(ckeys)
    #     print(f"top candidates = {candidate_dict[ckey_max]}")

def sort_dictionary_pg(dd: DDict, num_return_sorted, num_procs: int, nodelist, cdd):
   
    num_procs_pn = num_procs//len(nodelist)
    run_dir = os.getcwd()
    key_list = dd.keys()
    key_list = [key for key in key_list if "iter" not in key and "model" not in key]
    # if "inf_iter" in key_list:
    #     key_list.remove("inf_iter")
    num_keys = len(key_list)
    direct_sort_num = max(len(key_list)//num_procs+1,1)
    print(f"Direct sorting {direct_sort_num} keys per process",flush=True)

    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(restart=False, policy=global_policy, pmi_enabled=True, ignore_error_on_exit=True)

    print(f"Launching sorting process group", flush=True)
    for node in nodelist:
        node_name = Node(node).hostname
        local_policy = Policy(placement=Policy.Placement.HOST_NAME, 
                            host_name=node_name, 
                            cpu_affinity=list(range(num_procs_pn)))
        grp.add_process(nproc=num_procs_pn, 
                            template=ProcessTemplate(target=mpi_sort, 
                                                    args=(dd, num_return_sorted,cdd), 
                                                    policy=local_policy,
                                                    cwd=run_dir))

    grp.init()
    grp.start()
    print(f"Starting Process Group for Sorting")

    
    grp.join()
    grp.stop()

def create_dummy_data(_dict,num_managers):

    NUMKEYS = 100

    for i in range(NUMKEYS):
        key=f"{i%num_managers}_{i}"
        _dict[key] = {"inf":[random.randrange(0,1000,1) for j in range(10)],
                      "smiles": [''.join([random.choice(string.ascii_uppercase + string.digits) for k in range(20)]) for j in range(10)]}


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
