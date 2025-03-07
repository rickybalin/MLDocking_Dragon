import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
import math
import sys
import os
import psutil
from time import perf_counter
import dragon
from dragon.utils import host_id
from dragon.data.ddict import DDict
from dragon.globalservices.api_setup import connect_to_infrastructure
connect_to_infrastructure()


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

def mpi_sort(_dict: DDict, num_keys: int, num_return_sorted: int, candidate_dict: DDict):
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
        
    print(f"Sort rank {rank} has started",flush=True)
    tic = perf_counter()
    
    current_host = host_id()
    manager_nodes = _dict.manager_nodes
    stats = _dict.stats
    cpu_num = psutil.Process().cpu_num()
    print(f"Sort rank {rank} on host {current_host} {cpu_num=}",flush=True)

    # Get local keys
    try:
        # Get the host of every rank
        global_hosts_by_rank = comm.allgather(current_host)
        # Get unique hosts
        all_hosts = []
        for h in global_hosts_by_rank:
            if h not in all_hosts:
                all_hosts.append(h)
        if rank == 0:
            print(f"Sorting hosts: {all_hosts}")

        # Create communicator for local ranks only
        for h in all_hosts:
            color = 0 if current_host == h else MPI.UNDEFINED
            new_comm = comm.Split(color=color, key=rank)
            if current_host == h:
                my_host_comm = new_comm
        local_rank = my_host_comm.Get_rank()
        print(f"Sort rank {rank} has local rank {local_rank} on host {current_host}")
        local_ranks = [i for i in range(size) if global_hosts_by_rank[i] == current_host]

        # Get keys on local rank and bcast them to other local ranks
        key_list = []
        for i in range(len(manager_nodes)):
            # local rank 0 gets keys from local managers
            if local_rank == 0:
                if manager_nodes[i].h_uid == current_host:
                    local_manager = i
                    dm = _dict.manager(i)
                    key_list.extend(dm.keys())
            else:
            # all other local ranks make an empty key_list of the correct size
                for dms in stats:
                    if dms.manager_id == i:
                        num_keys = dms.num_keys
                        key_list.extend(['' for i in range(num_keys)])
                        
        # Communicate key_list to local ranks
        key_list = my_host_comm.bcast(key_list, root=0)
        #print(f"Sort rank {rank} has {key_list=}")
    except Exception as e:
        print(f"Exception {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise(e)

    # Split local keys by cpu_num (redundant with local rank)
    num_local_ranks = len(local_ranks)
    direct_sort_num = len(key_list)//num_local_ranks
    
    my_key_list = []
    min_index = cpu_num*direct_sort_num
    if cpu_num < num_local_ranks -1:
        max_index = (cpu_num+1)*direct_sort_num
    else:
        max_index = num_keys
    my_key_list = key_list[min_index:max_index]
    toc = perf_counter()
    print(f"Sort rank {rank} got keys in {toc-tic} seconds, sorting {len(my_key_list)} keys",flush=True)

    # Direct sort keys assigned to this rank
    tic = perf_counter()
    my_results = []
    for i,key in enumerate(my_key_list):
        try:
            #print(f"Sort rank {rank} getting key {key} on iter {i}",flush=True)
            val = _dict[key]
            #print(f"Sort rank {rank} finished getting key {key} on iter {i}",flush=True)
        except Exception as e:
            print(f"Failed to pull {key} from dict", flush=True)
            print(f"Exception {e}",flush=True)
            raise(e)
        
        if any(val["inf"]):
            try:
                #print(f"rank {rank}: make tuple list on iter {i}",flush=True)
                this_value = list(zip(val["inf"],val["smiles"],val["model_iter"]))
                my_results.extend(this_value)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno, flush=True)
                print(e, flush=True)
                raise(e)
            #print(f"rank {rank}: local sort on iter {i}",flush=True)
            #this_value.sort(key=lambda tup: tup[0])
            #print(f"rank {rank}: finished local sort on iter {i}",flush=True)
            #my_results = merge(this_value, my_results, num_return_sorted)
            
    my_results.sort(key=lambda tup: tup[0])
    my_results = my_results[-num_return_sorted:]
    toc = perf_counter()
    print(f"Rank {rank} finished direct sort in {toc-tic} seconds; starting local merge",flush=True)

    # Merge results between ranks
    max_k = math.ceil(math.log2(size))
    max_j = size//2
    print(f"{max_k=} {max_j=}",flush=True)
    try:
        for k in range(max_k):
            offset = 2**k
            for j in range(max_j):
                if rank ==0: print(f"rank 0 cond val is {k=} {j=} {offset=} {(2**(k+1))*j}")
                if rank == (2**(k+1))*j:         
                    neighbor_result = comm.recv(source = rank + offset)
                    my_results = merge(my_results,neighbor_result,num_return_sorted)
                    print(f"rank {rank}: my_results has {len(my_results)} values")
                    #print(f"{rank=}: {k=} {offset=} {len(neighbor_result)=}")
                if rank == (2**(k+1))*j + offset:
                    comm.send(my_results,rank - offset)
            max_j = max(max_j//2,1)
    except Exception as e:
        print(f"Merge failed on rank {rank}",flush=True)
        print(f"{e}",flush=True)
        with open("sort_controller.log","a") as f:
            f.write(f"Merge failed on rank {rank}: {e}\n")
    # rank 0 collects the final sorted list
    print(f"Rank {rank} finished local merge",flush=True)
    
    if rank == 0:
        print(f"Collected sorted results on rank 0",flush=True)
        #print(f"{my_results=}")
        # put data in candidate_dict
        top_candidates = my_results
        num_top_candidates = len(my_results)
        with open("sort_controller.log", "a") as f:
            f.write(f"Collected {num_top_candidates=}\n")
        print(f"Collected {num_top_candidates=}",flush=True)
        if num_top_candidates > 0:
            # candidate_keys = candidate_dict.keys()
            # if "iter" in candidate_keys:
            #     candidate_keys.remove("iter")
            # print(f"candidate keys {candidate_keys}")
            # ckey = "0"
            # if len(candidate_keys) > 0:
            #     ckey = str(int(max(candidate_keys))+1)
            last_list_key = candidate_dict["max_sort_iter"]
            ckey = str(int(last_list_key) + 1)
            candidate_inf,candidate_smiles,candidate_model_iter = zip(*top_candidates)
            non_zero_infs = len([cinf for cinf in candidate_inf if cinf != 0])
            print(f"Sorted list contains {non_zero_infs} non-zero inference results out of {len(candidate_inf)}")
            sort_val = {"inf": list(candidate_inf), "smiles": list(candidate_smiles), "model_iter": list(candidate_model_iter)}
            save_list(candidate_dict, ckey, sort_val)
            
    MPI.Finalize()
    return

def save_list(candidate_dict, ckey, sort_val):
    candidate_dict[ckey] = sort_val
    candidate_dict["sort_iter"] = int(ckey)
    candidate_dict["max_sort_iter"] = ckey
    print(f"candidate dictionary on iter {int(ckey)}",flush=True)



    
