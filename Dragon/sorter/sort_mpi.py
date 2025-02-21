import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
import math

import dragon
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

def mpi_sort(_dict, num_keys, num_return_sorted, candidate_dict):
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
        
    print(f"Sort rank {rank} has started",flush=True)

    if rank == 0:
        key_list = list(_dict.keys())
        key_list = [key for key in key_list if "iter" not in key and "model" not in key]
        key_list.sort()
        print(f"Sort rank {rank} retrieved keys",flush=True)
    else:
        key_list = ['' for i in range(num_keys)]

    key_list = comm.bcast(key_list, root=0)
    
    direct_sort_num = max(len(key_list)//size+1,1)
    print(f"Sort rank {rank} sorting {direct_sort_num} keys",flush=True)
    my_key_list = []
    if rank*direct_sort_num < num_keys:
        my_key_list = key_list[rank*direct_sort_num:min((rank+1)*direct_sort_num,num_keys)]
    
    # Direct sort keys assigned to this rank
    my_results = []
    for key in my_key_list:
        try:
            if rank == 0:
                print(f"Getting key {key}",flush=True)
            print(f"Sort rank {rank} getting key {key}",flush=True)
            val = _dict[key]
            print(f"Sort rank {rank} finished getting key {key}",flush=True)
            if rank == 0:
                print(f"Got key {key}",flush=True)
        except Exception as e:
            print(f"Failed to pull {key} from dict", flush=True)
            print(f"Exception {e}",flush=True)
            raise(e)
        if any(val["inf"]):
            this_value = list(zip(val["inf"],val["smiles"],val["model_iter"]))
            this_value.sort(key=lambda tup: tup[0])
            my_results = merge(this_value, my_results, num_return_sorted)
            print(f"rank {rank}: my_results has {len(my_results)} values")
    print(f"Rank {rank} finished direct sort; starting local merge",flush=True)

    # Merge results between ranks
    max_k = math.ceil(math.log2(size))
    max_j = size//2
    try:
        for k in range(max_k):
            offset = 2**k
            for j in range(max_j):
                #if rank ==0: print(f"rank 0 cond val is {k=} {j=} {offset=} {(2**(k+1))*j}")
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

def save_list(candidate_dict, ckey, sort_val):
    candidate_dict[ckey] = sort_val
    candidate_dict["sort_iter"] = int(ckey)
    candidate_dict["max_sort_iter"] = ckey
    print(f"candidate dictionary on iter {int(ckey)}",flush=True)



    
