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

def mpi_sort(_dict, num_return_sorted, candidate_dict):
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    key_list = _dict.keys()
    if "inf_iter" in key_list:
        key_list.remove("inf_iter")
    num_keys = len(key_list)
    direct_sort_num = max(len(key_list)//size+1,1)
    if rank == 0:
        print(f"Direct sorting {direct_sort_num} keys per process",flush=True)

    my_key_list = []
    if rank*direct_sort_num < num_keys:
        my_key_list = key_list[rank*direct_sort_num:min((rank+1)*direct_sort_num,num_keys)]

    # Direct sort keys assigned to this rank
    my_results = []
    for key in my_key_list:
        val = _dict[key]
        if any(val["inf"]):
            this_value = list(zip(val["inf"],val["smiles"],val["model_iter"]))
            this_value.sort(key=lambda tup: tup[0])
            my_results = merge(this_value, my_results, num_return_sorted)

    # Merge results between ranks
    max_k = math.ceil(math.log2(size))
    max_j = size//2

    for k in range(max_k):
        offset = 2**k
        for j in range(max_j):
            #if rank ==0: print(f"rank 0 cond val is {k=} {j=} {offset=} {(2**(k+1))*j}")
            if rank == (2**(k+1))*j:         
                neighbor_result = comm.recv(source = rank + offset)
                merge(my_results,neighbor_result,num_return_sorted)
                #print(f"{rank=}: {k=} {offset=} {neighbor_result=}")
            if rank == (2**(k+1))*j + 2**k:
                comm.send(my_results,rank - offset)
        max_j = max(max_j//2,1)


    # rank 0 collects the final sorted list
    if rank == 0:
        # put data in candidate_dict
        top_candidates = my_results
        num_top_candidates = len(my_results)
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
            sort_val = {"inf": list(candidate_inf), "smiles": list(candidate_smiles), "model_iter": list(candidate_model_iter)}
            # check for new smiles
            if last_list_key == "-1":
                # save if this is the first iteration
                save_list(candidate_dict, ckey, sort_val)
            else:
                last_list = candidate_dict[last_list_key]
                if len(last_list["smiles"]) != len(candidate_smiles):
                    # Save list if it is a different length than previous list
                    save_list(candidate_dict, ckey, sort_val)
                else:
                    not_in_common = list(set(last_list["smiles"]) ^ set(candidate_smiles))
                    if len(not_in_common) > 0:
                        # Save list if there are smiles not shared by both
                        save_list(candidate_dict, ckey, sort_val)
                    else:
                        # if no new smiles, check if inference has been updated
                        last_list_model_iters = list(set(candidate_dict[last_list_key["model_iter"]]))
                        current_model_iters = list(set(candidate_model_iter))

                        if len(last_list_model_iters) == 1 and len(current_model_iters) == 1 and last_list_model_iters[0] != current_model_iters[0]:
                            # Save if every smiles in each list was inferred with a different model
                            save_list(candidate_dict, ckey, sort_val)
                        else:
                            # Check smiles by smiles for differences in ordering or model_iter
                            for i in range(len(candidate_smiles)):
                                if candidate_smiles[i] != last_list["smiles"][i] or candidate_model_iter[i] != last_list["model_iter"]:
                                    # Save list if there are some smiles done with different inference models
                                    save_list(candidate_dict, ckey, sort_val)
                                    break    
                
    MPI.Finalize()

def save_list(candidate_dict, ckey, sort_val):
    candidate_dict[ckey] = sort_val
    candidate_dict["sort_iter"] = int(ckey)
    candidate_dict["max_sort_iter"] = ckey
    print(f"candidate dictionary on iter {int(ckey)}",flush=True)



    
