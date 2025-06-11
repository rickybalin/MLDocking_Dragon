import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from time import perf_counter
import argparse
import os
import socket
import csv
#from dragon.utils import host_id
#from dragon.data.ddict import DDict
#from dragon.globalservices.api_setup import connect_to_infrastructure
#connect_to_infrastructure()


def mpi_sort(num_files: int, num_return_sorted: int):
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0: print(f"MPI Sorting starting on {size} ranks")

    debug = False
    tic_start = perf_counter()
    
    # Read the files
    tic_files = perf_counter()
    data_path = os.getenv("WORK_PATH")
    predicted_data_path = data_path + "/predicted_data"
    if rank == 0:
        # Get list of files
        all_files = os.listdir(predicted_data_path)
        all_files = [os.path.join(predicted_data_path, f) \
                    for f in all_files \
                    if os.path.isfile(os.path.join(predicted_data_path, f))]
        avg = len(all_files) // size
        rem = len(all_files) % size
        split_files = []
        start = 0
        for i in range(size):
            end = start + avg + (1 if i < rem else 0)
            split_files.append(all_files[start:end])
            start = end
        #split_files = all_files[rank::size]
    else:
        split_files = None
    file_subset = comm.scatter(split_files, root=0)
    toc_files = perf_counter()
    files_time = toc_files - tic_files
    if debug: print(f'Rank {rank} is reading {len(file_subset)} files',flush=True)

    # Create communicator for local ranks only
    tic = perf_counter()
    hostname = socket.gethostname()
    hostnames = comm.allgather(hostname)
    for h in hostnames:
        color = True if hostname == h else MPI.UNDEFINED
        new_comm = comm.Split(color=color, key=rank)
        if hostname == h:
            my_host_comm = new_comm
    local_rank = my_host_comm.Get_rank()
    local_size = my_host_comm.Get_size()
    toc = perf_counter()
    setup_time = toc - tic
    all_setup_times = comm.gather(setup_time, root=0)
    if rank == 0:
        ave_setup_time = sum(all_setup_times) / len(all_setup_times)
        print(f"Average file and comm setup time is {ave_setup_time:.3f} seconds", flush=True)

    # Direct sort keys assigned to this rank
    tic = perf_counter()
    my_results = []
    val = {"smiles": [], "score": []}
    read_time = 0
    for i,file in enumerate(file_subset):
        tic_read = perf_counter()
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key, value in row.items():
                    if key == "score":
                        try:
                            val[key].append(float(value))
                        except:
                            val[key].append(0.0)
                    else:
                        val[key].append(value)
        toc_read = perf_counter()
        read_time += toc_read - tic_read

        # Only include this key if it has non-zero inf values
        num_smiles = len(val['score'])
        this_value = list(zip(val["score"],
                                val["smiles"]))
                                # for _ in range(num_smiles))
        my_results.extend(this_value)
        my_results.sort(key=lambda tup: tup[0])
        my_results = my_results[-num_return_sorted:]
     
    my_results.sort(key=lambda tup: tup[0])
    my_results = my_results[-num_return_sorted:]
    toc = perf_counter()
    direct_sort_time = toc - tic
    all_direct_sort_times = comm.gather(direct_sort_time, root=0)
    if rank == 0:
        ave_direct_sort_time = sum(all_direct_sort_times) / len(all_direct_sort_times)
        print(f"Average direct sort time is {ave_direct_sort_time:.3f} seconds", flush=True)
    #print(f"Rank {rank} finished direct sort in {toc-tic} seconds; found {len(my_results)} results; starting local merge",flush=True)
            
    # Combine local results
    com_tic = perf_counter()
    local_results = []
    num_my_results = len(my_results)
    for i in range(num_return_sorted):
    #for i,res in enumerate(my_results[::-1]):
        tic = perf_counter()
        if i < num_my_results:
            res = my_results[::-1][i]
        else:
            res = (0.0,'dummy',-1)
        gathered_local_results = my_host_comm.gather(res, root=0)
        toc = perf_counter()
        if local_rank == 0:
            continue_loop = True
            #print(f"Sort rank {rank} gathered in {toc-tic} seconds on iter {i}",flush=True)
            last_results = local_results.copy()
            local_results.extend(gathered_local_results)
            local_results.sort(key=lambda tup: tup[0])
            local_results = local_results[-num_return_sorted:]
            # Since we are pulling results in reverse order from all lists,
            # once adding new entries does not change the trucated list
            # there are no more entries that will make it in the top candidates.
            # The local_rank can stop gathering results.
            if last_results == local_results:
                #print(f"Rank {rank} exiting local merge",flush=True)
                continue_loop = False
        else:
            assert gathered_local_results is None
            local_results = None
            continue_loop = True
        continue_loop = my_host_comm.bcast(continue_loop, root=0)
        if not continue_loop:
            break
    #print(f"Rank {rank} finished node local merge",flush=True)

    # Merge results between nodes
    # Make a communicator that includes all ranks with local_rank=0    
    
    color = True if local_rank == 0 else MPI.UNDEFINED
    root_comm = comm.Split(color=color, key=rank)
    root_rank = None
    #print(f"Rank {rank} starting node global merge",flush=True)
    if local_rank == 0:
        root_size = root_comm.Get_size()
        root_rank = root_comm.Get_rank()
        #print(f"Rank {rank} has {local_rank=} and {root_rank=}",flush=True)
        all_results = []
        num_local_results = len(local_results)
        for i in range(num_return_sorted):
            if i < num_local_results:
                res = local_results[::-1][i]
            else:
                res = (0.0,'dummy',-1)
            #print(f"Rank {rank} gathering results for iter {i}",flush=True)
            gathered_node_results = root_comm.gather(res, root=0)
            #print(f"Rank {rank} finished gather for iter {i}; is None? {gathered_node_results==None}",flush=True)
            if root_rank == 0:
                continue_loop = True
                last_results = all_results.copy()
                all_results.extend(gathered_node_results)
                all_results.sort(key=lambda tup: tup[0])
                all_results = all_results[-num_return_sorted:]
                if last_results == all_results:
                    #print(f"Rank {rank} exiting node merge after iter {i}",flush=True)
                    continue_loop = False
            else:
                assert gathered_node_results is None
                all_results = None
                continue_loop = True
            continue_loop = root_comm.bcast(continue_loop, root=0)
            if not continue_loop:
                break
    com_toc = perf_counter()
    com_time = com_toc - com_tic

    write_time = 0
    if root_rank == 0:
        print(f"Collected sorted results on rank 0 in {com_time:.3f} seconds",flush=True)
        #print(f"{my_results=}")
        # put data in candidate_dict

        #print(f"Rank {rank} moving onto saving",flush=True)
        
        #top_candidates = all_results
        # filter out any 0 values or dummy values
        #print(f"Number of results {len(all_results)=}",flush=True)
        #print(all_results,flush=True)
        top_candidates = [c for c in all_results if float(c[0]) > 0 and c[1] != 'dummy']
        num_top_candidates = len(top_candidates)
        print(f"Collected {num_top_candidates=}",flush=True)
        if num_top_candidates > 0:
            
            candidate_inf,candidate_smiles = zip(*top_candidates)
            non_zero_infs = len([cinf for cinf in candidate_inf if cinf != 0])
            
            print(f"Sorted list contains {non_zero_infs} non-zero inference results out of {len(candidate_inf)}",flush=True)
            sort_val = {"inf": list(candidate_inf), "smiles": list(candidate_smiles)}
        
            tic_write = perf_counter()
            sorted_data_path = data_path + "/sorted_data"
            if not os.path.exists(sorted_data_path):
                os.makedirs(sorted_data_path)
            with open(sorted_data_path+'/sorted_smiles.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['smiles', 'score'])
                writer.writerows(zip(candidate_smiles,candidate_inf))
            toc_write = perf_counter()
            write_time = toc_write - tic_write
    
    comm.Barrier()
    toc_end = perf_counter()
            #save_list(candidate_dict, current_sort_iter+1, sort_val)    
    #print(f"Rank {rank} done",flush=True)
    if rank == 0:
        total_io_time = read_time + write_time + files_time
        total_comm_time = setup_time + com_time
        print(f"Performed sorting of {num_top_candidates} compounds: total={toc_end-tic_start}, IO={total_io_time}, comm={total_comm_time}",flush=True)
    MPI.Finalize()
    
    return

def save_list(candidate_dict, ckey, sort_val):
    candidate_dict.bput("current_sort_list", sort_val)
    candidate_dict.bput("current_sort_iter", ckey)
    print(f"candidate dictionary on iter {int(ckey)} and saved",flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--num_files', type=int, default=1,
                        help='number of files to read')
    parser.add_argument('--num_return_sorted', type=int, default=1000,
                        help='number of sorted smiles to return')
    args = parser.parse_args()
    mpi_sort(args.num_files, args.num_return_sorted)



    
