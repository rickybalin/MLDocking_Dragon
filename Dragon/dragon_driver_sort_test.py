import os
#import logging
#logger = logging.getLogger(__name__)
from time import perf_counter
import argparse
from typing import List
import shutil
import pathlib
import dragon
from math import ceil
import multiprocessing as mp
from dragon.data.ddict import DDict
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy

from data_loader.data_loader_presorted import load_inference_data, get_files
from sorter.sorter import sort_dictionary_pg, sort_dictionary_ddict
from data_loader.data_loader_presorted import get_files
from driver_functions import max_data_dict_size
#from inference.launch_inference import launch_inference

if __name__ == "__main__":

    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--data_dictionary_mem_fraction', type=float, default=0.7,
                        help='fraction of memory dedicated to data dictionary')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=8,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    parser.add_argument('--top_candidates', type=int, default=1000,
                        help='Number of compounds in top candidate list')
    args = parser.parse_args()
  
    print("Begun dragon driver",flush=True)
    print(f"Reading inference data from path: {args.data_path}",flush=True)
    mp.set_start_method("dragon")

   # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    tot_nodelist = alloc.nodes

    # Get info on the number of files
    base_path = pathlib.Path(args.data_path)
    files, num_files = get_files(base_path)
    print(f"There are {num_files} files")

    # Start distributed dictionary and load data
    tot_mem = num_tot_nodes*args.mem_per_node
    data_dict_mem, candidate_dict_mem = max_data_dict_size(num_files, max_pool_frac = 0.5)
    print(f"Setting data_dict size to {data_dict_mem} GB and candidate_dict size to {candidate_dict_mem} GB")
    if data_dict_mem + candidate_dict_mem > tot_mem:
        print(f"Sum of dictionary sizes exceed total mem: {data_dict_mem=} {candidate_dict_mem=} {tot_mem=}", flush=True)
        raise Exception("Not enough memory for DDicts")
    data_dict_mem *= (1024*1024*1024)
    candidate_dict_mem *= (1024*1024*1024)

    # Start inference DDict
    data_dd = DDict(args.managers_per_node, 
                    num_tot_nodes, 
                    data_dict_mem, 
                    policy=None,
                    trace=False
    )
    print(f"Launched inference DDict with total memory {data_dict_mem} on {num_tot_nodes} nodes", flush=True)

    # Start candidate DDict
    cand_dd = DDict(args.managers_per_node, 
                    num_tot_nodes, 
                    candidate_dict_mem, 
                    policy=None, 
                    trace=False
    )
    print(f"Launched candidate DDict with total memory {candidate_dict_mem} on {num_tot_nodes} nodes", flush=True)

    # Launch data loader
    max_loader_procs = args.max_procs_per_node * num_tot_nodes
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(
        target=load_inference_data,
        args=(
            data_dd,
            args.data_path,
            max_loader_procs,
            num_tot_nodes * args.managers_per_node,
            num_files
        ),
    )
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        num_keys = len(data_dd.keys())
        print(f"Loaded {num_keys} keys in {load_time:.3f} seconds", flush=True)
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")
    loader_proc.close()
    #tic = perf_counter()
    print("Here are the stats after data loading...")
    print("++++++++++++++++++++++++++++++++++++++++")
    print(data_dd.stats)
    #toc = perf_counter()
    #load_time = toc - tic
    #print(f"Retrieved dictionary stats in {load_time:.3f} seconds", flush=True)

    # Sort with MPI
    print(f"\n\nLaunching sorting with MPI ...", flush=True)
    cand_dd.bput("max_sort_iter",-1)
    cand_dd.bput('current_sort_iter', -1)
    max_sorter_procs = args.max_procs_per_node*num_tot_nodes
    sorter_proc = mp.Process(target=sort_dictionary_pg, 
                                args=(data_dd,
                                    args.top_candidates, 
                                    max_sorter_procs, 
                                    tot_nodelist,
                                    cand_dd))
    sorter_proc.start()
    sorter_proc.join()
    sorter_proc.close()
    print(f"Finished sorting with MPI\n\n", flush=True)

    # Sort with DDict
    print(f"\n\nLaunching sorting with DDict ...", flush=True)
    cand_dd.bput("max_sort_iter",-1)
    cand_dd.bput('current_sort_iter', -1)
    sort_dictionary_ddict(data_dd, args.top_candidates, cand_dd)
    print(f"Finished sorting with DDict\n\n", flush=True)

