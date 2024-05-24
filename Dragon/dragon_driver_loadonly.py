import os
from time import perf_counter
import numpy as np
import argparse
from typing import List

import dragon
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict
#from dragon.data.distdictionary.dragon_dict import DragonDict
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy

from data_loader.data_loader_presorted import load_inference_data


def parseNodeList() -> List[str]:
    """
    Parse the node list provided by the scheduler

    :return: PBS node list
    :rtype: list 
    """
    nodelist = []
    hostfile = os.getenv('PBS_NODEFILE')
    with open(hostfile) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
        nodelist = [line.split('.')[0] for line in nodelist]
    return nodelist

if __name__ == "__main__":
    
    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--inf_dd_nodes', type=int, default=1,
                        help='number of nodes the dictionary distributed across')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=8,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--dictionary_timeout', type=int, default=10,
                        help='Timeout for Dictionary in seconds')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    parser.add_argument('--only_load_data', type=int, default=0,
                        help='Flag to only do the data loading')
    
    args = parser.parse_args()
    
    

    print("Begun dragon driver", flush=True)
    print(f"Reading inference data from path: {args.data_path}", flush=True)
    # Get information about the allocation
    mp.set_start_method("dragon")
    pbs_nodelist = parseNodeList()
    alloc = System()
    num_tot_nodes = alloc.nnodes()
    tot_nodelist = alloc.nodes

    # Set up and launch the inference DDict
    inf_dd_nodelist = tot_nodelist[:args.inf_dd_nodes]
    inf_dd_mem_size = args.mem_per_node*args.inf_dd_nodes
    inf_dd_mem_size *= (1024*1024*1024)
    
    num_sort_nodes = max(num_tot_nodes - args.inf_dd_nodes, 1)
    sort_dd_nodelist = tot_nodelist[args.inf_dd_nodes:num_tot_nodes]
    if len(sort_dd_nodelist) == 0:
        sort_dd_nodelist = tot_nodelist[-1]

    # Start distributed dictionary used for inference
    #inf_dd_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=Node(inf_dd_nodelist).hostname)
    # Note: the host name based policy, as far as I can tell, only takes in a single node, not a list
    #       so at the moment we can't specify to the inf_dd to run on a list of nodes.
    #       But by setting inf_dd_nodes < num_tot_nodes, we can make it run on the first inf_dd_nodes nodes only
    inf_dd_policy = None
    inf_dd = DDict(args.managers_per_node, args.inf_dd_nodes, inf_dd_mem_size, 
                   timeout=args.dictionary_timeout, policy=inf_dd_policy)
    print(f"Launched Dragon Dictionary for inference with total memory size {inf_dd_mem_size}", flush=True)
    print(f"on {args.inf_dd_nodes} nodes", flush=True)
    print(f"{pbs_nodelist[:args.inf_dd_nodes]}", flush=True)

    # Place key used to stop workflow (possible way of syncing components)
    #inf_dd['keep_runing'] = True # needs update to inference.run_inference.split_dict_keys
    
    # Launch the data loader component
    max_procs = args.max_procs_per_node*args.inf_dd_nodes
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(target=load_inference_data, 
                             args=(inf_dd, 
                                   args.data_path, 
                                   max_procs, 
                                   args.inf_dd_nodes*args.managers_per_node),
                            )
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        print(f"Loaded inference data in {load_time:.3f} seconds", flush=True)
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")

    
    inf_dd.destroy()
   


