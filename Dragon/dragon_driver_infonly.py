import os
from time import perf_counter
import argparse
from typing import List

import dragon
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy

from data_loader.data_loader_presorted import load_inference_data
from inference.launch_inference import launch_inference

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
    parser.add_argument('--dictionary_node_num', type=int, default=-1,
                        help='Number of nodes the dictionary distributed across; -1 is all nodes')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--channels_per_manager', type=int, default=20,
                        help='channels per manager for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=8,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--dictionary_timeout', type=int, default=10,
                        help='Timeout for Dictionary in seconds')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
   
    args = parser.parse_args()

    start_time = perf_counter()
    print("Begun dragon driver", flush=True)
    print(f"Reading inference data from path: {args.data_path}", flush=True)
    # Get information about the allocation
    mp.set_start_method("dragon")
    pbs_nodelist = parseNodeList()
    alloc = System()
    num_tot_nodes = alloc.nnodes()
    tot_nodelist = alloc.nodes
    tot_mem = args.mem_per_node*num_tot_nodes

    # Set up and launch the inference DDict
    data_dict_mem = tot_mem #int(args.data_dictionary_mem_fraction*tot_mem)
    data_dict_mem *= (1024*1024*1024)
    
    # Start distributed dictionary used for inference
    #inf_dd_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=Node(inf_dd_nodelist).hostname)
    # Note: the host name based policy, as far as I can tell, only takes in a single node, not a list
    #       so at the moment we can't specify to the inf_dd to run on a list of nodes.
    #       But by setting inf_dd_nodes < num_tot_nodes, we can make it run on the first inf_dd_nodes nodes only
    dictionary_node_num = num_tot_nodes
    if args.dictionary_node_num != -1:
        if args.dictionary_node_num > 0:
            dictionary_node_num = args.dictionary_node_num
        else:
            print(f"Dictionary nodes not valid! {args.dictionary_node_num=}",flush=True)
            sys.exit()
    print(f"Placing dictionary on {dictionary_node_num} nodes out of {num_tot_nodes} nodes", flush=True)

    data_dict_mem = args.mem_per_node*dictionary_node_num
    data_dict_mem *= (1024*1024*1024)

    data_dd = DDict(args.managers_per_node, dictionary_node_num, data_dict_mem, 
                   timeout=3600, num_streams_per_manager=args.channels_per_manager)
    print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem}", flush=True)
    print(f"on {num_tot_nodes} nodes", flush=True)
    
    # Launch the data loader component
    max_procs = args.max_procs_per_node*num_tot_nodes
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(target=load_inference_data, 
                             args=(data_dd, 
                                   args.data_path, 
                                   max_procs, 
                                   num_tot_nodes*args.managers_per_node),
                            )
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        print(f"Loaded inference data in {load_time:.3f} seconds", flush=True)
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")

    # Set Event and Launch other Components
    continue_event = None
        
    # Launch the data inference component
    inference_node_list = tot_nodelist
    # If the dictionary was only placed on the first dictionary_node_num nodes,
    # place the inference processes on the remainder of the nodes
    if num_tot_nodes - dictionary_node_num > 0:
        print(f"Placing inference processes on remaining nodes: {num_tot_nodes-dictionary_node_num} nodes", flush=True)
        inference_node_list = tot_nodelist[dictionary_node_num:]
    else:
        print(f"Placing inference processes on all nodes {num_tot_nodes=}", flush=True)
    
    num_procs = 4*len(inference_node_list)
    inf_num_limit=None
    print(f"Launching inference with {num_procs} processes ...", flush=True)
    tic = perf_counter()
    inf_proc = mp.Process(target=launch_inference, args=(data_dd, 
                                                    inference_node_list, 
                                                    num_procs, 
                                                    continue_event,
                                                    inf_num_limit,))
    inf_proc.start()
    
    inf_proc.join()
    toc = perf_counter()
    infer_time = toc - tic
    print(f"Performed inference in {infer_time:.3f} seconds \n", flush=True)

    # Close the dictionary
    print("Closing the Dragon Dictionary and exiting ...", flush=True)

    data_dd.destroy()
    end_time = perf_counter()
    print(f"Total time {end_time - start_time} s", flush=True)
   


