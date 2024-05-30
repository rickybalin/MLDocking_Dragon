import os
from time import perf_counter
import argparse
from typing import List

import dragon
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict
#from dragon.data.distdictionary.dragon_dict import DragonDict
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy

from data_loader.data_loader_presorted import load_inference_data
from inference.launch_inference import launch_inference
from sorter.sorter import sort_controller as sort_dictionary
from docking_sim.launch_docking_sim import launch_docking_sim
from training.launch_training import launch_training

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
    parser.add_argument('--data_dictionary_mem_fraction', type=float, default=0.7,
                        help='fraction of memory dedicated to data dictionary')
    parser.add_argument('--inference_node_num', type=int, default=1,
                        help='number of nodes running inference')
    parser.add_argument('--sorting_node_num', type=int, default=1,
                        help='number of nodes running sorting')
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
    parser.add_argument('--only_load_data', type=int, default=0,
                        help='Flag to only do the data loading')
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

    # num_training_nodes = 1
    # num_sorting_nodes = args.sorting_node_num
    # num_inference_nodes = args.inference_node_num
    # num_docking_nodes = num_tot_nodes - num_inference_nodes - num_sorting_nodes - num_training_nodes
    
    # for this 1.5 loop test set inference and docking to all the nodes and sorting and training to one node
    node_counts = {"sorting": 1, 
                    "training": 1, 
                    "inference": num_tot_nodes,
                    "docking": num_tot_nodes}
    
    nodelists = {}
    offset = 0
    for key in node_counts.keys():
        nodelists[key] = tot_nodelist[:node_counts[key]]
        
    
    print(f"{nodelists=}")    

    # Set up and launch the inference DDict
    # inf_dd_nodelist = tot_nodelist[:args.inf_dd_nodes]
    # inf_dd_mem_size = args.mem_per_node*args.inf_dd_nodes
    # inf_dd_mem_size *= (1024*1024*1024)
    data_dict_mem = int(args.data_dictionary_mem_fraction*tot_mem)
    candidate_dict_mem = tot_mem - data_dict_mem
    data_dict_mem *= (1024*1024*1024)
    candidate_dict_mem *= (1024*1024*1024)

    # Start distributed dictionary used for inference
    #inf_dd_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=Node(inf_dd_nodelist).hostname)
    # Note: the host name based policy, as far as I can tell, only takes in a single node, not a list
    #       so at the moment we can't specify to the inf_dd to run on a list of nodes.
    #       But by setting inf_dd_nodes < num_tot_nodes, we can make it run on the first inf_dd_nodes nodes only
    data_dd = DDict(args.managers_per_node, num_tot_nodes, data_dict_mem, 
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

    cand_dd = DDict(args.managers_per_node, num_tot_nodes, candidate_dict_mem, 
                    timeout=3600, policy=None, 
                    num_streams_per_manager=args.channels_per_manager)
    print(f"Launched Dragon Dictionary for top candidates with total memory size {candidate_dict_mem}", flush=True)
    print(f"on {num_tot_nodes} nodes", flush=True)

    # Set Event and Launch other Components
    # Set the continue event to None for each component to run one iter
    continue_event = None  
    
    # Number of top candidates to produce
    top_candidate_number = 5000

    max_iter = 3
    iter = 0
    while iter < max_iter:
        iter_start = perf_counter()
        # Launch the data inference component
        num_procs = 4*node_counts["inference"]
        #inf_num_limit = 16
        print(f"Launching inference with {num_procs} processes ...", flush=True)
        tic = perf_counter()
        inf_proc = mp.Process(target=launch_inference, args=(data_dd, 
                                                            nodelists["inference"], 
                                                            num_procs, 
                                                            continue_event,
                                                            inf_num_limit,)
                                                            )
        inf_proc.start()
        inf_proc.join()
        toc = perf_counter()
        infer_time = toc - tic
        print(f"Performed inference in {infer_time:.3f} seconds \n", flush=True)

        # Launch data sorter component and create candidate dictionary
        tic = perf_counter()
        max_sorter_procs = args.max_procs_per_node*node_counts["sorting"]
        sorter_proc = mp.Process(target=sort_dictionary, 
                                args=(data_dd, 
                                    top_candidate_number, 
                                    max_sorter_procs, 
                                    nodelists["sorting"],
                                    cand_dd, 
                                    continue_event))
        sorter_proc.start()
        sorter_proc.join()
        toc = perf_counter()
        infer_time = toc - tic
        print(f"Performed sorting in {infer_time:.3f} seconds \n", flush=True)
    
        # Launch Docking Simulations
        print(f"Launched Docking Simulations", flush=True)
        tic = perf_counter()
        num_procs = args.max_procs_per_node*node_counts["docking"]
        dock_proc = mp.Process(target=launch_docking_sim, 
                                args=(cand_dd, 
                                        nodelists["docking"], 
                                        num_procs, 
                                        continue_event))
        dock_proc.start()
        dock_proc.join()
        toc = perf_counter()
        infer_time = toc - tic
        print(f"Performed docking in {infer_time:.3f} seconds \n", flush=True)
        
        # Launch Training
        print(f"Launched Fine Tune Training", flush=True)
        tic = perf_counter()
        BATCH = 64
        EPOCH = 500
        train_proc = mp.Process(target=launch_training, 
                                args=(data_dd, 
                                        nodelists["training"][0], # training is always 1 node
                                        cand_dd, 
                                        continue_event,
                                        BATCH,
                                        EPOCH,
                                        top_candidate_number))
        train_proc.start()
        train_proc.join()
        toc = perf_counter()
        print(f"Performed training in {toc-tic} seconds \n", flush=True)
        iter_end = perf_counter()
        print(f"Performed iter {iter} in {iter_end - iter_start} seconds \n", flush=True)

    # Close the dictionary
    print("Closing the Dragon Dictionary and exiting ...", flush=True)
    cand_dd.destroy()
    data_dd.destroy()
    end_time = perf_counter()
    print(f"Total time {end_time - start_time} s", flush=True)



