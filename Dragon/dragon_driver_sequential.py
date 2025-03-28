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

from data_loader.data_loader_presorted import load_inference_data
from inference.launch_inference import launch_inference
from sorter.sorter import sort_dictionary_pg
from docking_sim.launch_docking_sim import launch_docking_sim
from training.launch_training import launch_training
from data_loader.data_loader_presorted import get_files
from driver_functions import max_data_dict_size


if __name__ == "__main__":

    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=8,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--max_iter', type=int, default=10,
                        help='Maximum number of iterations')
    parser.add_argument('--dictionary_timeout', type=int, default=10,
                        help='Timeout for Dictionary in seconds')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')

    args = parser.parse_args()
    # Start driver
    start_time = perf_counter()
    print("Begun dragon driver",flush=True)
    print(f"Reading inference data from path: {args.data_path}",flush=True)
    mp.set_start_method("dragon")

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    tot_nodelist = alloc.nodes

    tot_mem = args.mem_per_node * num_tot_nodes

    # Get info about gpus and cpus
    gpu_devices = os.getenv("GPU_DEVICES").split(",")
    num_gpus = len(gpu_devices)

    # for this sequential loop test set inference and docking to all the nodes and sorting and training to one node
    node_counts = {
        "sorting": num_tot_nodes,
        "training": 1,
        "inference": num_tot_nodes,
        "docking": num_tot_nodes,
    }

    nodelists = {}
    offset = 0
    for key in node_counts.keys():
        nodelists[key] = tot_nodelist[:node_counts[key]]

    # Set the number of nodes the dictionary uses
    num_dict_nodes = num_tot_nodes

    # Get info on the number of files
    base_path = pathlib.Path(args.data_path)
    files, num_files = get_files(base_path)

    tot_mem = args.mem_per_node*num_tot_nodes
    print(f"There are {num_files} files")

    # Set up and launch the inference data DDict and top candidate DDict
    data_dict_mem, candidate_dict_mem = max_data_dict_size(num_files)
    print(f"Setting data_dict size to {data_dict_mem} GB and candidate_dict size to {candidate_dict_mem} GB")

    if data_dict_mem + candidate_dict_mem > tot_mem:
        print(f"Sum of dictionary sizes exceed total mem: {data_dict_mem=} {candidate_dict_mem=} {tot_mem=}", flush=True)
        raise Exception("Not enough memory for DDicts")

    data_dict_mem *= (1024*1024*1024)
    candidate_dict_mem *= (1024*1024*1024)

    # Start distributed dictionary used for inference
    # inf_dd_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=Node(inf_dd_nodelist).hostname)
    # Note: the host name based policy, as far as I can tell, only takes in a single node, not a list
    #       so at the moment we can't specify to the inf_dd to run on a list of nodes.
    #       But by setting inf_dd_nodes < num_tot_nodes, we can make it run on the first inf_dd_nodes nodes only
   
    data_dd = DDict(args.managers_per_node, num_dict_nodes, data_dict_mem, trace=True)
    print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem}", flush=True)
    print(f"on {num_dict_nodes} nodes", flush=True)
    print(f"{data_dd.stats=}")
    
    # Launch the data loader component
    max_procs = args.max_procs_per_node * num_tot_nodes
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(
        target=load_inference_data,
        args=(
            data_dd,
            args.data_path,
            max_procs,
            num_tot_nodes * args.managers_per_node,
        ),
    )
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        print(f"Loaded inference data in {load_time:.3f} seconds", flush=True)
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")

    tic = perf_counter()
    print("Here are the stats after data loading...")
    print("++++++++++++++++++++++++++++++++++++++++")
    print(data_dd.stats)
    toc = perf_counter()
    load_time = toc - tic
    print(f"Retrieved dictionary stats in {load_time:.3f} seconds", flush=True)
    num_keys = len(data_dd.keys())
    
    cand_dd = DDict(args.managers_per_node, num_dict_nodes, candidate_dict_mem, policy=None, trace=True)
    print(f"Launched Dragon Dictionary for top candidates with total memory size {candidate_dict_mem}", flush=True)
    print(f"on {num_dict_nodes} nodes", flush=True)
    
    # Number of top candidates to produce
    if num_tot_nodes < 3:
        top_candidate_number = 1000
    else:
        top_candidate_number = 5000

    max_iter = args.max_iter
    iter = 0
    while iter < max_iter:
        print(f"*** Start loop iter {iter} ***")
        #print(f"{data_dd.stats=}")
        iter_start = perf_counter()
        # Launch the data inference component
        num_procs = num_gpus*node_counts["inference"]

        print(f"Launching inference with {num_procs} processes ...", flush=True)
        if num_tot_nodes < 3:
            inf_num_limit = 8
            print(
                f"Running small test on {num_tot_nodes}; limiting {inf_num_limit} keys per inference worker"
            )
        else:
            inf_num_limit = None

        tic = perf_counter()
        inf_proc = mp.Process(
            target=launch_inference,
            args=(
                data_dd,
                nodelists["inference"],
                num_procs,
                inf_num_limit,
            ),
        )
        inf_proc.start()
        inf_proc.join()
        toc = perf_counter()
        infer_time = toc - tic
        print(f"Performed inference in {infer_time:.3f} seconds \n", flush=True)

        # Launch data sorter component and create candidate dictionary
        print(f"Launching sorting ...", flush=True)
        tic = perf_counter()
        if iter == 0:
            cand_dd["max_sort_iter"] = "-1"

        if os.getenv("USE_MPI_SORT"):
            print("Using MPI sort",flush=True)
            max_sorter_procs = args.max_procs_per_node*node_counts["sorting"]
            sorter_proc = mp.Process(target=sort_dictionary_pg, 
                                     args=(data_dd,
                                           top_candidate_number, 
                                           max_sorter_procs, 
                                           nodelists["sorting"],
                                           cand_dd))
            sorter_proc.start()
            sorter_proc.join()
        else:
            print("Filter sort not yet included", flush=True)

        toc = perf_counter()
        infer_time = toc - tic
        print(f"Performed sorting of {num_keys} keys in {infer_time:.3f} seconds \n", flush=True)

        # Launch Docking Simulations
        print(f"Launched Docking Simulations", flush=True)
        tic = perf_counter()
        num_procs = args.max_procs_per_node * node_counts["docking"]
        dock_proc = mp.Process(
            target=launch_docking_sim,
            args=(cand_dd, iter, num_procs, nodelists["docking"]),
        )
        dock_proc.start()
        dock_proc.join()
        toc = perf_counter()
        infer_time = toc - tic
        #os.rename("finished_run_docking.log", f"finished_run_docking_{iter}.log")
        print(f"Performed docking in {infer_time:.3f} seconds \n", flush=True)

        print(f"Candidate Dictionary stats:", flush=True)
        print(cand_dd.stats)

        # Launch Training
        print(f"Launched Fine Tune Training", flush=True)
        tic = perf_counter()
        BATCH = 64
        EPOCH = 500
        train_proc = mp.Process(
            target=launch_training,
            args=(
                data_dd,
                nodelists["training"][0],  # training is always 1 node
                cand_dd,
                BATCH,
                EPOCH,
            ),
        )
        train_proc.start()
        train_proc.join()
        toc = perf_counter()
        print(f"Performed training in {toc-tic} seconds \n", flush=True)
        iter_end = perf_counter()
        print(
            f"Performed iter {iter} in {iter_end - iter_start} seconds \n", flush=True
        )
        iter += 1

    # Close the dictionary
    print("Closing the Dragon Dictionary and exiting ...", flush=True)
    cand_dd.destroy()
    data_dd.destroy()
    end_time = perf_counter()
    print(f"Total time {end_time - start_time} s", flush=True)
