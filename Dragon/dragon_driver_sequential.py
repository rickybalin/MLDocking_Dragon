import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from time import perf_counter
import argparse
from typing import List
import random
import shutil
import pathlib
import dragon
from math import ceil
import multiprocessing as mp
from dragon.data.ddict import DDict
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy

#import logging
#log = logging.getLogger(__name__)
#logging.basicConfig(
#    level=logging.INFO,
#    format='%(levelname)s - %(message)s'
#)

from data_loader.data_loader_presorted import load_inference_data
from inference.launch_inference import launch_inference
from sorter.sorter import sort_dictionary_pg, sort_dictionary
from docking_sim.launch_docking_sim import launch_docking_sim
from training.launch_training import launch_training
from data_loader.data_loader_presorted import get_files
from data_loader.model_loader import load_pretrained_model
from driver_functions import max_data_dict_size, output_sims


if __name__ == "__main__":

    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--inference_node_num', type=int, default=1,
                        help='number of nodes running inference')
    parser.add_argument('--sorting_node_num', type=int, default=1,
                        help='number of nodes running sorting')
    parser.add_argument('--simulation_node_num', type=int, default=1,
                        help='number of nodes running docking simulation')
    parser.add_argument('--training_node_num', type=int, default=1,
                        help='number of nodes running training')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=8,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--max_iter', type=int, default=1,
                        help='Maximum number of iterations')
    parser.add_argument('--dictionary_timeout', type=int, default=10,
                        help='Timeout for Dictionary in seconds')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    parser.add_argument('--logging', type=str, default="info",
                        help='Logging level')
    args = parser.parse_args()

    # Start driver
    start_time = perf_counter()
    print("Begun dragon driver")
    print(f"Reading inference data from path: {args.data_path}", flush=True)
    mp.set_start_method("dragon")

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    print(f"Running on {num_tot_nodes} total nodes",flush=True)
    tot_nodelist = alloc.nodes
    tot_mem = args.mem_per_node * num_tot_nodes

    # Distribute nodes
    print()
    if args.inference_node_num != args.sorting_node_num:
        raise("Inference and sorting nodes must be the same for this colocated deployment!")
    if num_tot_nodes != (args.inference_node_num + args.simulation_node_num): 
        raise(f"Node partitioning not valid! Inference and simulation nodes must add up to total nodes.")
    node_counts = {
        "inference": args.inference_node_num,
        "sorting": args.sorting_node_num,
        "simulation": args.simulation_node_num,
        "training": args.training_node_num, 
    }
    nodelist = {
        "inference": tot_nodelist[:args.inference_node_num],
        "sorting": tot_nodelist[:args.sorting_node_num],
        "simulation": tot_nodelist[args.inference_node_num:],
        "training": [tot_nodelist[-1]]
    }
    for key, val in nodelist.items():
        print(f"Component {key} running on {val} nodes",flush=True)

    # Get info about gpus and cpus
    gpu_devices = os.getenv("GPU_DEVICES")
    if gpu_devices is not None:
        gpu_devices = gpu_devices.split(",")
        num_gpus = len(gpu_devices)
    else:
        num_gpus = 0

    # Set up and launch the dictionaries
    # There are 3 dictionaries:
    # 1. data dictionary for inference
    # 2. simulation dictionary for docking simulation results
    # 3. model and candidate dictionary for training
    # The model and candidate dictionary will be checkpointed
    num_files = 24
    data_dict_mem, sim_dict_mem, model_list_dict_mem = max_data_dict_size(num_files, node_counts, max_pool_frac=0.5)
    print(f"Setting data_dict size to {data_dict_mem} GB")
    print(f"Setting sim_dict size to {sim_dict_mem} GB")
    print(f"Setting model_list_dict size to {model_list_dict_mem} GB")

    # Convert memory sizes to bytes
    data_dict_mem *= (1024 * 1024 * 1024)
    sim_dict_mem *= (1024 * 1024 * 1024)
    model_list_dict_mem *= (1024 * 1024 * 1024)

    # Initialize Dragon Dictionaries for inference, docking simulation, and model list
    dd_cpu_bind = [48, 49, 50, 51, 99, 100, 101, 103]
    data_dd_policy = [Policy(placement=Policy.Placement.HOST_NAME, host_name=Node(nodelist["inference"][node]).hostname, cpu_affinity=dd_cpu_bind) \
                      for node in range(len(nodelist["inference"]))]
    data_dd = DDict(None, 
                    None, 
                    data_dict_mem, 
                    policy=data_dd_policy)
    print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem} on {node_counts['inference']} nodes", flush=True)
    sim_dd_policy = [Policy(placement=Policy.Placement.HOST_NAME, host_name=Node(nodelist["simulation"][node]).hostname, cpu_affinity=dd_cpu_bind) \
                      for node in range(len(nodelist["simulation"]))]
    sim_dd = DDict(None, 
                   None, 
                   sim_dict_mem,
                   policy=sim_dd_policy)
    print(f"Launched Dragon Dictionary for docking simulation with total memory size {sim_dict_mem} on {node_counts['simulation']} nodes", flush=True)
    model_dd_policy = Policy(cpu_affinity=dd_cpu_bind)
    model_list_dd = DDict(args.managers_per_node, 
                          num_tot_nodes, 
                          model_list_dict_mem, 
                          policy=model_dd_policy, 
                          working_set_size=10)
    print(f"Launched Dragon Dictionary for model list with total memory size {model_list_dict_mem} on {num_tot_nodes} nodes", flush=True)


    # Load data into the data dictionary
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    max_procs = (args.max_procs_per_node-len(dd_cpu_bind)*2) * node_counts["inference"]
    tic = perf_counter()
    loader_proc = mp.Process(
        target=load_inference_data,
        args=(
            data_dd,
            args.data_path,
            max_procs,
            node_counts["inference"] * args.managers_per_node,
        ),
        kwargs={
            'num_files': num_files,
            'nodelist': None,
            'load_split_factor': 1
        }
    )
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        print(f"Loaded inference data in {load_time:.3f} seconds", flush=True)
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")

    if args.logging == "debug":
        print("Here are the stats after data loading...")
        print("Data Dictionary stats:", flush=True)
        print(data_dd.stats)

    # Load pretrained model
    load_pretrained_model(model_list_dd)

    # Initialize simulated compounds list
    sim_dd.bput('simulated_compounds', [])

    # Update driver log
    with open("driver_times.log", "w") as f:
        f.write(f"# {load_time=}\n")
    num_keys = len(data_dd.keys())
    with open("driver_times.log", "a") as f:
        f.write(f"# {num_keys=}\n")
    with open("driver_times.log", "a") as f:
        f.write(f"# {num_files=}\n")
    
    # Number of top candidates to produce
    if num_tot_nodes < 4:
        top_candidate_number = 1000
    else:
        top_candidate_number = 10000

    print(f"Finished workflow setup in {(perf_counter()-start_time):.3f} seconds", flush=True)

    # Start sequential loop
    max_iter = args.max_iter
    iter = 0
    with open("driver_times.log", "a") as f:
        f.write(f"# iter  infer_time  sort_time  dock_time  train_time \n")
    while iter < max_iter:
        print(f"*** Start loop iter {iter} ***")
        iter_start = perf_counter()

        print(f"Current checkpoint: {model_list_dd.checkpoint_id}", flush=True)

        # Launch the data inference component
        num_procs = num_gpus*node_counts["inference"]
        print(f"Launching inference with {num_procs} processes ...", flush=True)
        if num_tot_nodes == 1:
            inf_num_limit = 8
            print(f"Running small test on {num_tot_nodes}; limiting {inf_num_limit} keys per inference worker")
        else:
            inf_num_limit = None

        tic = perf_counter()
        inf_proc = mp.Process(
            target=launch_inference,
            args=(
                data_dd,
                model_list_dd,
                nodelist["inference"],
                num_procs,
                inf_num_limit,
            ),
        )
        inf_proc.start()
        inf_proc.join()
        toc = perf_counter()
        infer_time = toc - tic
        print(f"Performed inference in {infer_time:.3f} seconds \n", flush=True)

        if inf_proc.exitcode != 0:
            raise Exception("Inference failed!\n")
        
        # Launch data sorter component
        print(f"Launching sorting ...", flush=True)
        tic = perf_counter()
        if iter == 0:
            model_list_dd.bput("max_sort_iter",-1)
            model_list_dd.bput('current_sort_iter', -1)
        random_number = int(0.1*top_candidate_number)
        print(f"Adding {random_number} random candidates to training", flush=True)
        if os.getenv("USE_MPI_SORT"):
            print("Using MPI sort",flush=True)
            max_sorter_procs = args.max_procs_per_node*node_counts["sorting"]
            sorter_proc = mp.Process(target=sort_dictionary_pg, 
                                     args=(data_dd,
                                           top_candidate_number,
                                           max_sorter_procs, 
                                           nodelist["sorting"],
                                           model_list_dd,
                                           random_number,
                                           ),
                                    )
            sorter_proc.start()
            sorter_proc.join()
        else:
            print("Using filter sort", flush=True)
            sorter_proc = mp.Process(target=sort_dictionary,
                                      args=(
                                            data_dd,
                                            top_candidate_number,
                                            model_list_dd,
                                            ),
                                      )
            sorter_proc.start()
            sorter_proc.join()
        if sorter_proc.exitcode != 0:
            raise Exception("Sorting failed\n")
        toc = perf_counter()
        sort_time = toc - tic
        print(f"Performed sorting of {num_keys} keys in {sort_time:.3f} seconds \n", flush=True)

        # Launch Docking Simulations
        print(f"Launched Docking Simulations", flush=True)
        tic = perf_counter()
        num_procs = (args.max_procs_per_node - len(dd_cpu_bind)*2) * node_counts["simulation"]
        num_procs = min(num_procs, top_candidate_number//4)
        dock_proc = mp.Process(
            target=launch_docking_sim,
            args=(model_list_dd, sim_dd, iter, num_procs, nodelist["simulation"]),
        )
        dock_proc.start()
        dock_proc.join()
        toc = perf_counter()
        dock_time = toc - tic
        #os.rename("finished_run_docking.log", f"finished_run_docking_{iter}.log")
        print(f"Performed docking in {dock_time:.3f} seconds \n", flush=True)
        
    #     print(f"Candidate Dictionary stats:", flush=True)
    #     print(cand_dd.stats)
        if dock_proc.exitcode != 0:
            raise Exception("Docking sims failed\n")

        # Launch Training
        print(f"Launched Fine Tune Training", flush=True)
        tic = perf_counter()
        BATCH = 64
        EPOCH = 150
        train_proc = mp.Process(
            target=launch_training,
            args=(
                model_list_dd,
                sim_dd,
                nodelist["training"][0],  # training is always 1 node
                BATCH,
                EPOCH,
            ),
        )
        train_proc.start()
        train_proc.join()
        toc = perf_counter()
        train_time = toc - tic
        print(f"Performed training in {train_time} seconds \n", flush=True)
        if train_proc.exitcode != 0:
            raise Exception("Training failed\n")
        iter_end = perf_counter()
        iter_time = iter_end - iter_start
        print(
            f"Performed iter {iter} in {iter_time} seconds \n", flush=True
        )
        with open("driver_times.log", "a") as f:
            f.write(f"{iter}  {infer_time}  {sort_time}  {dock_time}  {train_time}\n")

        #tic = perf_counter()
        #output_sims(model_list_dd, iter=iter)
        #toc = perf_counter()
        #print(f"Output candidates in {toc -tic} seconds",flush=True)
    
        model_list_dd.checkpoint()
        iter += 1


    # # Close the dictionary
    # print("Closing the Dragon Dictionary and exiting ...", flush=True)
    # # cand_dd.destroy()
    # data_dd.destroy()
    # end_time = perf_counter()
    # print(f"Total time {end_time - start_time} seconds", flush=True)
