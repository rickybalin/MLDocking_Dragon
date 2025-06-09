import os
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
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes')
    parser.add_argument('--sort_procs_per_node', type=int, default=1,
                        help='Number of processes per node for MPI sorting')
    parser.add_argument('--max_iter', type=int, default=1,
                        help='Maximum number of iterations')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    parser.add_argument('--logging', type=str, default="info",
                        help='Logging level')
    parser.add_argument('--inference_and_sort', type=str, default="False", choices=["False", "True"],
                        help='Perform inference and sorting only')
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
        print(f"Component {key} running on {[Node(node).hostname for node in val]} nodes",flush=True)

    # Get info about gpus and cpus
    gpu_devices = os.getenv("GPU_DEVICES")
    if gpu_devices is not None:
        gpu_devices = gpu_devices.split(",")
        num_gpus = len(gpu_devices)
    else:
        num_gpus = 0
    
    # Number of top candidates to produce
    if num_tot_nodes <= 3:
        top_candidate_number = 1000
    else:
        top_candidate_number = 10000

    print(f"\nFinished workflow setup in {(perf_counter()-start_time):.3f} seconds\n", flush=True)

    # Start sequential loop
    max_iter = args.max_iter
    iter = 0
    with open("driver_times.log", "w") as f:
        f.write(f"# iter  infer_time  sort_time  dock_time  train_time \n")
    while iter < max_iter:
        print(f"\n*** Start loop iter {iter} ***")
        iter_start = perf_counter()

        # Launch the data inference component
        print(f"Launching inference ...", flush=True)
        tic = perf_counter()
        if num_tot_nodes <= 3:
            inf_num_limit = 24
        else:
            inf_num_limit = None
        inf_proc = mp.Process(
            target=launch_inference,
            args=(args.data_path,
                nodelist["inference"],
            ),
            kwargs={
            'inf_num_limit': inf_num_limit,
            }
        )
        inf_proc.start()
        inf_proc.join()
        toc = perf_counter()
        infer_time = toc - tic
        print(f"Executed inference mp.Process in {infer_time:.3f} seconds \n", flush=True)
        if inf_proc.exitcode != 0:
            raise Exception("Inference failed!\n")
        
        # Launch data sorter component
        print(f"Launching sorting ...", flush=True)
        tic = perf_counter()
        print("Using MPI sort",flush=True)
        max_sorter_procs = (args.sort_procs_per_node) * node_counts["sorting"]
        sorter_proc = mp.Process(target=sort_dictionary_pg, 
                                    args=(
                                        top_candidate_number,
                                        max_sorter_procs, 
                                        nodelist["sorting"],
                                        ),
                                )
        sorter_proc.start()
        sorter_proc.join()
        if sorter_proc.exitcode != 0:
            raise Exception("Sorting failed\n")
        toc = perf_counter()
        sort_time = toc - tic
        print(f"Executed sorting mp.Process in {sort_time:.3f} seconds \n", flush=True)
        if args.inference_and_sort == "True":
            sys.exit()

        # Launch Docking Simulations
        print(f"Launched docking simulations ...", flush=True)
        tic = perf_counter()
        max_num_procs = top_candidate_number//4
        dock_proc = mp.Process(
            target=launch_docking_sim,
            args=( 
                1, 
                max_num_procs, 
                nodelist["simulation"]),
        )
        dock_proc.start()
        dock_proc.join()
        if dock_proc.exitcode != 0:
            raise Exception("Docking sims failed\n")
        toc = perf_counter()
        dock_time = toc - tic
        print(f"Executed docking mp.Process in {dock_time:.3f} seconds \n", flush=True)

        # Launch Training
        print(f"Launched Fine Tune Training", flush=True)
        tic = perf_counter()
        BATCH = 64
        EPOCH = 150
        train_proc = mp.Process(
            target=launch_training,
            args=(
                nodelist["training"][0],  # training is always 1 node
                BATCH,
                EPOCH,
            ),
        )
        train_proc.start()
        train_proc.join()
        toc = perf_counter()
        train_time = toc - tic
        print(f"Executed training mp.Process in {train_time} seconds \n", flush=True)
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
    
        iter += 1


    end_time = perf_counter()
    print(f"Total time {end_time - start_time} seconds", flush=True)
