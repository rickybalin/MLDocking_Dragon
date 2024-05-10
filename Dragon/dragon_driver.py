from time import perf_counter
import numpy as np
import argparse

import dragon
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict
#from dragon.data.distdictionary.dragon_dict import DragonDict

from data_loader.data_loader_presorted import load_inference_data
from inference.launch_inference import launch_inference_mpi

if __name__ == "__main__":
    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes the dictionary distributed across')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--total_mem_size', type=int, default=16,
                        help='total managed memory size for dictionary in GB')
    parser.add_argument('--max_procs', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    args = parser.parse_args()

    # Start distributed dictionary
    mp.set_start_method("dragon")
    total_mem_size = args.total_mem_size * (1024*1024*1024)
    dd = DDict(args.managers_per_node, args.num_nodes, total_mem_size)
    print("Launched Dragon Dictionary \n", flush=True)

    # Launch the data loader component
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(target=load_inference_data, args=(dd,args.data_path,args.max_procs))
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    load_time = toc - tic
    print(f"Loaded inference data in {load_time:.3f} seconds \n", flush=True)

    # Launch the data inference component
    num_ranks = 4
    print("Launching inference with 4 ranks ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(target=launch_inference_mpi, args=(dd,num_ranks))
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    infer_time = toc - tic
    print(f"Performed inference in {infer_time:.3f} seconds \n", flush=True)

    # Close the dictionary
    print("Closing the Dragon Dictionary and exiting ...", flush=True)
    dd.destroy()
