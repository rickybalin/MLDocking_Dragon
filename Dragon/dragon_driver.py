from time import perf_counter
import numpy as np
import dragon
import argparse
import multiprocessing as mp
from dragon.data.distdictionary.dragon_dict import DragonDict

from data_loader.data_loader_presorted import load_inference_data

if __name__ == "__main__":
    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes the dictionary distributed across')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--total_mem_size', type=int, default=1,
                        help='total managed memory size for dictionary in GB')
    parser.add_argument('--max_procs', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    args = parser.parse_args()

    # Start distributed dictionary
    mp.set_start_method("dragon")
    total_mem_size = args.total_mem_size * (1024*1024*1024)
    # total_mem_size is total across all nodes or on each node?
    # what happens when we run out of memory?
    # is the memory pre-allocated?
    # how close can we get to SSD memory on node?
    dd = DragonDict(args.managers_per_node, args.num_nodes, total_mem_size)
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

    # Close the dictionary
    print("Closing the Dragon Dictionary and exiting ...", flush=True)
    dd.close()
