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
    parser.add_argument('--mem_per_node', type=int, default=1,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--dictionary_timeout', type=int, default=10,
                        help='Timeout for Dictionary in seconds')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    args = parser.parse_args()
    
    print("Begun dragon driver", flush=True)
    print(f"data_path:{args.data_path}", flush=True)
    print(f"num_nodes:{args.num_nodes}", flush=True)

    # Start distributed dictionary
    mp.set_start_method("dragon")

    # Set the total mem size to have a minimum of 1 GB per node
    total_mem_size = args.mem_per_node*args.num_nodes
    print(f"total_mem_size:{total_mem_size}", flush=True)
    total_mem_size*=(1024*1024*1024)
    
    print("Started Dragon Dictionary Launch", flush=True)
    sys.stdout.flush()
    dd = DDict(args.managers_per_node, args.num_nodes, total_mem_size,timeout=args.dictionary_timeout)
    print("Launched Dragon Dictionary", flush=True)

    # Launch the data loader component
    max_procs = args.max_procs_per_node*args.num_nodes
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(target=load_inference_data, args=(dd,args.data_path,max_procs,args.num_nodes*args.managers_per_node))
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    load_time = toc - tic
    print(f"Loaded inference data in {load_time:.3f} seconds \n", flush=True)

    # Launch the data inference component
    num_ranks = 4*args.num_nodes
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
