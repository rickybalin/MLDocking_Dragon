from time import perf_counter
import numpy as np
import dragon
import argparse
import multiprocessing as mp
from dragon.data.distdictionary.dragon_dict import DragonDict

def load_inference_data(_dict, n_samples):
    """Load some synthetic inference data into the Dragon dictionary
    """
    x = np.linspace(0, PI/2, n_samples, dtype=np.float32)
    y = np.sin(x)
    test_data = np.vstack((x,y))
    _dict["test_data"] = test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes the dictionary distributed across')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--total_mem_size', type=int, default=1,
                        help='total managed memory size for dictionary in GB')

    my_args = parser.parse_args()
    mp.set_start_method("dragon")

    # Instantiate the dictionary and start the processes
    total_mem_size = my_args.total_mem_size * (1024*1024*1024)
    dd = DragonDict(my_args.managers_per_node, my_args.num_nodes, total_mem_size)
    print("Launched Dragon Dictionary \n", flush=True)

    # Launch the data loader component
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(target=load_inference_data, args=(dd,my_args.n_samples))
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    data_size = my_args.n_samples*2*4/1024
    print(f"Loaded {data_size}KB in {toc-tic} seconds \n", flush=True)

    # Close the dictionary
    print("Done here. Closing the Dragon Dictionary", flush=True)
    dd.close()
