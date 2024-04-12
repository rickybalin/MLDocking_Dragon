import pathlib
import gzip
from time import perf_counter
from typing import Tuple
import argparse
import time

import dragon
import multiprocessing as mp
from dragon.data.distdictionary.dragon_dict import DragonDict

global data_dict 
data_dict = None

def init_worker(q):
    global data_dict
    data_dict = q.get()
    return

def get_files(base_p: pathlib.PosixPath) -> Tuple[list, int]:
    """Return the file paths

    :param base_p: file path to location of raw data
    :type base_p: pathlib.PosixPath
    :return: tuple with list of file paths and number of files
    :rtype: tuple
    """
    files = []
    file_count = 0
    
    smi_files = base_p.glob("**/*.smi")
    gz_files = base_p.glob("**/*.gz")
    smi_file_list = []
    for file in sorted(smi_files):
        fname = str(file).split("/")[-1].split(".")[0]
        smi_file_list.append(fname)
        files.append(file)
        file_count += 1
    for file in sorted(gz_files):
        fname = str(file).split("/")[-1].split(".")[0]
        if fname not in smi_file_list:
            files.append(file)
            file_count += 1
    return files, file_count

def read_smiles(file_path: pathlib.PosixPath):
    """Read the smile strings from file

    :param file_path: file path to open
    :type file_path: pathlib.PosixPath
    """
    global data_dict

    smiles = []
    f_name = str(file_path).split("/")[-1]
    f_extension = str(file_path).split("/")[-1].split(".")[-1]
    if f_extension=="smi":
        with file_path.open() as f:
            for line in f:
                smile = line.split("\t")[0]
                smiles.append(smile)
    elif f_extension=="gz":
        with gzip.open(str(file_path), 'rt') as f:
            for line in f:
                smile = line.split("\t")[0]
                smiles.append(smile)
    
    data_dict[f_name] = smiles

    
def load_inference_data(_dict, data_path: str, max_procs: int):
    """Load pre-sorted inference data from files and to Dragon dictionary

    :param _dict: Dragon distributed dictionary
    :type _dict: ...
    :param data_path: path to pre-sorted data
    :type data_path: str
    :param max_procs: maximum number of processes to launch for loading
    :type max_procs: int
    """
    # Get list of files to read
    base_path = pathlib.Path(data_path)
    files, num_files = get_files(base_path)
    print(f"{num_files=}", flush=True)

    num_procs = min(max_procs, num_files)
    print(f"Number of pool procs is {num_procs}",flush=True)
    
    # Launch Pool
    initq = mp.Queue(maxsize=num_procs)
    for _ in range(num_procs):
        initq.put(_dict)
        
    pool = mp.Pool(num_procs, initializer=init_worker, initargs=(initq,))
    print(f"Pool initialized", flush=True)
    pool.map(read_smiles, files)
    print(f"Mapped function complete", flush=True)
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes the dictionary distributed across')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--total_mem_size', type=int, default=8,
                        help='total managed memory size for dictionary in GB')
    parser.add_argument('--max_procs', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')
    args = parser.parse_args()

    # Start distributed dictionary
    mp.set_start_method("dragon")
    total_mem_size = args.total_mem_size * (1024*1024*1024)
    dd = DragonDict(args.managers_per_node, args.num_nodes, total_mem_size)
    print("Launched Dragon Dictionary \n", flush=True)

    # Launch the data loader
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    load_inference_data(dd, args.data_path, args.max_procs)
    toc = perf_counter()
    load_time = toc - tic
    print(f"Loaded inference data in {load_time:.3f} seconds \n", flush=True)

    # Close the dictionary
    print("Done, closing the Dragon Dictionary", flush=True)
    dd.close()
