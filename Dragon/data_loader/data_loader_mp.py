import pathlib
import gzip
from time import perf_counter
from typing import Optional
import argparse

import dragon
import multiprocessing as mp
from multiprocessing.queues import Queue

def get_files(base_p: pathlib.PosixPath) -> list:
    """Count the number of files in all sub-directories

    :param base_p: file path to location of raw data
    :type base_p: pathlib.PosixPath
    :return: total number of files
    :rtype: int
    """
    files = []
    file_count = 0
    for sub_dir in base_p.iterdir():
        if sub_dir.is_dir():
            smi_files = sub_dir.glob("**/*.smi")
            gz_files = sub_dir.glob("**/*.smi.gz")
            smi_file_list = []
            for file in smi_files:
                fname = str(file).split("/")[-1].split(".")[0]
                smi_file_list.append(fname)
                files.append(file)
                file_count += 1
            for file in gz_files:
                if fname not in smi_file_list:
                    files.append(file)
                    file_count += 1
    return files

def read_smiles(file_path: pathlib.PosixPath, q: Optional[Queue] = None) -> list:
    """Read the smile strings from file

    :param file_path: file path to open
    :type file_path: pathlib.PosixPath
    :return: list of smiles strings
    :rtype: list
    """
    smiles = []
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
    if q is not None:
        q.put(smiles)
    else:
        return smiles

def read_subdir(sub_dir_p: pathlib.PosixPath, q: Queue) -> None:
    """Read the files from a sub directory

    :param sub_dir_p: path to sub-directory
    :type sub_dir_p: pathlib.PosixPath
    :param q: Queue to put data in
    :type q: mp.queues.Queue
    """
    sub_data_dict = {}
    smi_files = sub_dir_p.glob("**/*.smi")
    gz_files = sub_dir_p.glob("**/*.smi.gz")
            
    # Loop over all compunds of specific type and accumulate smiles
    # smi files first
    smi_file_list = []
    for file in smi_files:
        fname = str(file).split("/")[-1].split(".")[0]
        smi_file_list.append(fname)
        sub_data_dict[fname] = read_smiles(file)

    # smi.gz files next
    gz_file_list = []
    for file in gz_files:
        fname = str(file).split("/")[-1].split(".")[0]
        if fname not in smi_file_list:
            gz_file_list.append(fname)
            sub_data_dict[fname] = read_smiles(file)
    
    # if process pool, do return here
    q.put(sub_data_dict)


def raw_data_loader_mp(data_path: str, granularity: str) -> dict:
    """Load raw inference data from files and load to Dragon dictionary

    :param data_path: file path to location of raw data
    :type data_path: str
    :return: dictionary containing the smile strings of all the inference compounds
    :rtype: dict
    """
    base_p = pathlib.Path(data_path)
    data_dict = {}

    # Determine granularity for data loading
    if granularity=="directory" or granularity=="directory_file":
        sub_dirs = [str(sub_dir).split("/")[-1] for sub_dir in base_p.iterdir() if sub_dir.is_dir()]
        num_procs = len(sub_dirs)

        # Launch processes for parallel loading
        print(f"\nLaunching {num_procs} sub-processes ... ")
        processes = []
        queues = []
        for ip in range(num_procs):
            sub_dir_p = base_p / sub_dirs[ip]
            q = mp.Queue()
            p = mp.Process(target=read_subdir, args=(sub_dir_p,q))
            processes.append(p)
            queues.append(q)

        for p in processes:
            p.start()

        for ip in range(num_procs):
            data_dict[sub_dirs[ip]] = queues[ip].get()
            processes[ip].join()
        print("Done \n", flush=True)

    elif granularity=="file":
        files = get_files(base_p)
        num_procs = len(files)

        # Launch processes for parallel loading
        print(f"\nLaunching {num_procs} sub-processes ... ")
        processes = []
        queues = []
        for ip in range(num_procs):
            q = mp.Queue()
            p = mp.Process(target=read_smiles, args=(files[ip],q))
            processes.append(p)
            queues.append(q)

        for p in processes:
            p.start()

        for ip in range(num_procs):
            fname = str(files[ip]).split("/")[-1].split(".")[0]
            data_dict[fname] = queues[ip].get()
            processes[ip].join()
        print("Done \n", flush=True)

    return data_dict


if __name__ == "__main__":
    # Import command line arguments
    parser = argparse.ArgumentParser(description='SMILES Data Loader')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC_22/zinc-22-2d/2d-small",
                        help='Path to SMILES strings to load')
    parser.add_argument('--granularity', type=str, default="directory",
                        help='Granularity used to load data (directory,file,directory_file)')
    parser.add_argument('--mp_launch', type=str, default="spawn",
                        help='Backend for multiprocessing (dragon,spawn)')
    args = parser.parse_args()

    # Start multiprocessing
    if args.mp_launch=="dragon":
        print("Using Dragon", flush=True)
        mp.set_start_method("dragon")
    else:
        print("Using multiprocessing", flush=True)
        mp.set_start_method("spawn")

    # Launch and time data loader
    print("Loading inference data from files ...")
    tic = perf_counter()
    data = raw_data_loader_mp(args.data_path, args.granularity)
    toc = perf_counter()
    load_time = toc - tic
    print(f"Done. Loaded inference data from files in {load_time:.3f} seconds \n")

    #print("Printing data dictionary for debugging:")
    #for key, val in data.items():
    #    print("sub dir: ", key)
    #    for keyy, vall in val.items():
    #        print(f"sub sub dir: {keyy} has {len(vall)} smiles")

