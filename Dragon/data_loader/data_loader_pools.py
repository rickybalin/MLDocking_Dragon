import pathlib
import gzip
from time import perf_counter
from typing import Optional, Tuple
import argparse
import filecmp

import dragon
import multiprocessing as mp

def get_files(base_p: pathlib.PosixPath) -> list:
    """Return the file paths in all sub-directories

    :param base_p: file path to location of raw data
    :type base_p: pathlib.PosixPath
    :return: list of file paths
    :rtype: list
    """
    files = []
    file_count = 0
    for sub_dir in sorted(base_p.iterdir()):
        if sub_dir.is_dir():
            smi_files = sub_dir.glob("**/*.smi")
            gz_files = sub_dir.glob("**/*.smi.gz")
            smi_file_list = []
            for file in sorted(smi_files):
                fname = str(file).split("/")[-1].split(".")[0]
                smi_file_list.append(fname)
                files.append(file)
                file_count += 1
            for file in sorted(gz_files):
                if fname not in smi_file_list:
                    files.append(file)
                    file_count += 1
    return files

def read_smiles(file_path: pathlib.PosixPath) -> list:
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
    return smiles

def read_subdir(sub_dir_p: pathlib.PosixPath) -> dict:
    """Read the files from a sub directory

    :param sub_dir_p: path to sub-directory
    :type sub_dir_p: pathlib.PosixPath
    :return: dictionary of file names and smiles strings
    :rtype: dict
    """
    sub_data_dict = {}
    smi_files = sub_dir_p.glob("**/*.smi")
    gz_files = sub_dir_p.glob("**/*.smi.gz")
            
    # Loop over all compunds of specific type and accumulate smiles
    # smi files first
    smi_file_list = []
    for file in sorted(smi_files):
        fname = str(file).split("/")[-1].split(".")[0]
        smi_file_list.append(fname)
        sub_data_dict[fname] = read_smiles(file)

    # smi.gz files next
    gz_file_list = []
    for file in sorted(gz_files):
        fname = str(file).split("/")[-1].split(".")[0]
        if fname not in smi_file_list:
            gz_file_list.append(fname)
            sub_data_dict[fname] = read_smiles(file)
    return sub_data_dict

def read_subdir_mp(sub_dir_p: pathlib.PosixPath) -> Tuple[list, list]:
    """Read the files from a sub directory with multiprocessing

    :param sub_dir_p: path to sub-directory
    :type sub_dir_p: pathlib.PosixPath
    :return: tuple with list of smiles strings and names
    :rtype: tuple
    """
    mp.set_start_method("dragon", force=True)
    sub_data_list = []
    smi_files = sub_dir_p.glob("**/*.smi")
    gz_files = sub_dir_p.glob("**/*.smi.gz")
            
    # Accumulate files to read
    file_list = []
    file_name_list = []
    for file in sorted(smi_files):
        file_list.append(file)
        fname = str(file).split("/")[-1].split(".")[0]
        file_name_list.append(fname)
    for file in sorted(gz_files):
        fname = str(file).split("/")[-1].split(".")[0]
        if fname not in file_name_list:
            file_list.append(file)
            file_name_list.append(fname)

    # Launch processes for parallel loading
    #num_procs = len(file_list)
    num_procs = 5
    pool = mp.Pool(num_procs)
    sub_data_list = pool.map(read_smiles, file_list)
    pool.close()
    pool.join()
    return file_name_list, sub_data_list
    

def raw_data_loader_mp(data_path: str, granularity: str, max_procs: int) -> Tuple[list, list]:
    """Load raw inference data from files

    :param data_path: file path to location of raw data
    :type data_path: str
    :param granularity: granularity used to read files
    :type granularity: str
    :return: list containing the smile strings of all the inference compounds
    :rtype: list
    """
    base_p = pathlib.Path(data_path)
    file_list = []
    data_list = []

    # Determine granularity for data loading
    if granularity=="directory" or granularity=="directory_file":
        sub_dirs = sorted([str(sub_dir).split("/")[-1] for sub_dir in base_p.iterdir() if sub_dir.is_dir()])
        sub_dir_paths = [base_p / sub_dirs[ip] for ip in range(len(sub_dirs))]
        num_procs = min(max_procs, len(sub_dirs))

        # Create and launch a Pool
        #print(f"\nLaunching a Pool with {num_procs} sub-processes ... ")
        pool = mp.Pool(num_procs)
        if granularity=="directory":
            data_list = pool.map(read_subdir, sub_dir_paths)
        elif granularity=="directory_file":
            result = pool.map(read_subdir_mp, sub_dir_paths)
            for files, data in result:
                file_list.extend(files)
                data_list.extend(data)
        pool.close()
        pool.join()
        #print("Done \n", flush=True)
    elif granularity=="file":
        files = get_files(base_p)
        file_list = [str(file).split("/")[-1].split(".")[0] for file in files]
        num_procs = min(max_procs, len(files))

         # Create and launch a Pool
        #print(f"\nLaunching a Pool with {num_procs} sub-processes ... ")
        pool = mp.Pool(num_procs)
        data_list = pool.map(read_smiles, files)
        pool.close()
        pool.join()
        #print("Done \n", flush=True)

    return file_list, data_list


if __name__ == "__main__":
    # Import command line arguments
    parser = argparse.ArgumentParser(description='SMILES Data Loader')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC_22/zinc-22-2d/2d-small",
                        help='Path to SMILES strings to load')
    parser.add_argument('--granularity', type=str, default="directory",
                        help='Granularity used to load data (directory,file,directory_file)')
    parser.add_argument('--mp_launch', type=str, default="spawn",
                        help='Backend for multiprocessing (dragon,spawn)')
    parser.add_argument('--max_procs', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--validate', type=str, default="no",
                        help='Validate the data loader with the serial case (yes,no)')
    args = parser.parse_args()

    # Start multiprocessing
    if args.mp_launch=="dragon":
        mp.set_start_method("dragon")
    else:
        mp.set_start_method("spawn")

    # Launch and time data loader
    print("Loading inference data from files")
    print(f"at path {args.data_path}")
    print(f"with granularity: {args.granularity}")
    print(f"with {args.mp_launch} ...", flush=True)
    tic = perf_counter()
    files, data = raw_data_loader_mp(args.data_path, args.granularity, args.max_procs)
    toc = perf_counter()
    load_time = toc - tic
    print(f"Loaded inference data in {load_time:.3f} seconds \n", flush=True)

    if args.validate!="no":
        base_dir = "/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/validation"
        case = args.data_path.split("/")[-1]
        mp_case = f"pool_{args.mp_launch}_{args.granularity}.txt"
        fname = base_dir + "/" + case + "_" + mp_case
        with open(fname, "w") as f:
            if args.granularity=="directory":
                for item in data:
                    for key, val in item.items():
                        f.write(f"{key}: {len(val)}\n")
            else:
                for name, item in zip(files,data):
                    f.write(f"{name}: {len(item)}\n")
        
        serial_fname = base_dir + "/" + case + "_serial.txt"
        if filecmp.cmp(serial_fname, fname):
            print("MP data validated!\n")
        else:
            print("WARNING: MP data NOT validated!\n")
  

