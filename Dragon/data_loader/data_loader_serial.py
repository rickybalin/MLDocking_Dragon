from pathlib import Path
import gzip
from time import perf_counter
import argparse

def raw_data_loader(data_path: str) -> dict:
    """Load raw inference data from files

    :param data_path: file path to location of raw data
    :type data_path: str
    :return: dictionary containing the smile strings of all the inference compounds
    :rtype: dict
    """
    base_p = Path(data_path)
    data_dict = {}

    # Loop over sub-directories of compunt sizes
    for sub_dir in sorted(base_p.iterdir()):
        if sub_dir.is_dir():
            sub_data_dict = {}
            sub_dir_name = str(sub_dir).split("/")[-1]

            smi_files = sub_dir.glob("**/*.smi")
            gz_files = sub_dir.glob("**/*.smi.gz")
            
            # Loop over all compunds of specific type and accumulate smiles
            # smi files first
            smi_file_list = []
            for file in sorted(smi_files):
                fname = str(file).split("/")[-1].split(".")[0]
                smi_file_list.append(fname)
                smiles = []
                with file.open() as f:
                    for line in f:
                        smile = line.split("\t")[0]
                        smiles.append(smile)
                sub_data_dict[fname] = smiles

            # smi.gz files next
            gz_file_list = []
            for file in sorted(gz_files):
                fname = str(file).split("/")[-1].split(".")[0]
                if fname not in smi_file_list:
                    smiles = []
                    gz_file_list.append(fname)
                    with gzip.open(str(file), 'rt') as f:
                        for line in f:
                            smile = line.split("\t")[0]
                            smiles.append(smile)
                    sub_data_dict[fname] = smiles

            data_dict[sub_dir_name] = sub_data_dict

    return data_dict


if __name__ == "__main__":
    # Import command line arguments
    parser = argparse.ArgumentParser(description='SMILES Data Loader')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC_22/zinc-22-2d/2d-small",
                        help='Path to SMILES strings to load')
    args = parser.parse_args()

    print("Loading inference data from files")
    print(f"at path {args.data_path} ...")
    tic = perf_counter()
    data = raw_data_loader(args.data_path)
    toc = perf_counter()
    load_time = toc - tic
    print(f"Done. Loaded inference data from files in {load_time:.3f} seconds \n", flush=True)

    base_dir = "/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/validation"
    case = args.data_path.split("/")[-1]
    fname = case + "_serial.txt"
    fname = base_dir + "/" + fname
    with open(fname, "w") as f:
        for key, val in data.items():
            for keyy, vall in val.items():
                f.write(f"{keyy}: {len(vall)}\n")
    print("Printed data to file for validation")
