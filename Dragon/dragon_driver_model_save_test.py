import os
#import logging
#logger = logging.getLogger(__name__)
from time import perf_counter
import argparse
import pathlib
import dragon
import multiprocessing as mp
from dragon.data.ddict import DDict
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy

from data_loader.data_loader_presorted import load_inference_data
from inference.launch_inference import launch_inference

from data_loader.data_loader_presorted import get_files
from inference.utils_transformer import ParamsJson, ModelArchitecture, pad
from inference.utils_encoder import SMILES_SPE_Tokenizer
from training.ST_funcs.clr_callback import *
from training.ST_funcs.smiles_regress_transformer_funcs import *


driver_path = os.getenv("DRIVER_PATH")

if __name__ == "__main__":

    # Import command line arguments
    parser = argparse.ArgumentParser(description='Distributed dictionary example')
    parser.add_argument('--data_dictionary_mem_fraction', type=float, default=0.7,
                        help='fraction of memory dedicated to data dictionary')
    parser.add_argument('--managers_per_node', type=int, default=1,
                        help='number of managers per node for the dragon dict')
    parser.add_argument('--mem_per_node', type=int, default=8,
                        help='managed memory size per node for dictionary in GB')
    parser.add_argument('--max_procs_per_node', type=int, default=10,
                        help='Maximum number of processes in a Pool')
    parser.add_argument('--max_iter', type=int, default=10,
                        help='Maximum number of iterations')
    parser.add_argument('--dictionary_timeout', type=int, default=10,
                        help='Timeout for Dictionary in seconds')
    parser.add_argument('--data_path', type=str, default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
                        help='Path to pre-sorted SMILES strings to load')

    args = parser.parse_args()
    #
    #logging.basicConfig(level=logging.INFO)
    #logger.info("Begun dragon driver")
    # Start driver
    start_time = perf_counter()
    print("Begun dragon driver",flush=True)
    print(f"Reading inference data from path: {args.data_path}",flush=True)
    mp.set_start_method("dragon")

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    tot_nodelist = alloc.nodes

    tot_mem = args.mem_per_node * num_tot_nodes

    # Get info about gpus and cpus
    gpu_devices = os.getenv("GPU_DEVICES").split(",")
    num_gpus = len(gpu_devices)

    # for this sequential loop test set inference and docking to all the nodes and sorting and training to one node
    node_counts = {
        "sorting": num_tot_nodes,
        "training": 1,
        "inference": num_tot_nodes,
        "docking": num_tot_nodes,
    }

    nodelists = {}
    offset = 0
    for key in node_counts.keys():
        nodelists[key] = tot_nodelist[:node_counts[key]]

    # Use a prime number of nodes for dictionaries
    num_dict_nodes = num_tot_nodes #get_prime_number(num_tot_nodes)

    # Get info on the number of files
    base_path = pathlib.Path(args.data_path)
    files, num_files = get_files(base_path)

    # mem_per_file = 12/8192
    # tot_mem = int(min(args.mem_per_node*num_tot_nodes,
    #               max(ceil(num_files*mem_per_file*100/args.data_dictionary_mem_fraction),2*num_tot_nodes)
    #               ))
    tot_mem = args.mem_per_node*num_tot_nodes
    print(f"There are {num_files} files, setting mem_per_node to {tot_mem/num_dict_nodes}")

    # Set up and launch the inference data DDict and top candidate DDict
    data_dict_mem = max(int(tot_mem), num_tot_nodes)
    candidate_dict_mem = max(int(tot_mem*(1.-args.data_dictionary_mem_fraction)), num_tot_nodes)
    print(f"Setting data_dict size to {data_dict_mem} GB and candidate_dict size to {candidate_dict_mem} GB")
    data_dict_mem *= (1024*1024*1024)
    candidate_dict_mem *= (1024*1024*1024)

    # Start distributed dictionary used for inference
    # inf_dd_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=Node(inf_dd_nodelist).hostname)
    # Note: the host name based policy, as far as I can tell, only takes in a single node, not a list
    #       so at the moment we can't specify to the inf_dd to run on a list of nodes.
    #       But by setting inf_dd_nodes < num_tot_nodes, we can make it run on the first inf_dd_nodes nodes only
   
    data_dd = DDict(args.managers_per_node, num_dict_nodes, data_dict_mem, trace=True)
    print(f"Launched Dragon Dictionary for inference with total memory size {data_dict_mem}", flush=True)
    print(f"on {num_dict_nodes} nodes", flush=True)
    print(f"{data_dd.stats=}")
    
    # Launch the data loader component
    max_procs = args.max_procs_per_node * num_tot_nodes
    print("Loading inference data into Dragon Dictionary ...", flush=True)
    tic = perf_counter()
    loader_proc = mp.Process(
        target=load_inference_data,
        args=(
            data_dd,
            args.data_path,
            max_procs,
            num_tot_nodes * args.managers_per_node,
        ),
    )
    loader_proc.start()
    loader_proc.join()
    toc = perf_counter()
    load_time = toc - tic
    if loader_proc.exitcode == 0:
        print(f"Loaded inference data in {load_time:.3f} seconds", flush=True)
    else:
        raise Exception(f"Data loading failed with exception {loader_proc.exitcode}")

    tic = perf_counter()
    print("Here are the stats after data loading...")
    print("++++++++++++++++++++++++++++++++++++++++")
    print(data_dd.stats)
    toc = perf_counter()
    load_time = toc - tic
    print(f"Retrieved dictionary stats in {load_time:.3f} seconds", flush=True)
    num_keys = len(data_dd.keys())
    
    cand_dd = DDict(args.managers_per_node, num_dict_nodes, candidate_dict_mem, policy=None, trace=True)
    print(f"Launched Dragon Dictionary for top candidates with total memory size {candidate_dict_mem}", flush=True)
    print(f"on {num_dict_nodes} nodes", flush=True)
    
    # Number of top candidates to produce
    if num_tot_nodes < 3:
        top_candidate_number = 1000
    else:
        top_candidate_number = 5000

    max_iter = args.max_iter
    iter = 0

    # Load model from disk and save to dictionary
    # Read HyperParameters
    print("Loading pre-trained model from disk", flush=True)
    json_file = driver_path + "inference/config.json"
    hyper_params = ParamsJson(json_file)

    # Load model and weights
    model = ModelArchitecture(hyper_params).call()
    model.load_weights(
        driver_path + f"inference/smile_regress.autosave.model.h5"
    )
    print("Loaded pretrained model", flush=True)
    
    # Iterate over the layers and their respective weights
    num_layers = 0
    num_weights = 0
    tot_memory = 0
    weight_keys = []
    for layer_idx, layer in enumerate(model.layers):
        num_layers += 1
        for weight_idx, weight in enumerate(layer.get_weights()):
            num_weights += 1
            # Create a key for each weight
            wkey = f'model_layer_{layer_idx}_weight_{weight_idx}'
            # Save the weight in the dictionary
            #weights_dict[wkey] = weight
            data_dd[wkey] = weight
            weight_keys.append(wkey)
            print(f"{wkey}: {weight.nbytes} bytes")
            tot_memory += weight.nbytes
    print(f"{num_layers=} {num_weights=} {tot_memory=}")
    print(f"saved keys: {weight_keys}")
    data_dd["model_weight_keys"] = weight_keys
    data_dd["model_iter"] = 1
    data_dd["model_hyper_params"] = hyper_params

    # Launch inference
    num_procs = num_gpus*node_counts["inference"]

    print(f"Launching inference with {num_procs} processes ...", flush=True)
    if num_tot_nodes < 3:
        inf_num_limit = 8
        print(
            f"Running small test on {num_tot_nodes}; limiting {inf_num_limit} keys per inference worker"
        )
    else:
        inf_num_limit = None

    tic = perf_counter()
    inf_proc = mp.Process(
        target=launch_inference,
        args=(
            data_dd,
            nodelists["inference"],
            num_procs,
            inf_num_limit,
        ),
    )
    inf_proc.start()
    inf_proc.join()
    toc = perf_counter()
    infer_time = toc - tic
    print(f"Performed inference in {infer_time:.3f} seconds \n", flush=True)
