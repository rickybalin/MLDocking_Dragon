from dragon.data.ddict import DDict
from dragon.native.machine import System, Node


def output_sims(cdd: DDict):

    keys = cdd.keys()
    last_list_key = cdd["max_sort_iter"]
    candidate_list = cdd[last_list_key]
    top_smiles = candidate_list['smiles']
    top_inf = candidate_list['inf']
    with open(f'top_candidates_{last_list_key}.out','w') as f:
        f.write("# smiles  inference_score  docking_scores\n")
        for i in range(len(top_smiles)):
            smiles = top_smiles[i]
            inference_score = top_inf[i]
            try:
                docking_score = cdd[smiles]
                f.write(f"{smiles}    {inference_score}    {docking_score}\n")
            except:
                print(f"Missing smiles {smiles} from candidate dict")
                f.write(f"{smiles}    {inference_score}    0\n")

def max_data_dict_size(num_keys: int, 
                       model_size=33, 
                       smiles_key_val_size=14.6, 
                       canidate_sim_size_per_iter=1.5, 
                       max_pool_frac=0.8):

    # Get information about the allocation
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    
    # Two sources of data in data dictionary
    # Smiles data: approx 14.6 MB per file
    # Trained model: approximately 33 MB, broadcast one copy to each node
    # Data needed is 33 MB*num_nodes + num_keys*14.6
    min_data_req = model_size*num_tot_nodes + smiles_key_val_size*num_keys

    # Assume we want to store 10 top cand lists and associated simulation results
    min_cand_dict_size = 10.*canidate_sim_size_per_iter

    # Assume you need 20 per cent overhead in data dictionary
    data_dict_size = min_data_req/(max_pool_frac)
    cand_dict_size = min_cand_dict_size/(max_pool_frac)

    # Convert from MB to GB
    cand_dict_size /= 1024
    data_dict_size /= 1024

    # Ensure there is a minimum of 1 GB per node
    cand_dict_size = max(cand_dict_size, num_tot_nodes)
    data_dict_size = max(data_dict_size, num_tot_nodes)

    max_mem = ddict_mem_check()

    print(f"Memory available for ddicts: {max_mem} GB")

    if cand_dict_size + data_dict_size > max_mem:
        raise Exception(f"Not enough mem for dictionaries: {max_mem=} {data_dict_size=} {cand_dict_size=}")

    return int(data_dict_size), int(cand_dict_size)


def ddict_mem_check(mem_fraction=0.5):

    # let's place the DDict across all nodes Dragon is running on
    my_system = System()

    total_mem = 0
    for huid in my_system.nodes:
        anode = Node(huid)
        total_mem += anode.physical_mem
    dict_mem = mem_fraction * total_mem  # use fraction of the mem

    # Convert dict_mem to GB
    dict_mem /= 1024*1024*1024

    return int(dict_mem)

def get_prime_number(n: int):
    # Use 1, 2 or 3 if that is n
    if n <= 3:
        return n
    
    # All prime numbers are odd except two
    if not (n & 1):
        n -= 1
    
    for i in range(n, 3, -2):
        isprime = True
        for j in range(3,n):
            if i % j == 0:
                isprime = False
                break
        if isprime:
            return i
    return 3

