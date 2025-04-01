def max_data_dict_size(num_keys, 
                       model_size=33, 
                       smiles_key_val_size=14.6, 
                       canidate_sim_size_per_iter=1.5, 
                       max_pool_frac=0.8):
    
    # Two sources of data in data dictionary
    # Smiles data: approx 14.6 MB per file
    # Trained model: approximately 33 MB
    # Data needed is 33 MB + num_keys*14.6
    min_data_req = model_size + smiles_key_val_size*num_keys

    # Assume we want to store 10 top cand lists and associated simulation results
    min_cand_dict_size = 10.*canidate_sim_size_per_iter

    # Assume you need 20 per cent overhead in data dictionary
    data_dict_size = min_data_req/max_pool_frac
    cand_dict_size = min_cand_dict_size/max_pool_frac

    cand_dict_size /= 1024
    data_dict_size /= 1024

    cand_dict_size = max(cand_dict_size, 1)
    data_dict_size = max(data_dict_size, 10)

    return int(data_dict_size), int(cand_dict_size)


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

