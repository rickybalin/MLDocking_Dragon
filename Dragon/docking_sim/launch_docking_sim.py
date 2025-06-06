import os
import csv
from time import perf_counter

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from .docking_openeye import run_docking

def read_output(stdout_conn: Connection) -> str:
    """Read stdout from the Dragon connection.

    :param stdout_conn: Dragon connection to rank 0's stdout
    :type stdout_conn: Connection
    :return: string with the output from stdout
    :rtype: str
    """
    output = ""
    try:
        # this is brute force
        while True:
            tmp = stdout_conn.recv()
            #print(tmp, flush=True)
            output += tmp
    except EOFError:
        pass
    finally:
        stdout_conn.close()
    return output


def launch_docking_sim(docking_iter, max_num_procs, nodelist):
    """Launch docking simulations

    :param cdd: Dragon distributed dictionary for top candidates
    :type dd: DDict
    :param num_procs: number of processes to use for docking
    :type num_procs: int
    """
    run_dir = os.getcwd()
    num_nodes = len(nodelist)

    available_cores = list(range(int(os.getenv("PROCS_PER_NODE"))))
    skip_cores = os.getenv("SKIP_THREADS").split(",")
    skip_cores = [int(c) for c in skip_cores]
    model_dd_cores = os.getenv("MODEL_DD_CPU_AFFINITY").split(",")
    model_dd_cores = [int(c) for c in model_dd_cores]
    sim_dd_cores = os.getenv("SIM_DD_CPU_AFFINITY").split(",")
    sim_dd_cores = [int(c) for c in sim_dd_cores]
    train_cores = os.getenv("TRAIN_CPU_AFFINITY").split(",")
    train_cores = [int(c) for c in train_cores]
    available_cores = [c for c in available_cores if c not in skip_cores]
    available_cores = [c for c in available_cores if c not in model_dd_cores]
    available_cores = [c for c in available_cores if c not in sim_dd_cores]
    available_cores = [c for c in available_cores if c not in train_cores]
    #print('Available cores for docking sim: ',available_cores,flush=True)
    num_procs_pn = len(available_cores)
    num_procs = num_procs_pn*num_nodes
    if num_procs > max_num_procs:
        num_procs = max_num_procs
        num_procs_pn = max_num_procs//num_nodes
    remainder_procs_pn = num_procs%num_nodes if num_procs%num_nodes != 0 else num_procs_pn
    print(f"Docking simulations running on {num_nodes} nodes and {num_procs} processes and {num_procs_pn} processes per node", flush=True)
        
    # Read sorted data and split compunds to various processes
    tic_read = perf_counter()
    driver_path = os.getenv("DRIVER_PATH")
    sorted_data_path = driver_path + "/sorted_data/sorted_smiles.csv"
    candidates = {"smiles": [], "score": []}
    with open(sorted_data_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key, value in row.items():
                    if key == "score":
                        candidates[key].append(float(value))
                    else:
                        candidates[key].append(value)
    toc_read = perf_counter()

    # Create the process group
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)
    for node_num in range(num_nodes):
        node_name = Node(nodelist[node_num]).hostname
        if node_num == num_nodes-1:
            num_procs_pn = remainder_procs_pn
        for proc in range(num_procs_pn):
            proc_id = node_num*num_procs_pn+proc
            local_policy = Policy(placement=Policy.Placement.HOST_NAME,
                                  host_name=node_name,
                                  cpu_affinity=[available_cores[proc]])
            grp.add_process(nproc=1,
                            template=ProcessTemplate(target=run_docking,
                                                        args=(
                                                            candidates["smiles"][proc_id::num_procs],
                                                            proc_id,
                                                            num_procs), 
                                                        cwd=run_dir,
                                                        policy=local_policy,
                                                        stdout=MSG_PIPE
                                                        )
                            )

    # Launch the ProcessGroup
    print(f"Starting Process Group for docking sims", flush=True)
    grp.init()
    grp.start()

    scores = []
    group_procs = [Process(None, ident=puid) for puid in grp.puids]
    for proc in group_procs:
        if proc.stdout_conn:
            std_out = read_output(proc.stdout_conn)
            score = std_out.split(",")
            for i in range(len(score)):
                tmp = score[i].replace("\n","")
                scores.append(float(tmp))
    grp.join()
    grp.close()
    print(f"Joined Process Group for Docking Sims",flush=True)

    # Collect candidate keys and save them to simulated keys
    tic_write = perf_counter()
    training_data_path = driver_path + "/training_data"
    if not os.path.exists(training_data_path):
        os.makedirs(training_data_path)
    with open(training_data_path+'/training_smiles.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['smiles', 'score'])
        writer.writerows(zip(candidates["smiles"],scores))
    toc_write = perf_counter()
    write_time = toc_write - tic_write
    
    total_io_time = write_time + (toc_read - tic_read)
    print(f'Performed docking simulation: total={0}, IO={total_io_time}',flush=True)
    

