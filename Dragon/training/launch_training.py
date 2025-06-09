import os
from time import perf_counter

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from .smiles_regress_transformer_run import fine_tune

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


def launch_training(model_dd: DDict, sim_dd: DDict, node, BATCH, EPOCH):
    """Launch the inference ruotine

    :param dd: Dragon distributed dictionary
    :type dd: DDict
    :param num_procs: number of processes to use for inference
    :type num_procs: int
    """
    run_dir = os.getcwd()

    # Create the process group
    gpu_devices = os.getenv("GPU_DEVICES").split(",")
    gpu_devices = [float(gid) for gid in gpu_devices]
    gpu_devices = [gpu_devices[0]] # training only needs 1 GPU
    cpu_affinity = os.getenv("TRAIN_CPU_AFFINITY").split(",")
    cpu_affinity = [int(cid) for cid in cpu_affinity]
    print(f'Launching training on {cpu_affinity} CPUs and {gpu_devices} GPU',flush=True)
    
    tic = perf_counter()
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)
    node_name = Node(node).hostname
    local_policy = Policy(placement=Policy.Placement.HOST_NAME, 
                          host_name=node_name, 
                          cpu_affinity=cpu_affinity, 
                          gpu_affinity=[gpu_devices[0]])
    grp.add_process(nproc=1, 
                    template=ProcessTemplate(target=fine_tune,
                                                args=(model_dd, 
                                                    sim_dd, 
                                                    BATCH, 
                                                    EPOCH, 
                                                ),
                                                kwargs={
                                                    'save_model': False,
                                                    'debug': False
                                                },  
                                                cwd=run_dir,
                                                policy=local_policy, 
                                                ))
    
    # Launch the ProcessGroup 
    print(f"Starting Process Group for training",flush=True)
    grp.init()
    grp.start()
    grp.join()
    grp.close()
    toc = perf_counter()
    print(f"Performed training in {toc-tic} seconds", flush=True)


