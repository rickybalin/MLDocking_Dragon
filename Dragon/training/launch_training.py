import os

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

#from .run_inference import infer_switch as infer
from .smiles_regress_transformer_run import training_switch

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
            print(tmp, flush=True)
            output += stdout_conn.recv()
    except EOFError:
        pass
    finally:
        stdout_conn.close()
    return output
 
def read_error(stderr_conn: Connection) -> str:
    """Read stderr from the Dragon connection.

    :param stderr_conn: Dragon connection to rank 0's stderr
    :type stderr_conn: Connection
    :return: string with the output from stderr
    :rtype: str
    """
    output = ""
    try:
        # this is brute force
        while True:
            output += stderr_conn.recv()
    except EOFError:
        pass
    finally:
        stderr_conn.close()
    return output

def launch_training(dd: DDict, node, candidate_dict: DDict, continue_event, BATCH, EPOCH, num_top_candidates):
    """Launch the inference ruotine

    :param dd: Dragon distributed dictionary
    :type dd: DDict
    :param num_procs: number of processes to use for inference
    :type num_procs: int
    """
    run_dir = os.getcwd()

    # Create the process group
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)

    grp = ProcessGroup(restart=False, ignore_error_on_exit=True, policy=global_policy)

    node_name = Node(node).hostname

    local_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=node_name, cpu_affinity=list(range(8)), device=Policy.Device.GPU, gpu_affinity=[3])
    grp.add_process(nproc=1, 
                    template=ProcessTemplate(target=training_switch,
                                                args=(dd, 
                                                    candidate_dict, 
                                                    continue_event, 
                                                    BATCH, EPOCH, 
                                                    num_top_candidates), 
                                                cwd=run_dir,
                                                policy=local_policy, 
                                                stdout=MSG_DEVNULL,
                                                stderr=MSG_DEVNULL))
    
    # Launch the ProcessGroup 
    grp.init()
    grp.start()
    print(f"Starting Process Group for Training")
    group_procs = [Process(None, ident=puid) for puid in grp.puids]
    for proc in group_procs:
        if proc.stdout_conn:
            std_out = read_output(proc.stdout_conn)
            print(std_out, flush=True)
        if proc.stderr_conn:
            std_err = read_error(proc.stderr_conn)
            print(std_err, flush=True)
    grp.join()
    grp.stop()
    print(f"Training process group stopped",flush=True)
    #print(dd["model_iter"])
    #print(dd["model"])

