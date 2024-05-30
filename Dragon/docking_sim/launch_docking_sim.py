import os

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from .docking_openeye import docking_switch

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

def launch_docking_sim(cdd: DDict, nodelist, num_procs: int, continue_event):
    """Launch the inference ruotine

    :param cdd: Dragon distributed dictionary for top candidates
    :type dd: DDict
    :param num_procs: number of processes to use for docking
    :type num_procs: int
    """
    num_nodes = len(nodelist)
    num_procs_pn = num_procs//num_nodes
    run_dir = os.getcwd()
    print(f"Nodes for docking {nodelist} {num_procs_pn=}",flush=True)

    # Create the process group
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    #grp = ProcessGroup(restart=False, pmi_enabled=True, ignore_error_on_exit=True, policy=global_policy)
    grp = ProcessGroup(restart=False, ignore_error_on_exit=True, policy=global_policy)
    for node_num in range(num_nodes):   
        node_name = Node(nodelist[node_num]).hostname
        for proc in range(num_procs_pn):
            proc_id = node_num*num_procs_pn+proc
            local_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=node_name,cpu_affinity=[proc])
            #args = [(cdd, num_procs, node_num*num_procs_pn+i, continue_event) for i in range(num_procs_pn)]
            grp.add_process(nproc=1, 
                            template=ProcessTemplate(target=docking_switch, 
                                                        args=(cdd, num_procs, proc_id, continue_event), 
                                                        cwd=run_dir,
                                                        policy=local_policy, 
                                                        stdout=MSG_PIPE,
                                                        stderr=MSG_PIPE))
    
    # Launch the ProcessGroup 
    grp.init()
    grp.start()
    print(f"Starting Process Group for Docking Sims", flush=True)
    group_procs = [Process(None, ident=puid) for puid in grp.puids]
    print(f"Docking processes:{grp.puids}",flush=True)
    for proc in group_procs:
        if proc.stdout_conn:
            std_out = read_output(proc.stdout_conn)
            print(std_out, flush=True)
        if proc.stderr_conn:
            std_err = read_error(proc.stderr_conn)
            print(std_err, flush=True)
    
    grp.join()
    grp.stop()
    #print(f"candidate keys {cdd.keys()}")
    total_sims = 0
    ckeys = cdd.keys()
    print(f"docking sims complete")
    for key in cdd.keys():
        if key[:9] == 'dock_iter':
            #print(f"{key=} {cdd[key]=}\n")
            total_sims += len(cdd[key]["smiles"])
    print(f"{total_sims=}")

