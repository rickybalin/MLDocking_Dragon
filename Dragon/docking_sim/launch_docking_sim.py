import os

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from .docking_openeye import run_docking


def launch_docking_sim(cdd, docking_iter, num_procs, nodelist):
    """Launch docking simulations

    :param cdd: Dragon distributed dictionary for top candidates
    :type dd: DDict
    :param num_procs: number of processes to use for docking
    :type num_procs: int
    """
    num_nodes = len(nodelist)
    num_procs_pn = num_procs//num_nodes
    run_dir = os.getcwd()

    # Create the process group
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)
    for node_num in range(num_nodes):
        node_name = Node(nodelist[node_num]).hostname
        for proc in range(num_procs_pn):
            proc_id = node_num*num_procs_pn+proc
            local_policy = Policy(placement=Policy.Placement.HOST_NAME,
                                  host_name=node_name,
                                  cpu_affinity=[proc])
            grp.add_process(nproc=1,
                            template=ProcessTemplate(target=run_docking,
                                                        args=(cdd,
                                                            docking_iter,
                                                            proc_id,
                                                            num_procs), 
                                                        cwd=run_dir,
                                                        policy=local_policy,
                                                        ))

    # Launch the ProcessGroup
    grp.init()
    grp.start()
    print(f"Starting Process Group for Docking Sims on {num_procs} procs", flush=True)
    grp.join()
    print(f"Joined Process Group for Docking Sims",flush=True)
    grp.close()
