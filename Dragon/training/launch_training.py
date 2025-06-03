import os

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from .smiles_regress_transformer_run import fine_tune


def launch_training(model_dd: DDict, sim_dd: DDict, node, BATCH, EPOCH):
    """Launch the inference ruotine

    :param dd: Dragon distributed dictionary
    :type dd: DDict
    :param num_procs: number of processes to use for inference
    :type num_procs: int
    """
    run_dir = os.getcwd()

    # Create the process group
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)

    node_name = Node(node).hostname

    local_policy = Policy(placement=Policy.Placement.HOST_NAME, 
                          host_name=node_name, 
                          cpu_affinity=list(range(8)), 
                          gpu_affinity=[3])
    grp.add_process(nproc=1, 
                    template=ProcessTemplate(target=fine_tune,
                                                args=(model_dd, 
                                                    sim_dd, 
                                                    BATCH, EPOCH, 
                                                    ), 
                                                cwd=run_dir,
                                                policy=local_policy, 
                                                ))
    
    # Launch the ProcessGroup 
    grp.init()
    grp.start()
    print(f"Starting Process Group for Training")
    
    grp.join()
    grp.close()
    print(f"Training process group stopped",flush=True)
    #print(dd["model_iter"])
    #print(dd["model"])

