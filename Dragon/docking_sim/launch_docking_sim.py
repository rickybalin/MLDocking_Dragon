import os

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from .docking_openeye import run_docking

def work_finished(nproc,file="docking_switch.log"):

    nfinished = 0
    if os.path.isfile(file):
        with open(file,"r") as f:
            lines = f.readlines()

            for line in lines:
                if "Finished docking sims" in line:
                    nfinished += 1
    if nfinished < nproc:
        return False
    else:
        return True
    

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
    grp = ProcessGroup(restart=False, ignore_error_on_exit=True, policy=global_policy)
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
    group_procs = [Process(None, ident=puid) for puid in grp.puids]
    
    #for proc in group_procs:
    #    if proc.stdout_conn:
    #        std_out = read_output(proc.stdout_conn)
    #        print(std_out, flush=True)
    #    if proc.stderr_conn:
    #        std_err = read_error(proc.stderr_conn)
    #        print(std_err, flush=True)
    
    #grp.join()
    while not work_finished(num_procs,file="finished_run_docking.log"):
        try:
            grp.join(timeout=10)
        except TimeoutError:
            continue
    print(f"Docking workers finished")
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
