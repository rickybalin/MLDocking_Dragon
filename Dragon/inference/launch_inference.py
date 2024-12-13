import os

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node
import os
import sys

from inference.utils_transformer import ParamsJson, ModelArchitecture, pad

from .run_inference import infer

driver_path = os.getenv("DRIVER_PATH")


def load_pretrained_model(dd: DDict):
    # Read HyperParameters 
    json_file = driver_path+'inference/config.json'
    hyper_params = ParamsJson(json_file)

    # Load model and weights
    try:
        #with open(f"pretrained_model.log","w") as sys.stdout:
        model = ModelArchitecture(hyper_params).call()
        model.load_weights(driver_path+f'inference/smile_regress.autosave.model.h5')
        print(f"{model=}", flush=True)
        dd["model"] = model
        dd["model_iter"] = 0
    except Exception as e:
        #eprint(e, flush=True)
        with open(f"pretrained_model.log",'a') as f:
            f.write(f"{e}")
    

def launch_inference(dd: DDict, nodelist, num_procs: int, inf_num_limit):
    """Launch the inference ruotine

    :param dd: Dragon distributed dictionary
    :type dd: DDict
    :param num_procs: number of processes to use for inference
    :type num_procs: int
    """
    num_inf_nodes = len(nodelist)
    num_procs_pn = num_procs//num_inf_nodes

    hostname = os.popen("hostname -f").read()
    if "americas" in hostname or "aurora" in hostname:
        inf_cpu_bind = [1, 10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100]
        inf_gpu_bind = [0.0,0.1,1.0,1.1,2.0,2.1,3.0,3.1,4.0,4.1,5.0,5.1]
    else:
        inf_cpu_bind = [4, 12, 20, 28]
        inf_gpu_bind = [3, 2, 1, 0]
    run_dir = os.getcwd()

    # Create the process group
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    #grp = ProcessGroup(restart=False, ignore_error_on_exit=True, policy=global_policy)
    grp = ProcessGroup(restart=False, ignore_error_on_exit=False, policy=global_policy)
    for node_num in range(num_inf_nodes):   
        node_name = Node(nodelist[node_num]).hostname
        for proc in range(num_procs_pn):
            proc_id = node_num*num_procs_pn+proc
            local_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=node_name, 
                                                                        cpu_affinity=[inf_cpu_bind[proc]],
                                                                        gpu_affinity=[inf_gpu_bind[proc]])
            grp.add_process(nproc=1, 
                            template=ProcessTemplate(target=infer, 
                                                     args=(dd,
                                                        num_procs,
                                                        proc_id, 
                                                        None, # Continue event not used in sequential wf
                                                        inf_num_limit,
                                                        ), 
                                                     cwd=run_dir,
                                                     policy=local_policy,))
    
    # Launch the ProcessGroup 
    grp.init()
    grp.start()
    print(f"Starting Process Group for Inference",flush=True)
    grp.join()
    grp.close()
