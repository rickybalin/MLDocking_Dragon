import os
from time import perf_counter

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from inference.utils_transformer import ParamsJson, ModelArchitecture, pad

from .run_inference import infer

driver_path = os.getenv("DRIVER_PATH")


def load_pretrained_model(dd: DDict):
    # Read HyperParameters
    json_file = driver_path + "inference/config.json"
    hyper_params = ParamsJson(json_file)

    # Load model and weights
    try:
        # with open(f"pretrained_model.log","w") as sys.stdout:
        model = ModelArchitecture(hyper_params).call()
        model.load_weights(driver_path + f"inference/smile_regress.autosave.model.h5")
        print(f"{model=}", flush=True)
        dd["model"] = model
        dd["model_iter"] = 0
    except Exception as e:
        # eprint(e, flush=True)
        with open(f"pretrained_model.log", "a") as f:
            f.write(f"{e}")


def launch_inference(data_dd: DDict, 
                     model_list_dd: DDict, 
                     nodelist,
                     num_procs: int = 1, 
                     inf_num_limit = None):
    """Launch the inference ruotine

    :param dd: Dragon distributed dictionary
    :type dd: DDict
    :param num_procs: number of processes to use for inference
    :type num_procs: int
    """
    num_inf_nodes = len(nodelist)

    num_ccs = 1
    if int(os.getenv("USE_CCS")) == 1:
        ccs_string = os.getenv("ZEX_NUMBER_OF_CCS")
        num_ccs = int(ccs_string.split(",")[0].split(":")[1])
        print(f"Using {num_ccs} CCS on Aurora PVC",flush=True)

    gpu_devices_string = os.getenv("GPU_DEVICES")
    inf_gpu_bind = []
    for g in gpu_devices_string.split(","):
        for _ in range(num_ccs):
            if "." in g:
                inf_gpu_bind.append([float(g)])
            else:
                inf_gpu_bind.append([int(g)])
    num_procs_pn = len(inf_gpu_bind)  # number of procs per node is number of gpus
    print(f"Inference running on {num_inf_nodes} nodes and {num_procs_pn} processes per node", flush=True)

    cpu_affinity_string = os.getenv("INF_CPU_AFFINITY")
    cpu_ranges = cpu_affinity_string.split(":")
    inf_cpu_bind = []
    for cr in cpu_ranges:
        bind_threads = []
        thread_ranges = cr.split(",")
        for tr in thread_ranges:
            t = tr.split("-")
            if len(t) == 1:
                bind_threads.append(int(t[0]))
            elif len(t) == 2:
                start_t = int(t[0])
                end_t = int(t[1])
                for st in range(start_t, end_t + 1):
                    bind_threads.append(st)
        inf_cpu_bind.append(bind_threads)

    run_dir = os.getcwd()
    #print(f"{inf_cpu_bind=}")
    #print(f"{inf_gpu_bind=}")
    if len(inf_cpu_bind) != len(inf_gpu_bind):
        raise (Exception("Number of cpu bindings does not match the number of gpus"))

    # Create the process group
    tic = perf_counter()
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy)
    for node_num in range(num_inf_nodes):
        node_name = Node(nodelist[node_num]).hostname
        for proc in range(num_procs_pn):
            proc_id = node_num * num_procs_pn + proc

            local_policy = Policy(placement=Policy.Placement.HOST_NAME,
                                  host_name=node_name, 
                                  cpu_affinity=inf_cpu_bind[proc],
                                  gpu_affinity=inf_gpu_bind[proc])
            grp.add_process(nproc=1, 
                            template=ProcessTemplate(target=infer, 
                                                     args=(data_dd,
                                                        model_list_dd,
                                                        num_procs_pn,
                                                        proc_id, 
                                                        None, # Continue event not used in sequential wf
                                                        inf_num_limit,
                                                        ), 
                                                     cwd=run_dir,
                                                     policy=local_policy,))
    
    # Launch the ProcessGroup 
    print(f"Starting Process Group for inference", flush=True)
    grp.init()
    grp.start()
    grp.join()
    grp.close()
    toc = perf_counter()
    print(f"Performed inference in {toc-tic} seconds", flush=True)
