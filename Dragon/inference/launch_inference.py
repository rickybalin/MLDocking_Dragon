import os
import pickle

import dragon
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, TemplateProcess, Popen, MSG_PIPE, MSG_DEVNULL
from dragon.utils import B64
from dragon.infrastructure.connection import Connection

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
            output += stdout_conn.recv()
    except EOFError:
        pass
    finally:
        stdout_conn.close()
    return output

def launch_inference_mpi(_dict, num_ranks: int):
    """Launch the MPI inference ruotine

    :param _dict: Dragon distributed dictionary
    :type _dict: ...
    :param num_ranks: number of MPI ranks to use for inference
    :type num_ranks: int
    """
    # Serialize the Dragon Dictionary
    serial_dict = B64.bytes_to_str(pickle.dumps(_dict))

    # Set up the MPI inference routine arguments
    exe = "python"
    args = ["run_inference_mpi.py", serial_dict]
    run_dir = os.getcwd()+"/inference/"

    # Create the process group
    grp = ProcessGroup(restart=False, pmi_enabled=True)

    # Pipe the stdout output from the head process to a Dragon connection,
    # and all other processes to DEVNULL
    grp.add_process(nproc=1, 
                    template=TemplateProcess(target=exe, 
                                             args=args, 
                                             cwd=run_dir, 
                                             stdout=Popen.PIPE))
    grp.add_process(nproc=num_ranks - 1,
                    template=TemplateProcess(target=exe, 
                                             args=args, 
                                             cwd=run_dir, 
                                             stdout=Popen.DEVNULL))
    
    # Launch the ProcessGroup (MPI inference routine)
    grp.init()
    grp.start()
    group_procs = [Process(None, ident=puid) for puid in grp.puids]
    for proc in group_procs:
        if proc.stdout_conn:
            std_out = read_output(proc.stdout_conn)
            print(std_out, flush=True)
    grp.join()
    grp.stop()

    _dict.close()