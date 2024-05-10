import os

import multiprocessing as mp
import dragon
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict.ddict import DDict

from .run_inference_mpi import infer

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

def mpi_worker(q):
    """Start the MPI inference job
    """
    dd = q.get()
    infer(dd)

def launch_inference_mpi(dd: DDict, num_ranks: int):
    """Launch the MPI inference ruotine

    :param dd: Dragon distributed dictionary
    :type dd: DDict
    :param num_ranks: number of MPI ranks to use for inference
    :type num_ranks: int
    """
    # Create a queue for each rank
    dd_q = mp.Queue(maxsize=num_ranks)
    for _ in range(num_ranks):
        dd_q.put(dd)

    # Set up the MPI inference routine arguments
    run_dir = os.getcwd()

    # Create the process group
    grp = ProcessGroup(restart=False, pmi_enabled=True)
    grp.add_process(nproc=1, 
                    template=ProcessTemplate(target=mpi_worker, 
                                             args=(dd_q,), 
                                             cwd=run_dir, 
                                             stdout=MSG_PIPE,
                                             stderr=MSG_PIPE))
    grp.add_process(nproc=num_ranks - 1,
                    template=ProcessTemplate(target=mpi_worker, 
                                             args=(dd_q,), 
                                             cwd=run_dir, 
                                             stdout=MSG_PIPE))
                                             #stdout=MSG_DEVNULL))
    
    # Launch the ProcessGroup (MPI inference routine)
    grp.init()
    grp.start()
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

