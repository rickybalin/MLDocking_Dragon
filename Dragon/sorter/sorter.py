from time import perf_counter
from typing import Tuple
import argparse
import os
import random
import gc
import dragon
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict
from dragon.native.process_group import ProcessGroup
from dragon.infrastructure.policy import Policy
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.native.machine import cpu_count, current, System, Node
from .sort_mpi import mpi_sort
import datetime
import time
import heapq

global data_dict
data_dict = None


def init_worker(q):
    global data_dict
    data_dict = q.get()
    return


def filter_candidate_keys(ckeys: list):
    ckeys = [
        key for key in ckeys if "iter" not in key and key[0] != "d" and key[0] != "l"
    ]
    return ckeys


def compare_candidate_results(
    candidate_dict,
    continue_event,
    num_return_sorted,
    ncompare=3,
    max_iter=100,
    purge=True,
):

    print(f"Comparing Candidate Lists")
    end_workflow = False
    candidate_keys = candidate_dict.keys()
    sort_iter = 0
    if "sort_iter" in candidate_keys:
        sort_iter = candidate_dict["sort_iter"]
        # candidate_keys.remove('iter')
    candidate_keys = filter_candidate_keys(candidate_keys)
    print(f"{candidate_keys=}")
    # ncompare = min(ncompare,len(candidate_keys))
    candidate_keys.sort(reverse=True)
    # print(f"{candidate_keys=}")
    num_top_candidates = 0
    if len(candidate_keys) > 0:
        num_top_candidates = len(candidate_dict[candidate_keys[0]]["smiles"])

    if sort_iter > max_iter:
        # end if maximum number of iterations reached
        end_workflow = True
        print(
            f"Ending workflow: sort_iter {sort_iter} exceeded max_iter {max_iter}",
            flush=True,
        )
    elif len(candidate_keys) > ncompare and num_top_candidates == num_return_sorted:
        # look for unique entries in most recent ncompare lists
        # only do this if there are enough lists to compare and if enough candidates have been identified

        not_in_common = []
        model_iters = []
        print(f"{candidate_dict=}", flush=True)
        for i in range(ncompare):
            for j in range(ncompare):
                if i < j:
                    ckey_i = candidate_keys[i]
                    ckey_j = candidate_keys[j]
                    not_in_common += list(
                        set(candidate_dict[ckey_i]["smiles"])
                        ^ set(candidate_dict[ckey_j]["smiles"])
                    )
                    model_iter_i = list(set(candidate_dict[ckey_i]["model_iter"]))
                    model_iter_j = list(set(candidate_dict[ckey_j]["model_iter"]))
                    model_iters += model_iter_i
                    model_iters += model_iter_j
        print(f"Number not in common {len(not_in_common)}")
        # ncompare consecutive lists are identical, end workflow
        print(f"{model_iters=}")

        # End workflow if ncompare lists with unique model_iters are idententical
        if len(not_in_common) == 0 and len(model_iters) == 2 * ncompare:
            if len(set(model_iters)) == ncompare:
                print(f"Ending workflow: {ncompare} lists identical", flush=True)
                end_workflow = True

        # If purge, only keep ncompare sorting lists
        if purge:
            if len(candidate_keys) > ncompare:
                del candidate_dict[candidate_keys[-1]]

    print(f"{end_workflow=}")
    if end_workflow:
        continue_event.clear()


def save_top_candidates_list(candidate_dict):
    ckeys = filter_candidate_keys(candidate_dict.keys())
    if len(ckeys) > 0:
        max_ckey = max(ckeys)
        top_candidates = candidate_dict[max_ckey]
        top_smiles = top_candidates["smiles"]
        lines = [sm + "\n" for sm in top_smiles]

        with open(f"top_candidates.out", "w") as f:
            f.writelines(lines)


def sort_controller(
    dd,
    num_return_sorted: str,
    max_procs: int,
    nodelist: list,
    candidate_dict,
    continue_event,
    checkpoint_interval_min=10,
):

    iter = 0
    with open("sort_controller.log", "a") as f:
        f.write(f"{datetime.datetime.now()}: Starting Sort Controller\n")
        f.write(
            f"{datetime.datetime.now()}: Sorting for {num_return_sorted} candidates\n"
        )

    ckeys = candidate_dict.keys()
    if "max_sort_iter" not in ckeys:
        candidate_dict["max_sort_iter"] = "-1"

    check_time = perf_counter()

    continue_flag = True

    while continue_flag:
        gc.collect()

        with open("sort_controller.log", "a") as f:
            f.write(f"{datetime.datetime.now()}: Starting iter {iter}\n")
        tic = perf_counter()
        print(f"Sort iter {iter}", flush=True)
        # sort_dictionary_queue(_dict, num_return_sorted, max_procs, key_list, candidate_dict)
        # sort_dictionary_pool(_dict, num_return_sorted, max_procs, key_list, candidate_dict)
        sort_dictionary_pg(dd, num_return_sorted, max_procs, nodelist, candidate_dict)
        print(f"Finished pg sort", flush=True)
        # dd["sort_iter"] = iter
        # max_ckey = candidate_dict["max_sort_iter"]
        # inf_results = candidate_dict[max_ckey]["inf"]
        # cutoff_check = [p for p in inf_results if p < 9 and p > 0]
        # print(f"Cutoff check: {len(cutoff_check)} inf vals below cutoff")
        compare_candidate_results(
            candidate_dict, continue_event, num_return_sorted, max_iter=50
        )

        if (check_time - perf_counter()) / 60.0 > checkpoint_interval_min:
            save_top_candidates_list(candidate_dict)
            check_time = perf_counter()
        toc = perf_counter()
        with open("sort_controller.log", "a") as f:
            f.write(f"{datetime.datetime.now()}: iter {iter}: sort time {toc-tic} s\n")
        iter += 1

        if continue_event is None:
            continue_flag = False
        else:
            continue_flag = continue_event.is_set()
    if continue_event is not None:
        ckeys = candidate_dict.keys()
        print(f"final {ckeys=}")
        save_top_candidates_list(candidate_dict)
    # ckeys = filter_candidate_keys(ckeys)
    # if len(ckeys) > 0:
    #     ckey_max = max(ckeys)
    #     print(f"top candidates = {candidate_dict[ckey_max]}")


# A SentinelQueue is a queue that raise EOFError when end of
# file is reached. It knows to do this by writing a sentinel
# into the queue when the queue is closed. A SentinelQueue
# contains a multiprocessing Queue. Other methods could
# be implemented for SentinelQueue, but are not needed in
# this example.
class SentinelQueue:
    _SENTINEL = "sentinel_queue_sentinel"

    def __init__(self, the_queue):
        self.queue = the_queue

    def get(self):
        item = self.queue.get()
        if item == SentinelQueue._SENTINEL:
            raise EOFError("At EOF")

        return item

    def put(self, item):
        self.queue.put(item)

    def close(self):
        self.queue.put(SentinelQueue._SENTINEL)
        self.queue.close()


# The PQEntry class is needed by the priority
# queue which is used to always know which queue
# to get the next value from. The __lt__ orders
# the priority queue elements by their original
# values. But the queue index of where the value
# came from is carried along with the value in the
# priority queue so the merging algorithm knows where
# to get the next value. In this way, the total number
# of entries in the priority queue is never more than the
# fanin value of the MergePool.
class PQEntry:
    def __init__(self, value, queue_index):
        self._value = value
        self._queue_index = queue_index

    def __lt__(self, other):
        return self.value[0] > other.value[0]

    @property
    def queue_index(self):
        return self._queue_index

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return "PQEntry(" + str(self.value) + "," + str(self.queue_index) + ")"

    def __str__(self):
        return repr(self)


def manager_sorter(dd, num_return_sorted, sorted_queue, manager_id):
    try:
        shutdown_event = mp.Event()
        shutdown_event.clear()

        # We colocate the shutdown event object with the manager_sorter
        # to make checking it as efficient as possible in the loop
        # further down in this algorithm. The main process will get
        # an event from each manager.
        sorted_queue.put(shutdown_event)

        my_manager = dd.manager(manager_id)

        keys = my_manager.keys()
        this_value = []
        for key in keys:
            val = my_manager[key]
            # print(val.keys(), flush=True) this prints
            # dict_keys(['f_name', 'smiles', 'inf'])
            this_value.extend(zip(val["inf"], val["smiles"]))
            this_value = heapq.nlargest(
                num_return_sorted, this_value, key=lambda x: x[0]
            )

        # print(f"From manager {manager_id} the list is: {this_value}", flush=True)

        # This could be uncommented to sort everything in memory at once, but that
        # might consume too much memory (and did on the medium dataset - resulting
        # in a task dying - I am sure it was oom). If this is uncommented, then
        # comment out the call to heapq.nlargest above. It would not be needed.
        # I tested the two alternatives and the nlargest choice did not seem to
        # take any more time than the sort below and it will not overwhelm memory
        # since it finds the nlargest piece by piece.
        # this_value.sort(key=lambda tup: tup[0], reverse=True)

        for i in range(num_return_sorted):
            # print(f"Putting value from manager {manager_id}", flush=True)
            if shutdown_event.is_set():
                break
            sorted_queue.put(this_value[i])

        sorted_queue.close()
    except Exception as ex:
        print(f"Exception in manager_sorter: {ex}", flush=True)

    print(f"Exiting sorter for manager {manager_id}", flush=True)


def sort_dictionary_pg(dd: DDict, num_return_sorted):

    stats = dd.dstats

    print(f"Finding the best {num_return_sorted} candidates.", flush=True)

    # At larger scales, it might be useful create more than one of these processes,
    # themselves in a process group where each of them is given an output queue and
    # a set of managers to merge (instead of merging all managers in one process).
    # Then the second-level merging would merge their managers, writing the merged
    # values to their output queues while a third level merge is done to merge all the
    # second level merges together. In this way, this algorithm can scale to whatever
    # size is necessary.

    grp = ProcessGroup(restart=False)

    sorted_queues = []
    shutdown_events = []

    # print(f"Launching sorting process group {nodelist}", flush=True)
    for manager_id in stats:
        sorted_queue = SentinelQueue(mp.Queue())
        sorted_queues.append(sorted_queue)

        node_name = stats[manager_id].hostname
        local_policy = Policy(
            placement=Policy.Placement.HOST_NAME,
            host_name=node_name,
        )
        grp.add_process(
            nproc=1,
            template=ProcessTemplate(
                target=manager_sorter,
                args=(dd, num_return_sorted, sorted_queue, manager_id),
                policy=local_policy,
            ),
        )

    print(f"Added processes to sorting group", flush=True)
    grp.init()
    grp.start()
    print(f"Starting Process Group for Sorting", flush=True)
    sort_start = perf_counter()

    # Get the shutdown event objects which are sent first.
    for i in range(len(sorted_queues)):
        shutdown_events.append(sorted_queues[i].get())

    # merge sorted values from manager_sorters
    # prime the priority queue
    priority_queue = []

    for i in range(len(sorted_queues)):
        try:
            item = sorted_queues[i].get()
            heapq.heappush(priority_queue, PQEntry(item, i))
        except EOFError:
            pass

    # merge the values from different managers
    candidate_list = []
    while len(priority_queue) > 0 and len(candidate_list) < num_return_sorted:
        # If items are not in strictly decreasing order for values, then
        # you need to reverse the < to a > in the PQEntry __lt__ method.
        item = heapq.heappop(priority_queue)
        candidate_list.append(item.value)

        try:
            next = sorted_queues[item.queue_index].get()
            heapq.heappush(priority_queue, PQEntry(next, item.queue_index))
        except EOFError:
            pass
    sort_end = perf_counter()

    if len(priority_queue) == 0:
        print("We ended with priority_queue length 0", flush=True)

    sort_time = sort_end - sort_start
    print(f"Performed sorting in {sort_time:.3f} seconds \n", flush=True)

    print("HERE IS THE CANDIDATE LIST (first 10 only)")
    print("******************************************", flush=True)
    print(candidate_list[:10], flush=True)

    for i in range(len(sorted_queues)):
        shutdown_events[i].set()

    while len(sorted_queues) > 0:
        sorted_queue = sorted_queues.pop()
        try:
            while True:
                # we get more items in case we need to wake up the manager sorter
                # from a blocking put operation so it can check the shutdown_event.
                sorted_queue.get()
        except:
            pass
        try:
            sorted_queue.destroy()
            del sorted_queue
        except:
            pass

    grp.join()
    grp.close()

    print("Finished Sorting!!!", flush=True)


def create_dummy_data(_dict, num_managers):

    NUMKEYS = 100

    for i in range(NUMKEYS):
        key = f"{i%num_managers}_{i}"
        _dict[key] = {
            "inf": [random.randrange(0, 1000, 1) for j in range(10)],
            "smiles": [
                "".join(
                    [
                        random.choice(string.ascii_uppercase + string.digits)
                        for k in range(20)
                    ]
                )
                for j in range(10)
            ],
        }


if __name__ == "__main__":
    # Import command line arguments
    parser = argparse.ArgumentParser(description="Distributed dictionary example")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes the dictionary distributed across",
    )
    parser.add_argument(
        "--managers_per_node",
        type=int,
        default=1,
        help="number of managers per node for the dragon dict",
    )
    parser.add_argument(
        "--mem_per_node",
        type=int,
        default=1,
        help="managed memory size per node for dictionary in GB",
    )
    parser.add_argument(
        "--max_procs_per_node",
        type=int,
        default=10,
        help="Maximum number of processes in a Pool",
    )
    parser.add_argument(
        "--dictionary_timeout",
        type=int,
        default=10,
        help="Timeout for Dictionary in seconds",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files",
        help="Path to pre-sorted SMILES strings to load",
    )
    args = parser.parse_args()

    # Start distributed dictionary
    mp.set_start_method("dragon")
    total_mem_size = args.mem_per_node * args.num_nodes * (1024 * 1024 * 1024)
    dd = DDict(args.managers_per_node, args.num_nodes, total_mem_size)
    print("Launched Dragon Dictionary \n", flush=True)

    loader_proc = mp.Process(
        target=create_dummy_data,
        args=(dd, args.num_nodes * args.managers_per_node),
    )  # ignore_exit_on_error=True)
    loader_proc.start()
    print("Process started", flush=True)

    loader_proc.join()
    print("Process ended", flush=True)

    print(f"Number of keys in dictionary is {len(dd.keys())}", flush=True)

    candidate_queue = mp.Queue()
    top_candidate_number = 10
    sorter_proc = mp.Process(
        target=sort_dictionary,
        args=(
            dd,
            top_candidate_number,
            args.max_procs_per_node * args.num_nodes,
            dd.keys(),
            candidate_queue,
        ),
    )  # ignore_exit_on_error=True)
    sorter_proc.start()
    print("Process started", flush=True)

    sorter_proc.join()
    print("Process ended", flush=True)
    top_candidates = candidate_queue.get(timeout=None)
    print(f"{top_candidates=}")

    # # Launch the data loader
    # print("Loading inference data into Dragon Dictionary ...", flush=True)
    # tic = perf_counter()
    # load_inference_data(dd, args.data_path, args.max_procs)
    # toc = perf_counter()
    # load_time = toc - tic
    # print(f"Loaded inference data in {load_time:.3f} seconds \n", flush=True)

    # Close the dictionary
    print("Done, closing the Dragon Dictionary", flush=True)
    dd.destroy()
