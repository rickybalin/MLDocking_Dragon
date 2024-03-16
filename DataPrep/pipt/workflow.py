import logging
import sys
from argparse import ArgumentParser
from functools import partial, update_wrapper
from pathlib import Path
from typing import Any, List, Optional

from colmena.models import Result
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.thinker import BaseThinker, agent, result_processor
from proxystore.store import register_store
from proxystore.store.file import FileStore

from pipt.parsl import ComputeSettingsTypes
from pipt.utils import BaseSettings, mkdir_validator, path_validator

# THIS WORKS
# def run_task(
#     smiles_batch: List[str],
#     receptor_oedu_file: Path,
#     output_dir: Path,
#     node_local_path: Optional[Path] = None,
# ) -> None:
#     """Run a single docking task on an input SMILES string.

#     Parameters
#     ----------
#     smiles_batch : List[str]
#         A list of SMILES strings.
#     receptor_oedu_file : Path
#         Path to the receptor .oedu file.
#     output_dir : Path
#         Path to the output directory to write a subdirectory for each
#         SMILES string containing `metrics.csv`.
#     node_local_path : Path
#         Path to a directory on the compute node that can be used for temp space.
#     """
#     import shutil
#     import subprocess
#     import uuid

#     # from pipt.openeye_dock_tools import run_parallel_docking
#     # Stage receptor file on node local storage
#     if node_local_path is not None:
#         # node_file_lock = node_local_path / "file.lock"
#         # if node_file_lock.exists():
#         tmp = node_local_path / f"receptor-{uuid.uuid4()}-{receptor_oedu_file.name}"
#         shutil.copy(receptor_oedu_file, tmp)
#         receptor_oedu_file = tmp  # node_local_path / receptor_oedu_file.name
#         # local_receptor_oedu_file = node_local_path / receptor_oedu_file.name
#         # if not local_receptor_oedu_file.exists():
#         #     shutil.copy(receptor_oedu_file, local_receptor_oedu_file)
#         # receptor_oedu_file = local_receptor_oedu_file

#     # # Run the docking computation
#     # docking_scores = run_parallel_docking(smiles_batch, receptor_oedu_file, num_workers=64)

#     # # Format the output file
#     # file_contents = "SMILES, DockingScore\n"
#     # file_contents += "\n".join(f"{smiles},{score}" for smiles, score in zip(smiles_batch, docking_scores))
#     # file_contents += "\n"

#     # # Write the output file
#     # with open(output_dir / f"metrics-{uuid.uuid4()}.csv", "w") as f:
#     #     f.write(file_contents)

#     smiles_file = node_local_path / f"smiles-{uuid.uuid4()}.smi"
#     with open(smiles_file, "w") as f:
#         f.write("\n".join(smiles_batch))

#     output_file = output_dir / f"metrics-{uuid.uuid4()}.csv"

#     command = f"python -m pipt.openeye_dock_tools -r {receptor_oedu_file} -s {smiles_file} -o {output_file} -t {node_local_path} -n 64"
#     subprocess.run(command.split())

#     # Clean up the receptor file
#     if node_local_path is not None:
#         receptor_oedu_file.unlink()


def run_task(
    smiles_batch: List[str],
    receptor_oedu_file: Path,
    output_dir: Path,
    node_local_path: Optional[Path] = None,
) -> None:
    """Run a single docking task on an input SMILES string.

    Parameters
    ----------
    smiles_batch : List[str]
        A list of SMILES strings.
    receptor_oedu_file : Path
        Path to the receptor .oedu file.
    output_dir : Path
        Path to the output directory to write a subdirectory for each
        SMILES string containing `metrics.csv`.
    node_local_path : Path
        Path to a directory on the compute node that can be used for temp space.
    """
    # import shutil
    import uuid

    from pipt.openeye_dock_tools import run_docking

    # Stage receptor file on node local storage
    # if node_local_path is not None:
    #     # node_file_lock = node_local_path / "file.lock"
    #     # if node_file_lock.exists():
    #     tmp = node_local_path / f"receptor-{uuid.uuid4()}-{receptor_oedu_file.name}"
    #     shutil.copy(receptor_oedu_file, tmp)
    #     receptor_oedu_file = tmp  # node_local_path / receptor_oedu_file.name
    # Run the docking computation
    docking_scores = [
        run_docking(smiles, receptor_oedu_file, node_local_path)
        for smiles in smiles_batch
    ]

    # Format the output file
    file_contents = "SMILES,DockingScore\n"
    file_contents += "\n".join(
        f"{smiles},{score}" for smiles, score in zip(smiles_batch, docking_scores)
    )
    file_contents += "\n"

    # Write the output file
    with open(output_dir / f"metrics-{uuid.uuid4()}.csv", "w") as f:
        f.write(file_contents)

    # # Clean up the receptor file
    # if node_local_path is not None:
    #     receptor_oedu_file.unlink()


class Thinker(BaseThinker):  # type: ignore[misc]
    def __init__(
        self,
        input_arguments: List[Any],
        result_dir: Path,
        num_parallel_tasks: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.task_idx = 0
        self.num_parallel_tasks = num_parallel_tasks
        self.input_arguments = input_arguments
        self.logger.info(f"Processing {len(self.input_arguments)} input arguments")

    def log_result(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        with open(self.result_dir / f"{topic}.json", "a") as f:
            print(result.json(exclude={"inputs", "value"}), file=f)

    def submit_task(self) -> None:
        # If we finished processing all the results, then stop
        if self.task_idx >= len(self.input_arguments):
            self.done.set()
            return

        task_args = self.input_arguments[self.task_idx]
        self.task_idx += 1

        self.queues.send_inputs(
            *task_args, method="run_task", topic="task", keep_inputs=False
        )

    @agent(startup=True)  # type: ignore[misc]
    def start_tasks(self) -> None:
        # Only submit num_parallel_tasks at a time
        for _ in range(self.num_parallel_tasks):
            self.submit_task()

    @result_processor(topic="task")  # type: ignore[misc]
    def process_task_result(self, result: Result) -> None:
        """Handles the returned result of the task function and log status."""
        self.log_result(result, "task")
        if not result.success:
            self.logger.warning("Bad task result")

        # The old task is finished, start a new one
        self.submit_task()


class WorkflowSettings(BaseSettings):
    """Provide a YAML interface to configure the workflow."""

    # Workflow setup parameters
    output_dir: Path
    """Path this particular workflow writes to."""

    # Inference parameters
    smiles_file: Path
    """File containing one SMILES string per line."""
    receptor_oedu_file: Path
    """Path to the receptor .oedu file to dock to."""
    smiles_batch_size: int = 100
    """Number of SMILES strings to process in a single task."""
    num_parallel_tasks: int = 4
    """Number of parallel task to run (should be the total number of GPUs)"""
    node_local_path: Optional[Path] = None
    """Node local storage option."""

    compute_settings: ComputeSettingsTypes
    """The compute settings to use."""

    # validators
    _output_dir_mkdir = mkdir_validator("output_dir")
    _smiles_file_exists = path_validator("smiles_file")
    # _receptor_oedu_file_exists = path_validator("receptor_oedu_file")

    def configure_logging(self) -> None:
        """Set up logging."""
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.output_dir / "runtime.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )


def batch_data(data: List[Any], batch_size: int) -> List[List[Any]]:
    """Batch data into chunks of size batch_size."""
    batches = [
        data[i * batch_size : (i + 1) * batch_size]
        for i in range(0, len(data) // batch_size)
    ]
    if len(data) > batch_size * len(batches):
        batches.append(data[len(batches) * batch_size :])
    return batches


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    cfg = WorkflowSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.output_dir / "params.yaml")
    cfg.configure_logging()

    # Make the proxy store
    store = FileStore(name="file", store_dir=str(cfg.output_dir / "proxy-store"))
    register_store(store)

    # Make the queues
    queues = PipeQueues(
        serialization_method="pickle",
        topics=["task"],
        proxystore_name="file",
        proxystore_threshold=10000,
    )

    # Define the parsl configuration (this can be done using the config_factory
    # for common use cases or by defining your own configuration.)
    parsl_config = cfg.compute_settings.config_factory(cfg.output_dir / "run-info")

    # Make output directory for tasks
    task_output_dir = cfg.output_dir / "tasks"
    task_output_dir.mkdir(exist_ok=True)

    # Assign constant settings to each task function
    my_run_task = partial(
        run_task,
        receptor_oedu_file=cfg.receptor_oedu_file,
        output_dir=task_output_dir,
        node_local_path=cfg.node_local_path,
    )
    update_wrapper(my_run_task, run_task)

    doer = ParslTaskServer([my_run_task], queues, parsl_config)

    # Read in SMILES input file into a list of SMILES strings (skip header)
    smiles_list = cfg.smiles_file.read_text().split("\n")[1:]
    smiles_batches = batch_data(smiles_list, cfg.smiles_batch_size)
    input_arguments = [(smiles_batch,) for smiles_batch in smiles_batches]

    assert not any(len(i) == 0 for i in input_arguments), "Empty input arguments"

    thinker = Thinker(
        queue=queues,
        input_arguments=input_arguments,
        result_dir=cfg.output_dir / "result",
        num_parallel_tasks=cfg.num_parallel_tasks,
    )
    logging.info("Created the task server and task generator")

    try:
        # Launch the servers
        doer.start()
        thinker.start()
        logging.info("Launched the servers")

        # Wait for the task generator to complete
        thinker.join()
        logging.info("Task generator has completed")
    finally:
        queues.send_kill_signal()

    # Wait for the task server to complete
    doer.join()

    # Clean up proxy store
    store.close()
