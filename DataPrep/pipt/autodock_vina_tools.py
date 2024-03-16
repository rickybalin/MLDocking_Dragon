import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import pipt
from pipt.openeye_dock_tools import smi_to_structure


def set_element(input_pdb_file: Path, output_pdb_file: Path) -> None:
    """Set the element of each atom in a PDB file.

    Parameters
    ----------
    input_pdb_file : Path
        Input PDB file.
    output_pdb_file : Path
        Output PDB file.
    """
    tcl_script = Path(pipt.__file__).parent / "tcl_utils" / "set_element.tcl"
    command = (
        f"vmd -dispdev text -e {tcl_script} -args {input_pdb_file} {output_pdb_file}"
    )
    subprocess.run(command.split())


def pdb_to_pdbqt(
    pdb_file: Path, pdbqt_file: Path, autodocktools_path: Path, ligand: bool = True
) -> None:
    """Convert a PDB file to a PDBQT file for a receptor.

    Parameters
    ----------
    pdb_file : Path
        Input PDB file.
    pdbqt_file : Path
        Output PDBQT file.
    autodocktools_path : Path
        Path to AutoDockTools folder.
    ligand : bool, optional
        Whether the PDB file is a ligand or receptor, by default True.
    """
    # Select the correct settings for ligand or receptor preparation
    script, flag = (
        ("prepare_ligand4.py", "l") if ligand else ("prepare_receptor4.py", "r")
    )

    command = (
        f"{autodocktools_path / 'MGLTools-1.5.6/bin/pythonsh'}"
        f" {autodocktools_path / 'MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24' / script}"
        f" -{flag} {pdb_file}"
        f" -o {pdbqt_file}"
        f" -U nphs_lps_waters"
    )
    subprocess.run(command.split())


def make_autodock_vina_config(
    input_receptor_pdbqt_file: Path,
    input_ligand_pdbqt_file: Path,
    output_conf_file: Path,
    output_ligand_pdbqt_file: Path,
    output_log_file: Path,
    center: Tuple[float, float, float],
    size: Tuple[int, int, int],
    exhaustiveness: int = 20,
    num_modes: int = 20,
    energy_range: int = 10,
) -> None:
    """Make a configuration file for AutoDock Vina.

    Parameters
    ----------
    input_receptor_pdbqt_file : Path
        Input receptor PDBQT file.
    input_ligand_pdbqt_file : Path
        Input ligand PDBQT file.
    output_conf_file : Path
        Output configuration file.
    output_ligand_pdbqt_file : Path
        Output ligand PDBQT file.
    output_log_file : Path
        Output log file.
    center : Tuple[float, float, float]
        Center of the search space.
    size : Tuple[int, int, int]
        Size of the search space.
    exhaustiveness : int, optional
        Exhaustiveness of the search, by default 20.
    num_modes : int, optional
        Number of binding modes to generate, by default 20.
    energy_range : int, optional
        Energy range, by default 10.
    """

    # Format configuration file
    file_contents = (
        f"receptor = {input_receptor_pdbqt_file}\n"
        f"ligand = {input_ligand_pdbqt_file}\n"
        f"center_x = {center[0]}\n"
        f"center_y = {center[1]}\n"
        f"center_z = {center[2]}\n"
        f"size_x = {size[0]}\n"
        f"size_y = {size[1]}\n"
        f"size_z = {size[2]}\n"
        f"exhaustiveness = {exhaustiveness}\n"
        f"num_modes = {num_modes}\n"
        f"energy_range = {energy_range}\n"
        f"out = {output_ligand_pdbqt_file}\n"
        f"log = {output_log_file}\n"
    )

    # Write configuration file
    with open(output_conf_file, "w") as f:
        f.write(file_contents)


def run_autodock_vina(
    autodock_vina_exe: Path,
    config_file: Path,
    output_ligand_pdbqt_file: Path,
    num_cpu: int = 1,
) -> float:
    """Run AutoDock Vina docking.

    Parameters
    ----------
    autodock_vina_exe : Path
        Path to AutoDock Vina executable.
    config_file : Path
        Path to AutoDock Vina configuration file.
    output_ligand_pdbqt_file: Path
        Output ligand PDBQT file from the docking.
    num_cpu : int, optional
        Number of CPUs to use, by default 1.

    Returns
    -------
    float : The docking score.
    """

    # Dock the ligand to a receptor
    command = f"{autodock_vina_exe} --config {config_file} --cpu {num_cpu}"
    subprocess.run(command.split())

    # Parse the docking score
    lines = output_ligand_pdbqt_file.read_text().split("\n")
    result_line = next(filter(lambda x: "REMARK VINA RESULT:" in x, lines))
    score = float(result_line.split()[3])
    return score


def run_docking(
    smiles: str,
    input_receptor_pdbqt_file: Path,
    autodocktools_path: Path,
    autodock_vina_exe: Path,
    center: Tuple[float, float, float],
    size: Tuple[int, int, int],
    num_cpu: int = 1,
) -> float:
    """Run AutoDock Vina docking."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set temporary file paths
        pdb_file = Path(tmpdir) / "ligand.pdb"
        input_pdbqt_file = Path(tmpdir) / "ligand_input.pdbqt"
        output_pdbqt_file = Path(tmpdir) / "ligand_out.pdbqt"
        config_file = Path(tmpdir) / "dock.inp"
        log_file = Path(tmpdir) / "dock.log"

        # Run the autodock vina workflow
        smi_to_structure(smiles, pdb_file)
        set_element(pdb_file, pdb_file)
        pdb_to_pdbqt(pdb_file, input_pdbqt_file, autodocktools_path)
        make_autodock_vina_config(
            input_receptor_pdbqt_file=input_receptor_pdbqt_file,
            input_ligand_pdbqt_file=input_pdbqt_file,
            output_conf_file=config_file,
            output_ligand_pdbqt_file=output_pdbqt_file,
            output_log_file=log_file,
            center=center,
            size=size,
        )
        score = run_autodock_vina(
            autodock_vina_exe=autodock_vina_exe,
            config_file=config_file,
            output_ligand_pdbqt_file=output_pdbqt_file,
            num_cpu=num_cpu,
        )

    return score
