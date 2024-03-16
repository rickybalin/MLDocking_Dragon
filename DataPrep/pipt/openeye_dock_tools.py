"""This module contains functions docking a molecule to a receptor using Openeye.

The code is adapted from this repository: https://github.com/inspiremd/Model-generation
"""
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from pathlib import Path
from typing import List, Optional

from openeye import oechem, oedocking, oeomega

from pipt.docking_utils import smi_to_structure
from pipt.utils import exception_handler


def from_mol(mol, isomer=True, num_enantiomers=1):
    """
    Generates a set of conformers as an OEMol object
    Inputs:
        mol is an OEMol
        isomers is a boolean controling whether or not the various diasteriomers of a molecule are created
        num_enantiomers is the allowable number of enantiomers. For all, set to -1
    """
    # Turn off the GPU for omega
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.GetTorDriveOptions().SetUseGPU(False)
    omega = oeomega.OEOmega(omegaOpts)

    out_conf = []
    if not isomer:
        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            out_conf.append(mol)
        else:
            oechem.OEThrow.Warning(
                "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
            )

    elif isomer:
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), 12, True):
            enantiomer = oechem.OEMol(enantiomer)
            ret_code = omega.Build(enantiomer)
            if ret_code == oeomega.OEOmegaReturnCode_Success:
                out_conf.append(enantiomer)
                num_enantiomers -= 1
                if num_enantiomers == 0:
                    break
            else:
                oechem.OEThrow.Warning(
                    "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
                )
    return out_conf


def from_string(smiles, isomer=True, num_enantiomers=1):
    """
    Generates an set of conformers from a SMILES string
    """
    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smiles):
        raise ValueError(f"SMILES invalid for string {smiles}")
    else:
        return from_mol(mol, isomer, num_enantiomers)


def from_structure(structure_file: Path) -> oechem.OEMol:
    """
    Generates an set of conformers from a SMILES string
    """
    mol = oechem.OEMol()
    ifs = oechem.oemolistream()
    if not ifs.open(str(structure_file)):
        raise ValueError(f"Could not open structure file: {structure_file}")

    if structure_file.suffix == ".pdb":
        oechem.OEReadPDBFile(ifs, mol)
    elif structure_file.suffix == ".sdf":
        oechem.OEReadMDLFile(ifs, mol)
    else:
        raise ValueError(f"Invalid structure file extension: {structure_file}")

    return mol


def select_enantiomer(mol_list):
    return mol_list[0]


def dock_conf(receptor, mol, max_poses: int = 1):
    dock = oedocking.OEDock()
    dock.Initialize(receptor)
    lig = oechem.OEMol()
    _ = dock.DockMultiConformerMolecule(lig, mol, max_poses)
    # the above line outputs the error
    return dock, lig


# Returns an array of length max_poses from above. This is the range of scores
def ligand_scores(dock, lig):
    return [dock.ScoreLigand(conf) for conf in lig.GetConfs()]


def best_dock_score(dock, lig):
    return ligand_scores(dock, lig)[0]


def write_ligand(ligand, output_path: Path) -> None:
    # TODO: If MAX_POSES != 1, we should select the top pose to save
    ofs = oechem.oemolostream()
    conf = list(ligand.GetConfs())[0]
    if ofs.open(str(output_path)):
        oechem.OEWriteMolecule(ofs, conf)
        ofs.close()
        return
    raise ValueError(f"Could not write ligand to {output_path}")


def write_receptor(receptor, output_path: Path) -> None:
    ofs = oechem.oemolostream()
    if ofs.open(str(output_path)):
        mol = oechem.OEMol()
        receptor.GetComponentsWithin(mol)
        oechem.OEWriteMolecule(ofs, mol)
        ofs.close()
    raise ValueError(f"Could not write receptor to {output_path}")


@cache  # Only read the receptor once
def read_receptor(receptor_oedu_file: Path):
    """Read the .oedu file into a GraphMol object."""
    receptor = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(str(receptor_oedu_file), receptor)
    return receptor


@exception_handler(default_return=0.0)
def run_docking(
    smiles: str, receptor_oedu_file: Path, temp_storage: Optional[Path] = None
) -> float:
    """Run OpenEye docking on a single ligand.

    Parameters
    ----------
    smiles : ste
        A single SMILES string.
    receptor_oedu_file : Path
        Path to the receptor .oedu file.
    temp_storage : Path
        Path to the temporary storage directory to write structures to,
        if None, use the current working Python's built in temp storage.

    Returns
    -------
    float
        The docking score of the best conformer.
    """

    with tempfile.NamedTemporaryFile(suffix=".pdb", dir=temp_storage) as fd:
        # Read input SMILES and generate conformer
        smi_to_structure(smiles, Path(fd.name))
        conformer = from_structure(Path(fd.name))

    # Read the receptor to dock to
    receptor = read_receptor(receptor_oedu_file)

    # Dock the ligand conformers to the receptor
    dock, lig = dock_conf(receptor, conformer, max_poses=1)

    # Get the best docking score (only one to consider)
    best_score = best_dock_score(dock, lig)

    return best_score


# def run_docking_old(smiles: str, receptor_oedu_file: Path) -> float:
#     """Run OpenEye docking on a single ligand.

#     Parameters
#     ----------
#     smiles : ste
#         A single SMILES string.
#     receptor_oedu_file : Path
#         Path to the receptor .oedu file.
#     output_path : Path
#         Path to the output directory to write `metrics.csv`
#         containing the best docking score and `ligand.pdb`
#         containing the best pose.

#     Returns
#     -------
#     float
#         The docking score of the best conformer.
#     """
#     # Create the output directory
#     # output_path.mkdir(exist_ok=True, parents=True)

#     start = time.time()
#     # A list of enantiomers
#     # conformers = select_enantiomer(from_string(smiles))
#     try:
#         conformers = select_enantiomer(from_string(smiles))
#     except IndexError:
#         out_pdbfile = Path("temp.pdb")
#         smi_to_structure(smiles, out_pdbfile)
#         conformers = from_structure(out_pdbfile)

#     print(f"Reading conformers took {time.time() - start} seconds")

#     start = time.time()
#     receptor = read_receptor(receptor_oedu_file)
#     print(f"Reading receptor took {time.time() - start} seconds")

#     start = time.time()
#     # Dock the ligand conformers to the receptor
#     dock, lig = dock_conf(receptor, conformers, max_poses=1)
#     print(f"Docking took {time.time() - start} seconds")

#     # Currently we generate 200 conformers for each ligand, but only take
#     #   the best pose, as scored by Openeye. It may be useful to consider
#     #   something about the range of poses.

#     start = time.time()
#     best_score = best_dock_score(dock, lig)
#     print(best_score)
#     print(f"Scoring took {time.time() - start} seconds")

#     # write_ligand(lig, output_path / "ligand.pdb")
#     # write_receptor(receptor, f"{output_path}/apo.pdb")

#     return best_score


def run_parallel_docking(
    smiles_batch: List[str],
    receptor_oedu_file: Path,
    num_workers: int,
    temp_storage: Optional[Path] = None,
) -> List[float]:
    # Run the docking computation
    worker_fn = partial(
        run_docking, receptor_oedu_file=receptor_oedu_file, temp_storage=temp_storage
    )
    docking_scores = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for score in pool.map(worker_fn, smiles_batch):
            docking_scores.append(score)
    return docking_scores


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--receptor", type=Path, required=True, help="Receptor .oedu file"
    )
    parser.add_argument(
        "-s", "--smiles", type=Path, required=True, help="Ligand SMILES .smi file"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output .csv file"
    )
    parser.add_argument("-t", "--storage", type=Path, default=None)
    parser.add_argument("-n", "--num_workers", type=int, default=1)
    args = parser.parse_args()

    smiles_list = args.smiles.read_text().split("\n")

    docking_scores = run_parallel_docking(
        smiles_list, args.receptor, args.num_workers, args.storage
    )

    # Format the output file
    file_contents = "SMILES, DockingScore\n"
    file_contents += "\n".join(
        f"{smiles},{score}" for smiles, score in zip(smiles_list, docking_scores)
    )
    file_contents += "\n"

    # Write the output file
    with open(args.output, "w") as f:
        f.write(file_contents)


# TODO: We should use 30 conformers
