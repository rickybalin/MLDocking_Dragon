"""This module contains functions docking a molecule to a receptor using Openeye.

The code is adapted from this repository: https://github.com/inspiremd/Model-generation
"""
import MDAnalysis as mda
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from pathlib import Path
from typing import List, Optional
import numpy as np
from openeye import oechem, oedocking, oeomega
import pandas as pd
from tqdm import tqdm
#from docking_utils import smi_to_structure
#from utils import exception_handler
import os
import time
from time import perf_counter
import datetime

import dragon
import multiprocessing as mp
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from typing import Any, Optional, Type, TypeVar, Union

import yaml
from pydantic_settings import BaseSettings as _BaseSettings
from pydantic import validator

_T = TypeVar("_T")

PathLike = Union[Path, str]

def exception_handler(default_return: Any = None):
    """Handle exceptions in a function by returning a `default_return` value."""

    def decorator(func):
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(
                    f"{func.__name__} raised an exception: {e} "
                    f"On input {args}, {kwargs}\nReturning {default_return}"
                )
                return default_return

        return wrapper

    return decorator


def _resolve_path_exists(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    p = value.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _resolve_mkdir(value: Path) -> Path:
    p = value.resolve()
    p.mkdir(exist_ok=True, parents=True)
    return p


def path_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_path_exists)
    return _validator


def mkdir_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_mkdir)
    return _validator


class BaseSettings(_BaseSettings):
    """Base settings to provide an easier interface to read/write YAML files."""

    def dump_yaml(self, filename: PathLike) -> None:
        with open(filename, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)  # type: ignore

'''
Functions
'''

def smi_to_structure(smiles: str, output_file: Path, forcefield: str = "mmff") -> None:
    """Convert a SMILES file to a structure file.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    output_file : Path
        EIther an output PDB file or output SDF file.
    forcefield : str, optional
        Forcefield to use for 3D conformation generation
        (either "mmff" or "etkdg"), by default "mmff".
    """
    # Convert SMILES to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)

    # Generate a 3D conformation for the molecule
    if forcefield == "mmff":
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
    elif forcefield == "etkdg":
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    else:
        raise ValueError(f"Unknown forcefield: {forcefield}")

    # Write the molecule to a file
    if output_file.suffix == ".pdb":
        writer = Chem.PDBWriter(str(output_file))
    elif output_file.suffix == ".sdf":
        writer = Chem.SDWriter(str(output_file))
    else:
        raise ValueError(f"Invalid output file extension: {output_file}")
    writer.write(mol)
    writer.close()



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
    err = dock.DockMultiConformerMolecule(lig, mol, max_poses)
    return dock, lig

# Returns an array of length max_poses from above. This is the range of scores
def ligand_scores(dock, lig):
    return [dock.ScoreLigand(conf) for conf in lig.GetConfs()]


def best_dock_score(dock, lig):
    return ligand_scores(dock, lig)#[0]


def write_ligand(ligand, output_dir: Path, smiles: str, lig_identify: str) -> None:
    # TODO: If MAX_POSES != 1, we should select the top pose to save
    ofs = oechem.oemolostream()
    for it, conf in enumerate(list(ligand.GetConfs())):
        if ofs.open(f'{str(output_dir)}/{lig_identify}/{it}.pdb'):
            oechem.OEWriteMolecule(ofs, conf)
            ofs.close()
    return
    raise ValueError(f"Could not write ligand to {output_path}")


def write_receptor(receptor, output_path: Path) -> None:
    ofs = oechem.oemolostream()
    if ofs.open(str(output_path)):
        mol = oechem.OEMol()
        contents = receptor.GetComponents(mol)#Within
        oechem.OEWriteMolecule(ofs, mol)
        ofs.close()
    return
    raise ValueError(f"Could not write receptor to {output_path}")

def run_test(smiles):
    return -1


@cache  # Only read the receptor once
def read_receptor(receptor_oedu_file: Path):
    """Read the .oedu file into a GraphMol object."""
    receptor = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(str(receptor_oedu_file), receptor)
    return receptor

#@cache
def create_proteinuniv(protein_pdb):
    protein_universe = mda.Universe(protein_pdb)
    return protein_universe

def create_complex(protein_universe, ligand_pdb):
    u1 = protein_universe
    u2 = mda.Universe(ligand_pdb)
    u = mda.core.universe.Merge(u1.select_atoms("all"), u2.atoms)#, u3.atoms)
    return u

def create_trajectory(protein_universe, ligand_dir, output_pdb_name, output_dcd_name):
    import MDAnalysis as mda
    ligand_files = sorted(os.listdir(ligand_dir))
    comb_univ_1 = create_complex(protein_universe, f'{ligand_dir}/{ligand_files[0]}').select_atoms("all")

    with mda.Writer(output_pdb_name, comb_univ_1.n_atoms) as w:
        w.write(comb_univ_1)
    with mda.Writer(output_dcd_name, comb_univ_1.n_atoms,) as w:
        for it, ligand_file in enumerate(ligand_files):
            comb_univ = create_complex(protein_universe, f'{ligand_dir}/{ligand_file}') 
            w.write(comb_univ)    # write a whole universe
            os.remove(f'{ligand_dir}/{ligand_file}')
    return


def docking_switch(cdd, num_procs, proc, continue_event):
    #for i in range(1):
    iter = 0
    if proc ==0:
        with open("docking_switch.log",'a') as f:
            f.write(f"{datetime.datetime.now()}: Starting Docking\n")
    last_top_candidate_list = "-1"
    
    continue_flag = True
    while continue_flag:

        ckeys = cdd.keys()
        # ckeys = [ckey for ckey in ckeys if "iter" not in ckey and ckey[0] != "d"]
        # ckey_max = max(ckeys)
        if "max_sort_iter" in ckeys:
            ckey_max = cdd["max_sort_iter"]
        else:
            ckey_max = '-1'
            
        # Only run new simulations if there is a fresh candidate list
        if ckey_max > last_top_candidate_list:
            ckeys.sort()
            
            if proc == 0:
                with open("docking_switch.log","a") as f:
                    f.write(f"{datetime.datetime.now()}: Docking on iter {iter} with candidate list {ckey_max}\n")
            # most recent sorted list
            top_candidates = cdd[ckey_max]["smiles"]
            num_candidates = len(top_candidates)

            if proc == 0:
                with open("docking_switch.log","a") as f:
                    f.write(f"iter {iter}: found {num_candidates} candidates to filter \n")

            # Partition top candidate list to get candidates for this process to simulate
            sims_per_proc = num_candidates//num_procs + 1
            my_candidates = top_candidates[proc*sims_per_proc:min((proc+1)*sims_per_proc,num_candidates)]

            with open("docking_switch.log","a") as f:
                f.write(f"iter {iter}: proc {proc}: found {len(my_candidates)} candidates to filter on proc \n")

            # check to see if we already have a sim result for this process' candidates
            if len(my_candidates) > 0:
                my_candidates = filter_candidates(cdd, my_candidates)

            # if there are new candidates to simulate, run sims
            if len(my_candidates) > 0:
                tic = perf_counter()
                with open("docking_switch.log","a") as f:
                    f.write(f"{iter} iter: simulating {len(my_candidates)} on proc {proc}\n")
                time_per_cand = run_docking(cdd, my_candidates, f"dock_iter{iter}_proc{proc}", proc)
                #time_per_cand =999
                if proc == 0:
                    cdd["docking_iter"] = iter
                toc = perf_counter()
                if proc == 0:
                    with open("docking_switch.log","a") as f:
                        f.write(f"{datetime.datetime.now()}: iter {iter}: docking sim time {toc-tic} s \n")
                    print(f"{cdd.keys()=}")
                last_top_candidate_list = ckey_max
                
                iter += 1
            else:
                if proc == 0:
                    with open("docking_switch.log","a") as f:
                        f.write(f"{datetime.datetime.now()}: iter {iter}: no sims run \n")
        else:
            if proc == 0:
                with open("docking_switch.log","a") as f:
                    f.write(f"{datetime.datetime.now()}: iter {iter}: no valid list {ckey_max} \n")
        if continue_event is None:
            continue_flag = False
        else:
            continue_flag = continue_event.is_set()   
            


def filter_candidates(cdd, candidates: list):

    # Get keys that store previous docking results
    ckeys = cdd.keys()
    cbkeys = [ckey for ckey in ckeys if ckey[0] == "d"]

    for cbk in cbkeys:
        check_smiles = cdd[cbk]["smiles"]
        if len(candidates) > 0:
            for c in candidates:
                if c in check_smiles:
                    candidates.remove(c)
        else:
            # Don't continue to query keys if there are no candidates left
            break
    return candidates


def run_docking(cdd, candidates, batch_key, proc):
    """Run OpenEye docking on a single ligand.
    
    Parameters
    ----------
    smiles : ste
        A single SMILES string.
    receptor_oedu_file : Path
        Path to the receptor .oedu file.
    max_confs : int
        Number of ligand poses to generate
    temp_storage : Path
        Path to the temporary storage directory to write structures to,
        if None, use the current working Python's built in temp storage.
    
    Returns
    -------
    float
        The docking score of the best conformer.
    """

    debug == True
    
    num_candidates = len(candidates)

    tic = perf_counter()
    temp_storage = "./docking_tmp/"
    hostname = os.popen("hostname -f").read()

    receptor_oedu_file = "/lus/grand/projects/hpe_dragon_collab/avasan/3clpro_7bqy.oedu"
    if "sirius" in hostname:
        receptor_oedu_file = "/home/csimpson/openeye/3clpro_7bqy.oedu"
    max_confs = 1

    simulated_smiles = []
    dock_scores = []

    if debug:
        with open(f"dock_worker_{proc}.log","a") as f:
            f.write(f"Simulating on proc {proc}\n")
    
    for smiles in candidates:
        try:
            try:
                conformers = select_enantiomer(from_string(smiles))
            except:
                print(f"Conformers failed in batch {batch_key}, returning 0 docking score", flush=True)
                simulated_smiles.append(smiles)
                dock_scores.append(0)
                continue

                # Not implementing this alternate way of getting conformers for now
                # with tempfile.NamedTemporaryFile(suffix=".pdb", dir=temp_storage) as fd:
                #     # Read input SMILES and generate conformer
                #     smi_to_structure(smiles, Path(fd.name))
                #     conformers = from_structure(Path(fd.name))

            # Read the receptor to dock to
            receptor = read_receptor(receptor_oedu_file)
            print(receptor)
            # Dock the ligand conformers to the receptor
            dock, lig = dock_conf(receptor, conformers, max_poses=max_confs)

            # Get the docking scores
            best_score = best_dock_score(dock, lig)

            simulated_smiles.append(smiles)
            dock_score = max(-1*np.mean(best_score),0.)
            dock_scores.append(dock_score)
        except:
            simulated_smiles.append(smiles)
            dock_scores.append(0)
    toc = perf_counter()
    time_per_cand = (toc-tic)/num_candidates
    if debug:
        with open(f"dock_worker_{proc}.log","a") as f:
            f.write(f"Simulated {num_candidates} in {toc-tic} s, {time_per_cand=}\n")
    cdd[batch_key] = {"smiles": simulated_smiles, "docking_scores": dock_scores}
    return time_per_cand

