import random
from collections import defaultdict
from pathlib import Path
from typing import List

from rdkit.Chem import MolToSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


def scaffold_based_sampling(
    molecules: List[str], sample_size: int, output_file: Path
) -> None:
    # Store a list of molecule indices for each scaffold
    scaffolds = defaultdict(list)
    for i, mol in enumerate(molecules):
        # Gets the Murcko scaffold of mol as a SMILES string
        scaffold = MolToSmiles(
            GetScaffoldForMol(mol)
        )  # TODO: See if Scaffold is hashable
        scaffolds[scaffold].append(i)

    # Select a set of molecules from each scaffold
    samples_per_scaffold = sample_size // len(scaffolds)
    samples = []
    for scaffold, mols in scaffolds.items():
        # Randomly select a molecule from the scaffold
        samples.extend(random.choices(mols, k=samples_per_scaffold))

    # Write each sampled molecule to a SMILES file with one molecule per line
    file_contents = "\n".join(MolToSmiles(molecules[i]) for i in samples)
    with open(output_file, "w") as f:
        f.write(file_contents)
