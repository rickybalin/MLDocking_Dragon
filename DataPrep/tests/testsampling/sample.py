from rdkit.Chem import MolToSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
import random
from collections import defaultdict
from pathlib import Path


def scaffold_based_sampling_v2(molecules, sample_size: int, output_file: Path):
    # Store a list of molecule indices for each scaffold
    scaffolds = defaultdict(list)
    for i, mol in enumerate(molecules):
        # Gets the Murcko scaffold of mol as a SMILES string
        scaffold = MolToSmiles(GetScaffoldForMol(mol)) # TODO: See if Scaffold is hashable
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


    
def scaffold_based_sampling(molecules, sample_size):
    # Extract the Murcko scaffolds from the molecules
    scaffolds = {}#$set()
    for mol in molecules:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffolds[scaffold].append()
        scaffolds.add(Chem.MolToSmiles(scaffold))

    # Randomly select scaffolds for sampling
    selected_scaffolds = random.sample(scaffolds, sample_size)

    # Collect compounds that belong to the selected scaffolds
    sampled_compounds = []
    for mol in molecules:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if Chem.MolToSmiles(scaffold) in selected_scaffolds:
            sampled_compounds.append(mol)

    return sampled_compounds

# Load the Enamine dataset from a SMILES file
smiles_file = 'enamine_dataset.smi'
molecules = [Chem.MolFromSmiles(line.strip()) for line in open(smiles_file)]

# Set the desired sample size
sample_size = 100  # adjust this according to your requirements

# Perform scaffold-based sampling
sampled_compounds = scaffold_based_sampling(molecules, sample_size)

# Print the sampled compounds
for compound in sampled_compounds:
    print(Chem.MolToSmiles(compound))

