if __name__ == "__main__":
    from argparse import ArgumentParser

    from pipt.autodock_vina_tools import smi_to_structure

    parser = ArgumentParser()
    parser.add_argument("-s", "--smiles", required=True)
    parser.add_argument("-p", "--pdb", required=True)
    parser.add_argument("-f", "--forcefield", default="mmff")
    args = parser.parse_args()

    # Read the SMILES file
    with open(args.smiles, "r") as f:
        smiles = f.read().strip()

    smi_to_structure(smiles, args.pdb, args.forcefield)
