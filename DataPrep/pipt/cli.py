def scaffold_based_sampling_cli() -> None:
    from argparse import ArgumentParser
    from pathlib import Path

    from pipt.sampling import scaffold_based_sampling

    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--molecules", type=Path, required=True, help="Ligand SMILES .dat file"
    )
    parser.add_argument("-s", "--sample_size", type=int, required=True)
    parser.add_argument("-o", "--outpath", type=Path, required=True)
    args = parser.parse_args()

    # Each line contains a SMILES string, except the first line which is a header
    molecules = args.ligand.read_text().split("\n")[1:]

    scaffold_based_sampling(molecules, args.sample_size, args.outpath)


def run_docking_cli() -> None:
    from argparse import ArgumentParser
    from pathlib import Path

    from pipt.openeye_dock_tools import run_docking

    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--receptor", type=Path, required=True, help="Receptor .oedu file"
    )
    parser.add_argument(
        "-l", "--ligand", type=Path, required=True, help="Ligand SMILES .smi file"
    )
    args = parser.parse_args()

    # If there are multiple SMILES in the file, only use the first one
    ligand = args.ligand.read_text().split("\n")[0]

    print("SMILES:", ligand)

    run_docking(ligand, args.receptor)


def run_autodock_vina() -> None:
    from argparse import ArgumentParser
    from pathlib import Path

    from pipt.autodock_vina_tools import run_docking

    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--receptor", type=Path, required=True, help="Receptor .pdbqt file"
    )
    parser.add_argument(
        "-l", "--ligand", type=Path, required=True, help="Ligand SMILES .smi file"
    )
    parser.add_argument(
        "-a",
        "--autodocktools_path",
        type=Path,
        required=True,
        help="Path to autodocktools",
    )
    parser.add_argument(
        "-e",
        "--autodock_vina_exe",
        type=Path,
        required=True,
        help="Path to autodock_vina",
    )
    parser.add_argument(
        "-n", "--num_cpu", type=int, required=True, help="Number of CPUs to use"
    )
    args = parser.parse_args()

    # If there are multiple SMILES in the file, only use the first one
    ligand = args.ligand.read_text().split("\n")[0]

    # TODO: Add CLI option
    center = (3.108, 8.092, 17.345)
    size = (40, 54, 44)

    run_docking(
        ligand,
        args.receptor,
        args.autodocktools_path,
        args.autodock_vina_exe,
        center,
        size,
        args.num_cpu,
    )


def concatenate_csv_files() -> None:
    from argparse import ArgumentParser
    from pathlib import Path

    import pandas as pd

    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_file", type=Path, required=True)
    args = parser.parse_args()

    # Read each CSV file into a DataFrame and store them in a list
    dfs = [pd.read_csv(p) for p in args.input_dir.glob("*.csv")]

    # Concatenate the DataFrames together
    combined_df = pd.concat(dfs, ignore_index=True)

    # Write the combined DataFrame to a new CSV file
    combined_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    concatenate_csv_files()
