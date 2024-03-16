# Parallelized Molecular Docking 
## Scripts written by Alexander Brace, Archit Vasan, and Ozan Gokdemir

## Table of Contents
- [PIPT](#pipt)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Installation on Polaris](#installation-on-polaris)
    - [Installation on Lambda](#installation-on-lambda)
  - [Usage](#usage)

## Installation

If you are using the OpenEye docking method, you need to obtain a copy of the license.
Once you have a license file, run:
```bash
cp /path/to/oe_license.txt ~/.oe_license.txt
echo "export OE_LICENSE=~/.oe_license.txt" >> ~/.bashrc
```

### Installation on Polaris
```bash
qsub -I -l select=1 -l walltime=1:00:00 -A RL-fold -q debug -l filesystems=home:eagle
module load conda/2023-01-10-unstable
conda activate
conda create -n pipt --clone base
conda activate pipt
conda install -c openeye openeye-toolkits -y
make install
```
## Usage

In order to dock ligands to a receptor with OpenEye, some setup is required to generate an `.oedu`
file for the receptor. [These](https://docs.eyesopen.com/applications/oedocking/make_receptor/make_receptor_setup.html) intructions provide a helpful guide.

To convert a smiles file to a PDB file:
```bash
python -m pipt.docking -s tests/data/test_smile.dat -p tests/data/test_smile.pdb
```

To test the OpenEye docking method:
```bash
python -m pipt.cli -r data/8gcy_receptor.oedu -l tests/data/test_smile.dat -o tests/data/output
```

To run the high-throughput docking workflow:
```bash
nohup python -m pipt.workflow -c examples/polaris_prod.yaml &
```

To concatenate docking csv files:
```bash
python -m pipt.cli -i prod_output/tasks -o SMILES_2M.csv
```
