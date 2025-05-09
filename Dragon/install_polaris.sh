#!/bin/bash

# Load modules
module use /soft/modulefiles
module load conda
conda activate base

# Install dragon in virtual environment
python -m venv _dragon_env --system-site-packages
source _dragon_env/bin/activate
pip install dragonhpc
dragon-config -a "ofi-runtime-lib=/opt/cray/libfabric/1.15.2.0/lib64"

# Install other packages
pip install -i https://pypi.anaconda.org/OpenEye/simple OpenEye-toolkits
pip install SmilesPE
pip install MDAnalysis rdkit
pip install pydantic_settings
