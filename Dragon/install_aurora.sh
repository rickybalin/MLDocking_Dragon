#!/bin/bash

WDIR=$PWD
DRAGON_DIR=$1

# Load modules
module load frameworks

# Create venv
cd $DRAGON_DIR
python -m venv --clear _dragon_env --system-site-packages
. _dragon_env/bin/activate
pip install --force-reinstall dragon*.whl

# Install other packages
pip install -i https://pypi.anaconda.org/OpenEye/simple OpenEye-toolkits
pip install SmilesPE
pip install MDAnalysis rdkit
pip install pydantic_settings
