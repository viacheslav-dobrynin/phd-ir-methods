#!/usr/bin/env bash
set -e

BASE_PATH=/opt/dlami/nvme

echo "Changing location"
cd "$BASE_PATH"
echo "Changed location"

echo "Installing gh repo"
sudo apt install gh
gh auth login
git clone https://github.com/viacheslav-dobrynin/phd-ir-methods
cd phd-ir-methods
git checkout method2_multimodality
echo "Installed gh repo"

echo "Installing conda"
CONDA_DIR="${CONDA_DIR:-$BASE_PATH/miniconda3}"
URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
mkdir -p "$BASE_PATH/tmp"
INST="$BASE_PATH/tmp/miniconda.sh"
wget -qO "$INST" "$URL"
bash "$INST" -b -p "$CONDA_DIR"
rm -f "$INST"
"$CONDA_DIR/bin/conda" init bash >/dev/null
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
conda --version
echo "Conda installed"

echo "Installing conda env"
conda env create -f "$BASE_PATH/phd-ir-methods/sparsifier_model/conda_env.yml"
conda activate sparsifier-model
echo "Conda env installed and activated"

echo "Preparing PyLucene"
cd "$BASE_PATH/phd-ir-methods/tools"
tar -xzf pylucene.tar.gz
tar -xzf jcc.tar.gz
cp jcc/libjcc3.so "$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/jvm/java-21-amazon-corretto/lib/server/"
echo "Prepared PyLucene"

echo "Running SparkBERT"
cd "$BASE_PATH/phd-ir-methods"
python -m sparsifier_model.run
echo "Finished"
