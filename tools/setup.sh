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
git checkout method1_filter_stopwords
echo "Installed gh repo"

echo "Installing conda"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INST="/tmp/miniconda.sh"
wget -qO "$INST" "$URL"
bash "$INST" -b -p "$CONDA_DIR"
rm -f "$INST"
"$CONDA_DIR/bin/conda" init bash >/dev/null
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
conda --version
echo "Conda installed"

echo "Installing conda env"
conda env create -f "$BASE_PATH/phd-ir-methods/spar_k_means_bert/conda_env.yml"
conda activate spar-k-means-bert
echo "Conda env installed and activated"

echo "Preparing PyLucene"
cd "$BASE_PATH/phd-ir-methods/tools"
tar -xzf pylucene.tar.gz
tar -xzf jcc.tar.gz
cp jcc/libjcc3.so "$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/jvm/java-21-amazon-corretto/lib/server/"
echo "Prepared PyLucene"

echo "Downloading Vector Distionary"
aws s3 cp s3://spark-bert-vector-dictionary/uploads/hnsw.index "$BASE_PATH/phd-ir-methods/hnsw.index"
aws s3 cp s3://spark-bert-vector-dictionary/uploads/faiss_idx_to_token.pickle "$BASE_PATH/phd-ir-methods/faiss_idx_to_token.pickle"
echo "Downloaded Vector Distionary"

echo "Running SparkBERT"
cd "$BASE_PATH/phd-ir-methods"
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/jvm/java-21-amazon-corretto/lib/server/"
python -m spar_k_means_bert.run --use-cache -d msmarco
echo "Finished"
