#!/usr/bin/env bash
# Download the datasets used by HyperFold and precompute the graph cache.
#
# Sources (see DATASETS.md):
#   - UniSite-DS (training/eval pkl files + PDB structures) and the
#     HOLO4K-sc / COACH420 benchmarks are distributed together at
#     https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1
#   - The benchmark structures themselves originate from
#     https://github.com/rdk/p2rank-datasets (mlig subsets, single-chain).
#
# EC / fold datasets (for scripts/train_ec.py / train_fold.py) are NOT part
# of the release above and require a manual step:
#   - EC: EnzymeCommission dataset of GearNet
#     (https://github.com/DeepGraphLearning/GearNet); place the preprocessed
#     split under data/EnzymeCommission_cdconv (see configs/ec.yaml).
#   - Fold: fold-classification dataset (data/fold3d/new_fold3d, see
#     configs/fold.yaml); derive it from the standard SCOPe-based fold
#     classification splits (DeepFRI-style fold dataset).
set -euo pipefail

DATASETS_DIR="${1:-datasets}"
HF_REPO="https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1/resolve/main"
mkdir -p "${DATASETS_DIR}"

download() {  # download <remote-file> <local-path>
    if [ -f "$2" ]; then
        echo "[skip] $2 already exists"
    else
        echo "[download] $1 -> $2"
        wget -c --show-progress "${HF_REPO}/$1" -O "$2"
    fi
}

echo "=== UniSite-DS (pkl files + structures, ~3.8 GB) ==="
download "unisite-ds-v1.tar.gz" "${DATASETS_DIR}/unisite-ds-v1.tar.gz"
if [ -d "${DATASETS_DIR}/unisite-ds/pkl_files" ]; then
    echo "[skip] ${DATASETS_DIR}/unisite-ds already extracted"
else
    echo "[extract] unisite-ds-v1.tar.gz"
    tar -xzf "${DATASETS_DIR}/unisite-ds-v1.tar.gz" -C "${DATASETS_DIR}"
fi

echo "=== Benchmark datasets (HOLO4K-sc / COACH420, ~100 MB) ==="
download "benchmark_datasets.tar.gz" "${DATASETS_DIR}/benchmark_datasets.tar.gz"
if [ -d "${DATASETS_DIR}/benchmark_datasets/holo4k-sc" ]; then
    echo "[skip] ${DATASETS_DIR}/benchmark_datasets already extracted"
else
    echo "[extract] benchmark_datasets.tar.gz"
    tar -xzf "${DATASETS_DIR}/benchmark_datasets.tar.gz" -C "${DATASETS_DIR}"
fi

# Reference predictions of the released model (optional, for evaluation
# sanity checks; see EVALUATION.md). Uncomment to fetch:
# download "unisite_results.tar.gz" "${DATASETS_DIR}/unisite_results.tar.gz"
# tar -xzf "${DATASETS_DIR}/unisite_results.tar.gz" -C "${DATASETS_DIR}"

echo "=== Precompute residue graph cache (rdkit, C-alpha graphs) ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/precompute_graphs.py" \
    --pkl_dir "${DATASETS_DIR}/unisite-ds/pkl_files" \
    --graph_cache "${DATASETS_DIR}/unisite-ds/graph_cache_v2"

echo "ALL_DONE: datasets ready under ${DATASETS_DIR}/"
