#!/usr/bin/env bash
# Reproducible build of the `mechinterp` conda env.
# Target machine: RTX 4090 (24GB), NVIDIA driver 535 -> CUDA 12.x.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n mechinterp python=3.11
conda activate mechinterp
pip install --upgrade pip

# IMPORTANT: driver 535 supports CUDA 12.x only. The default `pip install torch`
# pulls a cu130 (CUDA 13) wheel whose runtime the 535 driver cannot load
# (torch.cuda.is_available() == False). Pin the cu126 build instead.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r "$(dirname "$0")/../requirements.txt"

# EleutherAI top-k SAE. The PyPI name `sparsify` is a DIFFERENT (Neural Magic)
# package that fails to build; the real one is `eai-sparsify`, from git.
pip install "git+https://github.com/EleutherAI/sparsify.git"

# saprmarks dictionary_learning: the SAE trainer that accepts an arbitrary
# DataLoader over a [N, d] activation tensor (no LM/token plumbing required).
pip install "git+https://github.com/saprmarks/dictionary_learning.git"

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO-CUDA")
import transformer_lens, sae_lens, sparsify, nnsight, dictionary_learning, open_clip
print("interp toolchain imports OK")
PY
echo "Done. Activate with: conda activate mechinterp"
