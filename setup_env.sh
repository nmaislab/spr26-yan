#!/bin/bash
# setup_env.sh – One-time environment setup for FedLLM-Guard experiments
# Run: bash setup_env.sh
# Then activate: conda activate fedllm

set -e
CONDA=/opt/miniconda3/bin/conda
ENV_NAME=fedllm

echo "=== Creating conda environment: $ENV_NAME ==="
$CONDA create -n $ENV_NAME python=3.12 -y

echo "=== Installing PyTorch (CUDA 12.x for RTX 4090) ==="
$CONDA run -n $ENV_NAME pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing Flower and FL dependencies ==="
$CONDA run -n $ENV_NAME pip install \
    "flwr[simulation]==1.9.0" \
    "flwr-datasets[vision]==0.1.0" \
    ray

echo "=== Installing experiment utilities ==="
$CONDA run -n $ENV_NAME pip install \
    matplotlib \
    scipy \
    pandas \
    requests \
    tqdm

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source /opt/miniconda3/bin/activate fedllm"
echo ""
echo "=== Optional: install Ollama for local LLM auditing ==="
echo "  curl -fsSL https://ollama.com/install.sh | sh"
echo "  ollama pull qwen2.5:7b"
echo "  ollama serve &   (run in separate terminal)"
