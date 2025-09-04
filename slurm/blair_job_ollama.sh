#!/bin/bash
#SBATCH -J ollama
#SBATCH -p  qTRDGPUM
#SBATCH --gres=gpu:A100:4
#SBATCH -A trends53c17
#SBATCH -t 72:00:00
#SBATCH -c 24
#SBATCH --mem=16g
#SBATCH -e ./err/err%A-%a.err
#SBATCH -o ./out/out%A-%a.out

export PATH=/trdapps/linux-x86_64/bin/:$PATH

# Ensure both GPUs are visible to Ollama
export CUDA_VISIBLE_DEVICES=0,1

# Set environment variables for large context RAG optimization
export OLLAMA_HOST_MEMORY=false
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_MMAP=true
export GGML_CUDA_FORCE_CUBLAS=1
export GGML_CUDA_FORCE_MMQ=1
export OLLAMA_HOST=0.0.0.0
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_MODELS=/data/users4/splis/ollama/models/
# Force GPU backend
export OLLAMA_BACKEND=gpu

# Run ollama serve
ollama serve
