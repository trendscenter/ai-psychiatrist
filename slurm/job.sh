#!/bin/bash
#SBATCH -J ollama
#SBATCH -p qTRDGPU
#SBATCH -A trends53c17
#SBATCH -t 00:30:00
#SBATCH -c 24
#SBATCH --mem=100g
#SBATCH --gres=gpu:A40:2
#SBATCH -e err%A-%a.err
#SBATCH -o out%A-%a.out

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