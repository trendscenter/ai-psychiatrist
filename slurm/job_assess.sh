#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p qTRD
#SBATCH -c 5
#SBATCH --mem=100g
#SBATCH -t 7200
#SBATCH -J aipsy
#SBATCH -e ./err/err%A-%a.err
#SBATCH -o ./out/out%A-%a.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xli77@gsu.edu

sleep 5s

source /home/users/xli77/anaconda3/bin/activate
conda activate aipsy

cd /data/users4/xli/ai-psychiatrist

python ollama_example.py

sleep 5s