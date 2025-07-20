#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p qTRD
#SBATCH -c 5
#SBATCH --mem=100g
#SBATCH -t 07:00:00
#SBATCH -J aipsy
#SBATCH -e ./err/err%A-%a.err
#SBATCH -o ./out/out%A-%a.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agreene46@student.gsu.edu

sleep 5s

source ~/.bashrc
source ~/init_miniconda3.sh

cd /data/users2/agreene46/ai-psychiatrist

conda activate

python quantitative_analysis.py

sleep 5s