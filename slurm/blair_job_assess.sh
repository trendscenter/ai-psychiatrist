#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p qTRD
#SBATCH -c 5
#SBATCH --mem=100g
#SBATCH -t 14:00:00
#SBATCH -J aipsy
#SBATCH -e ./err/err%A-%a.err
#SBATCH -o ./out/out%A-%a.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nblair7@gsu.edu


sleep 5s

source /home/users/nblair7/miniconda3/bin/activate
conda activate aipsy

cd /home/users/nblair7/ai-psychiatrist

export SUBJECT_TIME_LIMIT=300

echo "Starting assessment with ${SUBJECT_TIME_LIMIT} second timeout per subject"
echo "Job started at: $(date)"

python qual_assessment.py

echo "Job finished at: $(date)"

sleep 5s
