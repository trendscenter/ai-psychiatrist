#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p qTRDHM
#SBATCH -c 5
#SBATCH --mem=30g
#SBATCH -t 15:15:00
#SBATCH -J aipsy_4F
#SBATCH -e ./err/err%A-%a.err
#SBATCH -o ./out/out%A-%a.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agreene46@student.gsu.edu

sleep 5s

source ~/.bashrc
source ~/init_miniconda3.sh

cd /data/users2/agreene46/ai-psychiatrist/quantitative_assessment

conda activate

CHUNK_STEPS="chunk_4_step_2"
EXAMPLES_NUMS="3"
NUM_RUNS="3"
DIMS="1024"
OLLAMA_NODE="arctrdagn031"

python embedding_batch_script.py --chunk_step "$CHUNK_STEPS" --examples_num "$EXAMPLES_NUMS" --num_runs "$NUM_RUNS" --dims "$DIMS" --ollama_node "$OLLAMA_NODE"
sleep 5s
