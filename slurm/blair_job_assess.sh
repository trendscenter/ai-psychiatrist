#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p qTRD
#SBATCH -c 5
#SBATCH --mem=16g
#SBATCH -t 14:00:00
#SBATCH -J aipsy
#SBATCH -e ./err/err%A-%a.err
#SBATCH -o ./out/out%A-%a.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nblair7@gsu.edu

# Print debug info
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"

# Create output directories if they don't exist
mkdir -p ./err
mkdir -p ./out

# Initialize conda properly
source /home/users/nblair7/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate aipsy

# Verify environment
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Change to script directory
cd /home/users/nblair7/ai-psychiatrist/qualitative_assessment
echo "Script directory: $(pwd)"

# Set environment variables
export PYTHONUNBUFFERED=1
export SUBJECT_TIME_LIMIT=300

echo "Starting assessment with ${SUBJECT_TIME_LIMIT} second timeout per subject"
echo "Python unbuffered: $PYTHONUNBUFFERED"

# Run the script with explicit python path and error handling
python -u qual_evaluationoriginal.py
exit_code=$?

echo "Script exit code: $exit_code"
echo "Job finished at: $(date)"

exit $exit_code