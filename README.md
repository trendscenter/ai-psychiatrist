# AI Psychiatrist Assistant

An AI psychiatrist assistant for measuring depression symptoms from clinical interview transcripts

## Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/trendscenter/ai-psychiatrist.git
```

2. Navigate to the cloned directory and create a new conda environment using the provided [`env_reqs.yml`](env_reqs.yml) file:
```bash
cd ai-psychiatrist
conda env create --name aipsy --file ./env_reqs.yml
```

3. Activate the conda environment:
```bash
conda activate aipsy
```

4. Create a new git branch for your changes:
```bash
git checkout -b dev_<your_last_name>
```
Replace `<your_last_name>` with your last name.

## Ollama on TReNDS Cluster

1. Start Ollama by submitting the SLURM job script [`slurm/job_ollama.sh`](slurm/job_ollama.sh):
```bash
cd slurm
sbatch job_ollama.sh
```

2. Check the status of the job using the command `squeue -u <username>`, where `<username>` is your username. Check the node that Ollama is running on in the output of the command. The node name is in the format `arctrdagnXXX`, where `XXX` is a number.

3. Once the job is running, you can access Ollama on the node. See the Python script [`ollama_example.py`](ollama_example.py) for an example of how to use Ollama. Update `OLLAMA_NODE` to the node where Ollama is running. Submit the SLURM job script [`slurm/job_assess.sh`](slurm/job_assess.sh) to run the Python code:
```bash
sbatch job_assess.sh
```

4. If you stop using Ollama, you can stop the job using the command `scancel <job_id>`, where `<job_id>` is the job ID of the Ollama job. You can find the job ID in the output of the command `squeue -u <username>`.

## Analysis

- [qualitative_assessment](qualitative_assessment) contains scripts for identifying social and biological risk factors.

- [quantitative_assessment](quantitative_assessment) contains scripts for predicting PHQ-8 scores.

- [visualization](visualization) contains scripts for generating figures.

## References

- [Ollama documentation](https://github.com/ollama/ollama)

- [TReNDS cluster documentation](https://trendscenter.github.io/wiki)

## Contributors

Adam Greene, Neviah Blair, Xinhui Li