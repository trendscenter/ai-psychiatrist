# AI Psychiatrist Assistant

An AI psychiatrist assistant for measuring depression symptoms from clinical interviews

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

1. In your home directory on cluster, type `echo $PATH` in terminal to see if trends apps are on your path. If not, add trends apps on your path using the command `PATH=/trdapps/linux-x86_64/bin/:$PATH`. Type `echo $PATH` in terminal again and you should be able to see `/trdapps/linux-x86_64/bin/:...` in the output.

2. Start Ollama by submitting the SLURM job script in [`slurm/job.sh`](slurm/job.sh):
```bash
cd slurm
sbatch job.sh
```

3. Check the status of the job using the command `squeue -u <username>`, where `<username>` is your username. Check the node that Ollama is running on in the output of the command. The node name is in the format `arctrdagnXXX`, where `XXX` is a number.

4. Once the job is running, you can access Ollama on the node. See the notebook [`ollama_example.ipynb`](ollama_example.ipynb) for an example of how to use Ollama.

5. If you stop using Ollama, you can stop the job using the command `scancel <job_id>`, where `<job_id>` is the job ID of the Ollama job. You can find the job ID in the output of the command `squeue -u <username>`.

## References

- [Ollama documentation](https://github.com/ollama/ollama)

- [TReNDS cluster documentation](https://trendscenter.github.io/wiki)