from pathlib import Path

top_dir = Path(__file__).parent.parent

# Each job runs ctf/run.py on one per-pair config (forecast + CTF evaluation + plots).
# run.py resumes from checkpoints/<dataset>_<pair_id>.pkl, so a job that hits the
# wall time can simply be resubmitted.
cmd_template = \
"""\
#!/bin/bash

#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem={memory}G
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --nice=0

#SBATCH --job-name="{identifier}"
#SBATCH --output={log_dir}/"{identifier}".out

#SBATCH --mail-type=NONE

identifier={identifier}

repo="{repo}"

echo "Running Python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}

cd $repo
eval "$(micromamba shell hook --shell bash)"
micromamba activate {env}

python -u $repo/ctf/run.py $repo/ctf/config/config_{dataset}_p{pair_id}.yaml

echo "Finished running Python"
"""

slurm_dir = top_dir / 'slurms'
log_dir = top_dir / 'logs'
slurm_dir.mkdir(exist_ok=True)
log_dir.mkdir(exist_ok=True)

# Clean up slurm repo
for file in slurm_dir.glob('*.slurm'):
    file.unlink()

# Per-dataset host memory (GB). mhd-2cyls pairs 8/9 load 3 x 1 GB training matrices
# on top of the LLM.
datasets = {
    "pke-hpmr": 32,
    "dynasty": 32,
    "msfr": 48,
    "mhd-2cyls": 64,
}
pair_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Verify with `hyakalloc` / `sinfo` before submitting
account = "amath"
partition = "gpu-rtx6k"
cpus = 4
time = "1-00:00:00"
env = "llmtime"

skip_count = 0
write_count = 0
total_count = 0

for dataset, memory in datasets.items():
    for pair_id in pair_ids:
        # Must match the identifier used by run.py
        identifier = f"{dataset}_{pair_id}"

        cmd = cmd_template.format(
            dataset=dataset,
            pair_id=pair_id,
            identifier=identifier,
            partition=partition,
            account=account,
            memory=memory,
            cpus=cpus,
            time=time,
            env=env,
            repo=top_dir.resolve(),
            log_dir=log_dir.resolve(),
        )

        total_count += 1

        # Skip creating slurms that are completed
        pickle_file = top_dir / 'pickles' / f'{identifier}.pkl'

        if pickle_file.exists():
            #print(f'Skipping {identifier}')
            skip_count += 1
            continue

        with open(slurm_dir / f'{identifier}.slurm', "w") as f:
            f.write(cmd)
            write_count += 1

print(f"Skipped {skip_count} jobs")
print(f"Created {write_count} jobs")
print(f"Total jobs: {total_count}")
