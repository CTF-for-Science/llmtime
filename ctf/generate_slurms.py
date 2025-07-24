from pathlib import Path

top_dir = Path(__file__).parent.parent

cmd_template = \
"""\
#!/bin/bash

#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem={memory}G
#SBATCH --cpus-per-task=5
#SBATCH --time=4-00:00:00
#SBATCH --nice=0

#SBATCH --job-name="{identifier}"
#SBATCH --output=/mmfs1/home/alexeyy/storage/CTF-for-Science/models/llmtime/logs/"{identifier}".out

#SBATCH --mail-type=NONE
#SBATCH --mail-user=alexeyy@uw.edu

identifier={identifier}

repo="/mmfs1/home/alexeyy/storage/CTF-for-Science/models/llmtime"

recon_ctx={recon_ctx}
dataset={dataset}
pair_id={pair_id}
validation={validation}

echo "Running Python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $repo
eval "$(micromamba shell hook --shell bash)"
micromamba activate llmtime

python -u $repo/ctf/forecast.py --dataset $dataset --pair_id $pair_id --validation $validation --identifier $identifier

echo "Finished running Python"
"""

# Clean up slurm repo
slurm_dir = top_dir / 'slurms'
for file in slurm_dir.glob('*.slurm'):
    file.unlink()

datasets = ["KS_Official", "Lorenz_Official"]
pair_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
validations = [0]
recon_ctxs = [20]
account = "amath"
partition = "gpu-rtx6k"
memory = 45

skip_count = 0
write_count = 0
total_count = 0

for validation in validations:
    for dataset in datasets:
        for pair_id in pair_ids:
            for recon_ctx in recon_ctxs:
                identifier = f"{dataset}_p{pair_id}_v{validation:d}_r{recon_ctx}"

                cmd = cmd_template.format(
                    dataset=dataset,
                    pair_id=pair_id,
                    validation=validation,
                    recon_ctx=recon_ctx,
                    identifier=identifier,
                    partition=partition,
                    account=account,
                    memory=memory,
                )

                total_count += 1

                # Skip creating slurms that are completed
                pickle_file = top_dir / 'pickles' / f'{identifier}.pkl'

                if pickle_file.exists():
                    #print(f'Skipping {identifier}')
                    skip_count += 1
                    continue

                with open(top_dir / 'slurms' / f'{identifier}.slurm', "w") as f:
                    f.write(cmd)
                    write_count += 1

print(f"Skipped {skip_count} jobs")
print(f"Created {write_count} jobs")
print(f"Total jobs: {total_count}")
