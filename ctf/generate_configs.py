# Write python code which generates config files for pair_ids 1-9 for each dataset below

from pathlib import Path

config_template = \
"""dataset:
  name: {dataset}
  pair_id:
  - {pair_id}
model:
  name: llmtime
  recon_ctx: 20
  validation: 0
{extra}"""

# Define the datasets and per-dataset extra model options.
# Every dataset is forecast in full space, one channel at a time (the original LLMTime method).
datasets = {
    "KS_Official": "",
    "Lorenz_Official": "",
    "pke-hpmr": "",     # 6 channels
    "dynasty": "",      # 123 channels
    "msfr": "",         # 19400 channels
    "mhd-2cyls": "",    # 122400 channels
}
pair_ids = list(range(1, 10))  # 1-9

# Get the directory where this script is located
config_dir = Path(__file__).parent / 'config'

# Generate config files
for dataset, extra in datasets.items():
    for pair_id in pair_ids:
        # Format the template with current values
        config_content = config_template.format(
            dataset=dataset,
            pair_id=pair_id,
            extra=extra,
        )
        
        # Create filename
        filename = f"config_{dataset}_p{pair_id}.yaml"
        filepath = config_dir / filename
        
        # Write the config file
        with open(filepath, 'w') as f:
            f.write(config_content)
        
        print(f"Generated: {filename}")

print(f"Generated {len(datasets) * len(pair_ids)} config files")
