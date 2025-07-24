# Write python code which generates config files for pair_ids 1-9 and for datasets KS_Official and Lorenz_Official

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
"""

# Define the datasets and pair_ids
datasets = ["KS_Official", "Lorenz_Official"]
pair_ids = list(range(1, 10))  # 1-9

# Get the directory where this script is located
config_dir = Path(__file__).parent / 'config'

# Generate config files
for dataset in datasets:
    for pair_id in pair_ids:
        # Format the template with current values
        config_content = config_template.format(
            dataset=dataset,
            pair_id=pair_id
        )
        
        # Create filename
        filename = f"config_{dataset}_p{pair_id}.yaml"
        filepath = config_dir / filename
        
        # Write the config file
        with open(filepath, 'w') as f:
            f.write(config_content)
        
        print(f"Generated: {filename}")

print(f"Generated {len(datasets) * len(pair_ids)} config files")