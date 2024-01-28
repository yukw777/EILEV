import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--account", required=True)
parser.add_argument("--partition", required=True)
parser.add_argument("--time", required=True)
parser.add_argument("--mem_per_gpu", required=True)
parser.add_argument("--prediction_file", required=True)
parser.add_argument("--model")
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--wandb_project", required=True)
parser.add_argument("--email")
parser.add_argument("--transformers_cache")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

email = ""
if args.email is not None:
    email = f"#SBATCH --mail-user={args.email}\n#SBATCH --mail-type=BEGIN,END"
transformers_cache = ""
if args.transformers_cache is not None:
    transformers_cache = f"export TRANSFORMERS_CACHE={args.transformers_cache}"
model = ""
if args.model is not None:
    model = f"--model {args.model}"

script = rf"""#!/bin/bash

#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --job-name=generate-{Path(args.prediction_file).stem}
{email}
#SBATCH --account={args.account}
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu={args.mem_per_gpu}
#SBATCH --output=%x-%j.log

module load python/3.10.4 cuda
{transformers_cache}
export WANDB_PROJECT={args.wandb_project}
poetry run python ../../scripts/baselines/majority/majority_generate_full_sent.py \
    {args.prediction_file} \
    --device cuda \
    {model} \
    --batch_size {args.batch_size} \
    --log_to_wandb
"""  # noqa: E501

print(script)
if not args.dry_run:
    subprocess.run(["sbatch"], input=script, text=True)
