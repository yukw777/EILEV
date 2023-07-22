import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--account", required=True)
parser.add_argument("--partition", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--shot", type=int, required=True)
parser.add_argument("--few_shot_narrated_actions_dir", required=True)
parser.add_argument("--eval_narrated_actions_dir", required=True)
parser.add_argument("--wandb_project", required=True)
parser.add_argument("--job_name_prefix", required=True)
parser.add_argument("--email")
parser.add_argument("--time", default="00-01:00:00")
parser.add_argument("--cpus", default=1, type=int)
parser.add_argument("--memory", default="16GB")
parser.add_argument("--num_dataloader_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--no_video_causal_mask", action="store_true")
parser.add_argument("--generation_config")
parser.add_argument("--transformers_cache")
args = parser.parse_args()

account = f"#SBATCH --account={args.account}"
partition = f"#SBATCH --partition={args.partition}"
email = ""
if args.email is not None:
    email = f"#SBATCH --mail-user={args.email}\n#SBATCH --mail-type=BEGIN,END"
no_video_causal_mask = ""
if args.no_video_causal_mask:
    no_video_causal_mask = "--no_video_causal_mask"
gen_config = ""
if args.generation_config is not None:
    gen_config = f"--generation_config {args.generation_config}"
transformers_cache = ""
if args.transformers_cache is not None:
    transformers_cache = f"export TRANSFORMERS_CACHE={args.transformers_cache}"


script = rf"""#!/bin/bash

{partition}
#SBATCH --time={args.time}
#SBATCH --job-name={args.job_name_prefix}-generate-narration-texts
{email}
{account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem-per-gpu={args.memory}
#SBATCH --output=%x-%j.log

module load python/3.10.4 cuda
{transformers_cache}
export WANDB_PROJECT={args.wandb_project}
export WANDB_NAME={args.job_name_prefix}-generate-narration-texts-{args.shot}-shot
poetry run python ../../scripts/ego4d/generate_narration_texts.py \
  --model {args.model} \
  --device cuda \
  --num_dataloader_workers {args.num_dataloader_workers} \
  --few_shot_narrated_actions_dir {args.few_shot_narrated_actions_dir} \
  --eval_narrated_actions_dir {args.eval_narrated_actions_dir} \
  --batch_size {args.batch_size} \
  --num_shot {args.shot} \
  --log_narration_texts \
  {no_video_causal_mask} \
  {gen_config}
"""  # noqa: E501
subprocess.run(["sbatch"], input=script, text=True)
