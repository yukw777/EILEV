import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--account", required=True)
parser.add_argument("--partition", required=True)
parser.add_argument("--time", required=True)
parser.add_argument("--email", required=True)
parser.add_argument("--job_name", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--annotation_path", required=True)
parser.add_argument("--epic_kitchens_55_video_dir_path", required=True)
parser.add_argument("--epic_kitchens_100_video_dir_path", required=True)
parser.add_argument("--frames_dir", required=True)
parser.add_argument("--subsample_frames", type=int, required=True)
parser.add_argument("--num_workers", type=int, required=True)
parser.add_argument("--mem_per_worker", required=True)
parser.add_argument("--transformers_cache")
args = parser.parse_args()

transformers_cache = ""
if args.transformers_cache is not None:
    transformers_cache = f"export TRANSFORMERS_CACHE={args.transformers_cache}"

script = rf"""#!/bin/bash

#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --job-name=extract-{args.job_name}-{args.split}
#SBATCH --mail-user={args.email}
#SBATCH --mail-type=BEGIN,END
#SBATCH --account={args.account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={args.num_workers}
#SBATCH --mem-per-cpu={args.mem_per_worker}
#SBATCH --output=%x-%j.log

module load python/3.10.4
{transformers_cache}
poetry run python ../../scripts/epic-kitchens/epic_kitchens_extract_frames.py \
    --annotation_path {args.annotation_path} \
    --epic_kitchens_55_video_dir_path {args.epic_kitchens_55_video_dir_path} \
    --epic_kitchens_100_video_dir_path {args.epic_kitchens_100_video_dir_path} \
    --frames_dir {os.path.join(args.frames_dir, f'{args.split}-{args.job_name}-subsample-{args.subsample_frames}')} \
    --model_name_or_path {args.model} \
    --num_subsample_frames {args.subsample_frames} \
    --num_workers {args.num_workers}
"""  # noqa: E501
subprocess.run(["sbatch"], input=script, text=True)
