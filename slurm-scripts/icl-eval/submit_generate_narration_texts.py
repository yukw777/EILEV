import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--account", required=True)
parser.add_argument("--partition", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--time", required=True)
parser.add_argument("--num_gpus", required=True, type=int)
parser.add_argument("--mem_per_gpu", required=True)
parser.add_argument("--num_dataloader_workers", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--eval_frames_dir", required=True)
parser.add_argument("--eval_annotation_file")
parser.add_argument("--in_context_query_map_file", required=True)
parser.add_argument("--in_context_example_frames_dir", required=True)
parser.add_argument("--in_context_example_annotation_file")
parser.add_argument("--wandb_project", required=True)
parser.add_argument("--job_name_prefix", required=True)
parser.add_argument("--email")
parser.add_argument("--generation_config")
parser.add_argument("--transformers_cache")
parser.add_argument("--shuffle_in_context_example_frames", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

email = ""
if args.email is not None:
    email = f"#SBATCH --mail-user={args.email}\n#SBATCH --mail-type=BEGIN,END"
gen_config = ""
if args.generation_config is not None:
    gen_config = f"--generation_config {args.generation_config}"
transformers_cache = ""
if args.transformers_cache is not None:
    transformers_cache = f"export TRANSFORMERS_CACHE={args.transformers_cache}"

multi_gpu = f"""RDZV_ID=$RANDOM
MASTER_NODE=$(srun --nodes=1 --ntasks=1 hostname)
srun --cpus-per-task {args.num_dataloader_workers} poetry run torchrun --nnodes={args.num_gpus} --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE \\
  ../../scripts/general/generate_narration_texts.py \\"""  # noqa: E501

single_gpu = "poetry run python ../../scripts/general/generate_narration_texts.py \\"

job_name = (
    args.job_name_prefix
    + "-"
    + os.path.splitext(os.path.basename(args.in_context_query_map_file))[0]
)

script = rf"""#!/bin/bash

#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --job-name=generate-narration-texts-{job_name}
{email}
#SBATCH --account={args.account}
#SBATCH --ntasks={args.num_gpus}
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task={args.num_dataloader_workers}
#SBATCH --mem-per-gpu={args.mem_per_gpu}
#SBATCH --output=%x-%j.log

module load python/3.10.4 cuda
{transformers_cache}
export WANDB_NAME={job_name}
{single_gpu if args.num_gpus < 2 else multi_gpu}
    --model {args.model} \
    --num_dataloader_workers {args.num_dataloader_workers} \
    --eval_frames_dir {args.eval_frames_dir} \
    {f'--eval_annotation_file {args.eval_annotation_file}' if args.eval_annotation_file is not None else ''} \
    --in_context_query_map_file {args.in_context_query_map_file} \
    --in_context_example_frames_dir {args.in_context_example_frames_dir} \
    {f'--in_context_example_annotation_file {args.in_context_example_annotation_file}' if args.in_context_example_annotation_file is not None else ''} \
    --batch_size {args.batch_size} \
    {gen_config} \
    {f'--shuffle_in_context_example_frames' if args.shuffle_in_context_example_frames else ''} \
    --wandb_project {args.wandb_project}
"""  # noqa: E501

print(script)
if not args.dry_run:
    subprocess.run(["sbatch"], input=script, text=True)
