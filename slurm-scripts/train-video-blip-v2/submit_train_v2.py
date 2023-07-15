import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--account", required=True)
parser.add_argument("--partition", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--run_name", required=True)
parser.add_argument("--num_gpus", required=True, type=int)
parser.add_argument("--mem_per_gpu", required=True)
parser.add_argument("--time", required=True)
parser.add_argument("--train_narrated_actions_dir", required=True)
parser.add_argument("--val_narrated_actions_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--dataloader_num_workers", type=int, required=True)
parser.add_argument("--train_batch_size", type=int, required=True)
parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
parser.add_argument("--per_device_eval_batch_size", type=int, required=True)
parser.add_argument("--email")
parser.add_argument("--transformers_cache")
parser.add_argument("--wandb_project", default="video-blip")
args = parser.parse_args()

email = ""
if args.email is not None:
    email = f"#SBATCH --mail-user={args.email}\n#SBATCH --mail-type=BEGIN,END"
transformers_cache = ""
if args.transformers_cache is not None:
    transformers_cache = f"export TRANSFORMERS_CACHE={args.transformers_cache}"

per_device_train_batch_size = (
    args.train_batch_size // args.gradient_accumulation_steps // args.num_gpus
)

output_dir = os.path.join(args.output_dir, args.run_name)

script = rf"""#!/bin/bash

#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --job-name=train-video-blip-v2
{email}
#SBATCH --account={args.account}
#SBATCH --ntasks={args.num_gpus}
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task={args.dataloader_num_workers}
#SBATCH --mem-per-gpu={args.mem_per_gpu}
#SBATCH --output=%x-%j.log

module load python/3.10.4 cuda
{transformers_cache}
export WANDB_PROJECT={args.wandb_project}
RDZV_ID=12345
MASTER_NODE=$(srun --nodes=1 --ntasks=1 hostname)
srun --cpus-per-task {args.dataloader_num_workers} poetry run torchrun --nnodes={args.num_gpus} --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE \
    ../../scripts/ego4d/train_v2.py \
    --model_name_or_path {args.model} \
    --num_subsample_frames 8 \
    --num_videos_per_sample 5 \
    --train_narrated_actions_dir {args.train_narrated_actions_dir} \
    --val_narrated_actions_dir {args.val_narrated_actions_dir} \
    --output_dir {output_dir} \
    --num_train_epochs 30 \
    --warmup_steps 1000 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size {per_device_train_batch_size} \
    --gradient_accumulation_steps {args.gradient_accumulation_steps} \
    --ddp_find_unused_parameters False \
    --per_device_eval_batch_size {args.per_device_eval_batch_size} \
    --weight_decay 0.05 \
    --dataloader_num_workers {args.dataloader_num_workers} \
    --bf16 True \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --report_to wandb \
    --run_name {args.run_name}
"""  # noqa: E501
subprocess.run(["sbatch"], input=script, text=True)
