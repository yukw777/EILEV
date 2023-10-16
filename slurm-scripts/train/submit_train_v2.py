import argparse
import base64
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
parser.add_argument("--train_frames_dir", required=True)
parser.add_argument("--train_annotation_file")
parser.add_argument("--val_frames_dir", required=True)
parser.add_argument("--val_annotation_file")
parser.add_argument("--output_dir", required=True)
parser.add_argument(
    "--train_num_in_context_examples_per_sample", type=int, required=True
)
parser.add_argument("--val_num_in_context_examples_per_sample", type=int, required=True)
parser.add_argument("--verb_noun_ratio", type=float, required=True)
parser.add_argument("--dataloader_num_workers", type=int, required=True)
parser.add_argument("--train_batch_size", type=int, required=True)
parser.add_argument("--per_device_train_batch_size", type=int, required=True)
parser.add_argument("--per_device_eval_batch_size", type=int, required=True)
parser.add_argument("--num_train_epochs", type=int, default=5)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--email")
parser.add_argument("--transformers_cache")
parser.add_argument("--wandb_project", required=True)
parser.add_argument("--resume_from_checkpoint", default=None)
parser.add_argument("--deepspeed_stage_2", action="store_true")
args = parser.parse_args()

email = ""
if args.email is not None:
    email = f"#SBATCH --mail-user={args.email}\n#SBATCH --mail-type=BEGIN,END"
transformers_cache = ""
if args.transformers_cache is not None:
    transformers_cache = f"export TRANSFORMERS_CACHE={args.transformers_cache}"
resume_from_checkpoint = ""
if args.resume_from_checkpoint is not None:
    resume_from_checkpoint = f"--resume_from_checkpoint {args.resume_from_checkpoint}"

deepspeed = ""
if args.deepspeed_stage_2:
    encoded_config = base64.urlsafe_b64encode(
        b"""{
"bf16": {
    "enabled": "auto"
},
"zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
        "device": "none",
        "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
},
"gradient_accumulation_steps": "auto",
"gradient_clipping": "auto",
"steps_per_print": 2000,
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto",
"wall_clock_breakdown": false
}"""
    ).decode()
    deepspeed = f"--deepspeed {encoded_config}"

gradient_accumulation_steps = (
    args.train_batch_size // args.per_device_train_batch_size // args.num_gpus
)

output_dir = os.path.join(args.output_dir, args.run_name)

train_annotation_file = ""
if args.train_annotation_file is not None:
    train_annotation_file = f"--train_annotation_file {args.train_annotation_file}"

val_annotation_file = ""
if args.val_annotation_file is not None:
    val_annotation_file = f"--val_annotation_file {args.val_annotation_file}"

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
RDZV_ID=$RANDOM
MASTER_NODE=$(srun --nodes=1 --ntasks=1 hostname)
srun --cpus-per-task {args.dataloader_num_workers} poetry run torchrun --nnodes={args.num_gpus} --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE \
    ../../scripts/general/train_v2.py \
    --model_name_or_path {args.model} \
    --num_subsample_frames 8 \
    --train_num_in_context_examples_per_sample {args.train_num_in_context_examples_per_sample} \
    --val_num_in_context_examples_per_sample {args.val_num_in_context_examples_per_sample} \
    --verb_noun_ratio {args.verb_noun_ratio} \
    --train_frames_dir {args.train_frames_dir} \
    {train_annotation_file} \
    --val_frames_dir {args.val_frames_dir} \
    {val_annotation_file} \
    --output_dir {output_dir} \
    --num_train_epochs {args.num_train_epochs} \
    --warmup_steps {args.warmup_steps} \
    --learning_rate 1e-5 \
    --per_device_train_batch_size {args.per_device_train_batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --ddp_find_unused_parameters False \
    --per_device_eval_batch_size {args.per_device_eval_batch_size} \
    --weight_decay 0.05 \
    --dataloader_num_workers {args.dataloader_num_workers} \
    --bf16 True \
    {deepspeed} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --report_to wandb \
    --run_name {args.run_name} \
    {resume_from_checkpoint}
"""  # noqa: E501
subprocess.run(["sbatch"], input=script, text=True)
