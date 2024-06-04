import argparse
import base64
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--account", required=True)
parser.add_argument("--partition", required=True)
parser.add_argument("--num_gpus", required=True, type=int)
parser.add_argument("--mem_per_gpu", required=True)
parser.add_argument("--time", required=True)
parser.add_argument("--dataloader_num_workers", type=int, required=True)
parser.add_argument("--train_batch_size", type=int, required=True)
parser.add_argument("--per_device_train_batch_size", type=int, required=True)
parser.add_argument("--email")
parser.add_argument("--transformers_cache")
parser.add_argument("--wandb_project")
parser.add_argument("--deepspeed_stage_2", action="store_true")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("train_args", nargs=argparse.REMAINDER)
args = parser.parse_args()

email = ""
if args.email is not None:
    email = f"#SBATCH --mail-user={args.email}\n#SBATCH --mail-type=BEGIN,END"

transformers_cache = ""
if args.transformers_cache is not None:
    transformers_cache = f"export TRANSFORMERS_CACHE={args.transformers_cache}"

wandb_project = ""
report_to_wandb = ""
if args.wandb_project is not None:
    wandb_project = f"export WANDB_PROJECT={args.wandb_project}"
    report_to_wandb = "--report_to wandb"


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

train_args_str = " ".join(args.train_args[1:])

script = rf"""#!/bin/bash

#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --job-name=train-eilev
{email}
#SBATCH --account={args.account}
#SBATCH --ntasks={args.num_gpus}
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task={args.dataloader_num_workers}
#SBATCH --mem-per-gpu={args.mem_per_gpu}
#SBATCH --output=%x-%j.log

module load python/3.10.4 cuda
{transformers_cache}
{wandb_project}
RDZV_ID=$RANDOM
MASTER_NODE=$(srun --nodes=1 --ntasks=1 hostname)
srun --cpus-per-task {args.dataloader_num_workers} poetry run torchrun --nnodes={args.num_gpus} --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE \
    ../../scripts/general/train_v2.py \
    {train_args_str} \
    {deepspeed} \
    {report_to_wandb} \
    --dataloader_num_workers {args.dataloader_num_workers} \
    --per_device_train_batch_size {args.per_device_train_batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps}
"""  # noqa: E501
print(script)
if not args.dry_run:
    subprocess.run(["sbatch"], input=script, text=True)
