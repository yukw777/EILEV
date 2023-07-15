#!/bin/bash
# Based on https://stackoverflow.com/a/29754866 and https://stackoverflow.com/a/36303809
# More safety, by turning some bugs into errors.
# Without `errexit` you don’t need ! and can replace
# ${PIPESTATUS[0]} with a simple $?, but I prefer safety.
set -o errexit -o pipefail -o noclobber -o nounset

# -allow a command to fail with !’s side effect on errexit
# -use return value from ${PIPESTATUS[0]}, because ! hosed $?
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

# option --output/-o requires 1 argument
LONGOPTS=time:,cpus:,memory:,shot:,model:,num_dataloader_workers:,batch_size:,no_video_causal_mask,generation_config:,email:,partition:,account:
OPTIONS=t:c:m:s:o:n:b:dg:e:p:a:

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

t=00-01:00:00
cpus=1
memory=16GB
shot=0
model=/nfs/turbo/coe-chaijy/checkpoints/video-blip/video-blip-v2-opt-2.7b-ego4d
num_dataloader_workers=0
batch_size=1
no_video_causal_mask=""
generation_config=""
email=""
partition=""
account=""
# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -t|--time)
            t="$2"
            shift 2
            ;;
        -c|--cpus)
            cpus="$2"
            shift 2
            ;;
        -m|--memory)
	    memory="$2"
            shift 2
            ;;
        -s|--shot)
	    shot="$2"
            shift 2
            ;;
        -o|--model)
	    model="$2"
            shift 2
            ;;
        -n|--num_dataloader_workers)
	    num_dataloader_workers="$2"
            shift 2
            ;;
        -b|--batch_size)
	    batch_size="$2"
            shift 2
            ;;
        -d|--no_video_causal_mask)
	    no_video_causal_mask="--no_video_causal_mask"
            shift
            ;;
        -g|--generation_config)
	    generation_config="--generation_config $2"
            shift 2
            ;;
        -e|--email)
            email="#SBATCH --mail-user=$2
#SBATCH --mail-type=BEGIN,END"
            shift 2
            ;;
        -p|--partition)
	    partition="#SBATCH --partition=$2"
            shift 2
            ;;
        -a|--account)
	    account="#SBATCH --account=$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

# this is how we'd handle non-option arguments
# but we don't need to here
#if [[ $# -ne 1 ]]; then
#    echo "$0: A single input file is required."
#    exit 4
#fi

sbatch <<EOT
#!/bin/bash

$partition
#SBATCH --time=$t
#SBATCH --job-name=video-blip-generate-narration-texts
$email
$account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem-per-gpu=$memory
#SBATCH --output=%x-%j.log

module load python/3.10.4 cuda
export TRANSFORMERS_CACHE=/scratch/chaijy_root/chaijy2/kpyu/.cache/huggingface/
export WANDB_PROJECT=video-blip-ego4d-eval
export WANDB_NAME=generate-narration-texts-$shot-shot
poetry run python ../../scripts/ego4d/generate_narration_texts.py \
  --model $model \
  --device cuda \
  --num_dataloader_workers $num_dataloader_workers \
  --few_shot_narrated_actions_dir /nfs/turbo/coe-chaijy/datasets/ego4d/fho_main_train_frames-448px-subsample-8 \
  --eval_narrated_actions_dir /nfs/turbo/coe-chaijy/datasets/ego4d/fho_main_val_frames-448px-subsample-8 \
  --batch_size $batch_size \
  --num_shot $shot \
  --log_narration_texts \
  $no_video_causal_mask \
  $generation_config
EOT
