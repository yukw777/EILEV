# EILEV: Efficient In-Context Learning in Vision-Language Models for Egocentric Videos

[![Demo](https://img.shields.io/badge/Website-Demo-ff69b4.svg)](https://dd71-141-212-106-177.ngrok-free.app) [![Paper](http://img.shields.io/badge/paper-arxiv.2311.17041-B31B1B.svg)](https://arxiv.org/abs/2311.17041)

![Teaser](figures/teaser.png)

EILEV is a novel training method that can elicit in-context learning in vision-language models (VLMs) for egocentric videos without requiring massive, naturalistic egocentric video datasets. It is an extension to the preliminary work done in [VideoBLIP](https://github.com/yukw777/VideoBLIP), which contains models trained using EILEV with no in-context examples.

## Setup

```bash
# Install poetry https://python-poetry.org/
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone git@github.com:yukw777/EILEV.git
cd EILEV

# Install EILEV in editable mode
pip install -e .
```

## Running Demo Locally

### EILEV

```bash
# Install extra packages
pip install -e ".[demo]"

# Run `python demo/eilev_demo.py --help` for details
# By default, the demo uses `kpyu/eilev-blip2-opt-2.7b`, which requires about 16GB of VRAM.
python demo/eilev_demo.py --device cuda
```

### VideoBLIP

```bash
# Install extra packages
pip install -e ".[demo]"

# Run `python demo/video_blip_demo.py --help` for details
# By default, the demo uses `kpyu/video-blip-flan-t5-xl-ego4d`, which requires about 16GB of VRAM.
python demo/video_blip_demo.py --device cuda
```

## Pretrained Weights

### EILEV

- [`kpyu/eilev-blip2-opt-2.7b`](https://huggingface.co/kpyu/eilev-blip2-opt-2.7b)
  - [`Salesforce/blip2-opt-2.7b`](https://huggingface.co/Salesforce/blip2-opt-2.7b) trained using EILEV on on Ego4D.
- [`kpyu/eilev-blip2-flan-t5-xl`](https://huggingface.co/kpyu/eilev-blip2-flan-t5-xl)
  - [`Salesforce/blip2-flan-t5-xl`](https://huggingface.co/Salesforce/blip2-flan-t5-xl) trained using EILEV on Ego4D.

### VideoBLIP

- [`kpyu/video-blip-opt-2.7b-ego4d`](https://huggingface.co/kpyu/video-blip-opt-2.7b-ego4d)
  - VideoBLIP initialized with [`Salesforce/blip2-opt-2.7b`](https://huggingface.co/Salesforce/blip2-opt-2.7b) and fine-tuned on Ego4D.
- [`kpyu/video-blip-flan-t5-xl`](https://huggingface.co/kpyu/video-blip-opt-2.7b-ego4d)
  - VideoBLIP initialized with [`Salesforce/blip2-flan-t5-xl`](https://huggingface.co/Salesforce/blip2-flan-t5-xl) and fine-tuned on Ego4D.

## Training

**1. Download Ego4D**
You need the `fho` benchmark data from Ego4D. Below is an example command to download it. Please refer to the official [Ego4D website](https://ego4d-data.org/) for more details.

```bash
ego4d --output_directory="<output_dir>" --datasets full_scale annotations --benchmarks fho
```

**2. Extract frames**
Once you have the `fho` benchmark data from Ego4D, run the following commands to split train and validation data and extract frames.

```bash
# First split train and validation data
python scripts/ego4d/split_train_val_test.py \
    path/to/ego4d/v2/annotations/fho_main.json \
    path/to/extracted/frames \
    path/to/ego4d/v2/full_scale

# Extract frames
SPLIT=(train|val|test)
MODEL=<your-base-blip2-model> # e.g., Salesforce/blip2-opt-2.7b
SUBSAMPLE_FRAMES=8
python scripts/ego4d/extract_frames.py \
    --fho_main_path path/to/ego4d/v2/annotations/fho_main.json \
    --split_path path/to/extracted/frames/fho_main_${SPLIT}.json \
    --video_dir path/to/ego4d/v2/full_scale \
    --frames_dir path/to/extracted/frames/fho_main_${SPLIT}_frames-${MODEL}-subsample-${SUBSAMPLE_FRAMES} \
    --model_name_or_path ${MODEL} \
    --num_subsample_frames ${SUBSAMPLE_FRAMES} \
    --num_workers 8 # should be adjusted based on the number of CPUs
```

**3. Train**

- `kpyu/eilev-blip2-opt-2.7b-ego4d`

```bash
# Takes about 1 day and 12 hours on 8 A40s
RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
torchrun --nproc_per_node=8 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE \
  scripts/general/train_v2.py \
  --model_name_or_path Salesforce/blip2-opt-2.7b \
  --num_subsample_frames 8 \
  --train_num_in_context_examples_per_sample 16 \
  --val_num_in_context_examples_per_sample 16 \
  --verb_noun_ratio 0.5 \
  --train_frames_dir path/to/extracted/train/frames \
  --val_frames_dir path/to/extracted/val/frames \
  --output_dir path/to/output \
  --num_train_epochs 5 \
  --warmup_steps 0 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --ddp_find_unused_parameters False \
  --per_device_eval_batch_size 8 \
  --weight_decay 0.05 \
  --dataloader_num_workers <num_cpus> \
  --bf16 True \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 10
```

- `kpyu/eilev-blip2-flan-t5-xl`

```bash
# Takes about 1 day and 12 hours on 8 A40s
RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
torchrun --nproc_per_node=8 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_NODE \
  scripts/general/train_v2.py \
  --model_name_or_path Salesforce/blip2-flan-t5-xl \
  --num_subsample_frames 8 \
  --train_num_in_context_examples_per_sample 16 \
  --val_num_in_context_examples_per_sample 16 \
  --verb_noun_ratio 0.5 \
  --train_frames_dir path/to/extracted/train/frames \
  --val_frames_dir path/to/extracted/val/frames \
  --output_dir path/to/output \
  --num_train_epochs 5 \
  --warmup_steps 0 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --ddp_find_unused_parameters False \
  --per_device_eval_batch_size 8 \
  --weight_decay 0.05 \
  --dataloader_num_workers <num_cpus> \
  --bf16 True \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 10
```

- `kpyu/video-blip-opt-2.7b-ego4d`

```bash
# Takes about 24 hours on one A40
python scripts/general/train_v1.py \
    --model_name_or_path Salesforce/blip2-opt-2.7b \
    --num_subsample_frames 8 \
    --train_narrated_actions_dir path/to/extracted/train/frames \
    --val_narrated_actions_dir path/to/extracted/val/frames \
    --output_dir <output_dir> \
    --num_train_epochs 5 \
    --warmup_steps 1000 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.05 \
    --dataloader_num_workers 2 \
    --bf16 True \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10
```

- `kpyu/video-blip-flan-t5-xl`

```bash
# Takes about 23 hours on one A40
python scripts/general/train_v1.py \
    --model_name_or_path Salesforce/blip2-flan-t5-xl \
    --num_subsample_frames 8 \
    --train_narrated_actions_dir path/to/extracted/train/frames \
    --val_narrated_actions_dir path/to/extracted/val/frames \
    --output_dir <output_dir> \
    --num_train_epochs 5 \
    --warmup_steps 1000 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.05 \
    --dataloader_num_workers 2 \
    --bf16 True \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10
```

## Errors for EPIC-KITCHENS Videos

While extracting frames, you may encounter the following error on some of the EPIC-KITCHENS videos.:

```
Invalid NAL unit size (1053738930 > 11544).
missing picture in access unit with size 11548
Invalid NAL unit size (1053738930 > 11544).
Error splitting the input into NAL units.
```

This is most likely due to the variable frame rate of some of the videos under low light conditions as described [here](https://github.com/epic-kitchens/epic-kitchens-55-annotations#video-information). The solution is to re-encode them at a constant frame rate by running the following command:

```
ffmpeg -i input_video.MP4 -c:v libx264 -crf 23 -preset medium -r 60 -c:a copy output_video_60fps.MP4
```

The following videos have been re-encoded:

```
P29_05.MP4
P30_08.MP4
```

## Development

```bash
# Install poetry https://python-poetry.org/
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone git@github.com:yukw777/EILEV.git
cd EILEV

# Install EILEV using poetry
# Note: if you notice "keyring" related error messages or poetry hanging,
# pelase export the following environment variable.
# More info can be found here: https://github.com/python-poetry/poetry/issues/8623
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install --with dev

# Activate the poetry virtualenv
poetry shell

# Run unit tests to verify the dev installation
pytest
```

## Citing EILEV

```
@misc{yu2023efficient,
      title={Efficient In-Context Learning in Vision-Language Models for Egocentric Videos},
      author={Keunwoo Peter Yu and Zheyuan Zhang and Fengyuan Hu and Joyce Chai},
      year={2023},
      eprint={2311.17041},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

We thank [Shane Storks](https://shanestorks.com/) for his valuable insight and comments. This work has been supported by the Defense Advanced Research Projects Agency (DARPA) under the PTG Program, Contract No. HR00112220003. The views expressed are those of the authors and do not necessarily reflect the views of DARPA.
