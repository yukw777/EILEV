import argparse
import logging
from pathlib import Path

import torch
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms import UniformTemporalSubsample
from transformers import Blip2Processor

from eilev.data.utils import generate_input_ids_and_labels_from_interleaved
from eilev.model.utils import process
from eilev.model.v2 import VideoBlipForConditionalGeneration


def generate(
    model: VideoBlipForConditionalGeneration,
    processor: Blip2Processor,
    videos_and_texts: list[str],
) -> None:
    video_path_handler = VideoPathHandler()
    # uniformly subsample 8 frames
    subsampler = UniformTemporalSubsample(8)
    prompts: list[tuple[str, int]] = [("", 0)]
    frames_list: list[torch.Tensor] = []
    for video_or_text in videos_and_texts:
        stripped = video_or_text.strip()
        if Path(stripped).is_file():
            # we have a video, so start a new text block
            # if the previous text block is not empty
            if prompts[-1][0] != "":
                prompts.append(("", 0))

            # process only the first 8 seconds if the video is longer than 8 seconds
            video = video_path_handler.video_from_path(stripped)
            end_sec = min(video.duration, 8)
            clip = video.get_clip(0, end_sec)

            frames = process(
                processor, video=subsampler(clip["video"].to(torch.uint8))
            ).pixel_values.squeeze(0)
            frames_list.append(frames)
            text_block, num_video = prompts[-1]
            prompts[-1] = (text_block, num_video + 1)
        else:
            logging.debug(f'"{stripped}" is not a file, so treating it as text.')
            text_block, num_video = prompts[-1]
            if text_block != "":
                text_block += " "
            text_block += stripped
            prompts[-1] = (text_block, num_video)
    inputs = generate_input_ids_and_labels_from_interleaved(
        processor.tokenizer,
        prompts,
        None,
        model.config.num_query_tokens,
        model.config.use_decoder_only_language_model,
    )

    # process the inputs
    generate_kwargs = {
        "pixel_values": torch.stack(frames_list).to(model.device),
        "input_ids": inputs["input_ids"].unsqueeze(0).to(model.device),
        "video_input_mask": inputs["video_input_mask"].unsqueeze(0).to(model.device),
        "max_new_tokens": 32,
        "num_beams": 5,
        "do_sample": False,
        "length_penalty": -1,
    }
    if model.config.text_config.architectures[0] == "OPTForCausalLM":
        # if the LLM is OPT, set eos_token_id to the newline character as this is the
        # setting used by BLIP-2.
        # https://github.com/salesforce/LAVIS/blob/7f00a0891b2890843f61c002a8e9532a40343648/lavis/models/blip2_models/blip2_opt.py#L91-L93
        generate_kwargs["eos_token_id"] = 50118

    generated_ids = model.generate(**generate_kwargs)  # type: ignore
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()

    print(f"Generated_text: {generated_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate action narrations using an EILEV-trained model."
    )
    parser.add_argument(
        "videos_and_texts",
        nargs="+",
        help="""space separated list of videos and texts, e.g., video_1.mp4 "What is the camera wearer doing? He's fixing a bicycle" video_2.mp4 "What is the camera wearer doing?\"""",  # noqa: E501
    )
    parser.add_argument("--model", default="kpyu/eilev-blip2-opt-2.7b")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model = VideoBlipForConditionalGeneration.from_pretrained(args.model).to(
        args.device
    )
    if args.processor is None:
        args.processor = args.model
    processor = Blip2Processor.from_pretrained(args.processor)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    generate(model, processor, args.videos_and_texts)
