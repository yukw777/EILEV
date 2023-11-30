import argparse

from pytorchvideo.data.video import VideoPathHandler
from transformers import Blip2Processor

from eilev.model.utils import process
from eilev.model.v1 import VideoBlipForConditionalGeneration


def generate(
    model: VideoBlipForConditionalGeneration,
    processor: Blip2Processor,
    video_path: str,
    prompt: str,
) -> None:
    video_path_handler = VideoPathHandler()
    # process only the first 10 seconds
    clip = video_path_handler.video_from_path(video_path).get_clip(0, 10)

    # sample a frame every 30 frames, i.e. 1 fps. We assume the video is 30 fps for now.
    frames = clip["video"][:, ::30, ...].unsqueeze(0)

    inputs = process(processor, video=frames, text=prompt.strip()).to(model.device)
    generated_ids = model.generate(
        **inputs,
        num_beams=4,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.5,
        do_sample=True,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()
    print(f"Generated_text: {generated_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an action narration using VideoBLIP."
    )
    parser.add_argument("video")
    parser.add_argument("prompt")
    parser.add_argument("--model", default="kpyu/video-blip-flan-t5-xl-ego4d")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = VideoBlipForConditionalGeneration.from_pretrained(args.model).to(
        args.device
    )
    if args.processor is None:
        args.processor = args.model
    processor = Blip2Processor.from_pretrained(args.processor)

    generate(model, processor, args.video, args.prompt)
