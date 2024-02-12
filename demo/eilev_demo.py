import argparse
import string
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import gradio as gr
import torch
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms import UniformTemporalSubsample
from transformers import Blip2Processor

from eilev.data.utils import generate_input_ids_and_labels_from_interleaved
from eilev.model.utils import process
from eilev.model.v2 import VideoBlipForConditionalGeneration


@dataclass
class State:
    text_blocks: list[str] = field(default_factory=lambda: [""])
    """list of previous text blocks.

    each text block is punctuated by one or more consecutive videos.
    """
    videos: list[torch.Tensor] = field(default_factory=list)
    """List of preprocessed videos."""
    text_block_video_map: list[list[int]] = field(default_factory=lambda: [[]])
    """Map from text block => list of videos."""


@torch.no_grad()
def respond(
    model: VideoBlipForConditionalGeneration,
    processor: Blip2Processor,
    state: State,
    chat_history: list[list[str | None | tuple]],
    beams: int,
    max_new_tokens: int,
) -> list[list[str | None | tuple]]:
    prompts: list[tuple[str, int]] = []
    for text_block, videos in zip(state.text_blocks, state.text_block_video_map):
        prompts.append((text_block, len(videos)))
    inputs = generate_input_ids_and_labels_from_interleaved(
        processor.tokenizer,
        prompts,
        None,
        model.config.num_query_tokens,
        model.config.use_decoder_only_language_model,
    )

    # process the inputs
    generate_kwargs = {
        "pixel_values": torch.stack(state.videos).to(model.device),
        "input_ids": inputs["input_ids"].unsqueeze(0).to(model.device),
        "video_input_mask": inputs["video_input_mask"].unsqueeze(0).to(model.device),
        "max_new_tokens": max_new_tokens,
        "num_beams": beams,
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

    # if the last character of the generated text is not a punctuation, add a period
    if generated_text and generated_text[-1] not in string.punctuation:
        generated_text += "."

    # add the message to the latest text block
    if state.text_blocks[-1] != "":
        state.text_blocks[-1] += " "
    state.text_blocks[-1] += generated_text
    chat_history.append([None, generated_text])

    return chat_history


EXAMPLES: dict[str, list[str | list[str]]] = {
    "Example from paper": [
        ["examples/dough-mixer.mp4"],
        "Question: What is the camera wearer doing?",
        "Answer: The camera wearer hits the scraper in his right hand on the dough"
        " mixer guard.",
        ["examples/paint.mp4"],
        "Question: What is the camera wearer doing?",
        "Answer: The camera wearer paints the wall in the room with the paint brush.",
        ["examples/trowel.mp4"],
        "Question: What is the camera wearer doing? Answer:",
    ],
    "Bicycle fixing and motorcycle riding": [
        ["examples/bike-fixing-0.mp4"],
        "What is the camera wearer doing?",
        "He is fixing a bicycle handle.",
        ["examples/motorcycle-riding-0.mp4"],
        "What is the camera wearer doing?",
    ],
}


def click_example(
    processor: Blip2Processor,
    video_path_handler: VideoPathHandler,
    example: list[str],
) -> tuple[State, list[list[str | None | tuple]]]:
    example_chat_history = EXAMPLES[example[0]]
    curr_path = Path(__file__).parent
    chat_history: list[list[str | None | tuple]] = []
    state = State()
    for msg in example_chat_history:
        if isinstance(msg, str):
            add_text(msg, state, chat_history)
        else:
            add_files(
                processor,
                video_path_handler,
                state,
                chat_history,
                [str(curr_path / file) for file in msg],
            )
    return state, chat_history


def add_text(
    message: str, state: State, chat_history: list[list[str | None | tuple]]
) -> tuple[str, list[list[str | None | tuple]]]:
    stripped_msg = message.strip()
    # add the message to the latest text block
    if state.text_blocks[-1] != "":
        state.text_blocks[-1] += " "
    state.text_blocks[-1] += stripped_msg

    # add the message to the chat history. No bot message yet.
    chat_history.append([stripped_msg, None])

    # return an empty string to clear out the chat input text box
    return "", chat_history


def add_files(
    processor: Blip2Processor,
    video_path_handler: VideoPathHandler,
    state: State,
    chat_history: list[list[str | None | tuple]],
    files,
) -> list[list[str | None | tuple]]:
    # new images/videos have been added, so start a new text block if the previous text
    # block is not empty.
    if state.text_blocks[-1] != "":
        state.text_blocks.append("")
        state.text_block_video_map.append([])

    # uniformly subsample 8 frames
    subsampler = UniformTemporalSubsample(8)
    for file in files:
        file_name = file if isinstance(file, str) else file.name
        # add the file to the chat history
        chat_history.append([(file_name,), None])

        # process only the first 8 seconds if the video is longer than 8 seconds
        video = video_path_handler.video_from_path(file_name)
        end_sec = min(video.duration, 8)
        clip = video.get_clip(0, end_sec)

        frames = subsampler(clip["video"].to(torch.uint8))

        # process the frames and add it to the video list
        state.videos.append(process(processor, video=frames).pixel_values.squeeze(0))

        # add the mapping
        state.text_block_video_map[-1].append(len(state.videos) - 1)

    return chat_history


def construct_demo(
    model: VideoBlipForConditionalGeneration,
    processor: Blip2Processor,
    video_path_handler: VideoPathHandler,
) -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            """# EILEV Demo
## Have a multi-modal conversation with an EILEV-trained model!
"""
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """**Instructions**
- You can compose messages arbitrarily interleaved with images, videos and texts.
- After entering your text message in the text box, simply press enter or click the **"Send"** button to send it.
- Click the **"Upload"** button to send an image or a video. You can upload multiple files at once.
- When you're done composing your message, click the **"Respond"** button to tell the model to respond."""  # noqa: E501
                )
            with gr.Column():
                gr.Markdown(
                    """**Limitations**
- Please upload only short videos (around 8 seconds) as we have limited computational resources.
- Due to computational limits, the model processes only the first 8 seconds of the uploaded videos."""  # noqa: E501
                )
        with gr.Row():
            with gr.Column(scale=7):
                with gr.Row():
                    chatbot = gr.Chatbot()
                with gr.Row():
                    state = gr.State(value=State)
                    chat_input = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter or click send",
                        container=False,
                        scale=10,
                        min_width=500,
                    )
                    chat_input.submit(
                        add_text,
                        inputs=[chat_input, state, chatbot],
                        outputs=[chat_input, chatbot],
                    )
                    send_button = gr.Button(value="Send")
                    send_button.click(
                        add_text,
                        inputs=[chat_input, state, chatbot],
                        outputs=[chat_input, chatbot],
                    )
                    upload_button = gr.UploadButton(
                        label="Upload",
                        file_types=["image", "video"],
                        file_count="multiple",
                    )
                    upload_button.upload(
                        partial(add_files, processor, video_path_handler),
                        inputs=[state, chatbot, upload_button],
                        outputs=[chatbot],
                    )
            with gr.Column(scale=3):
                respond_button = gr.Button(value="Respond", variant="primary")
                beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="# of beams",
                    render=False,
                )
                max_new_tokens = gr.Slider(
                    minimum=20,
                    maximum=64,
                    value=32,
                    step=1,
                    label="Max new tokens",
                    render=False,
                )
                respond_button.click(
                    partial(respond, model, processor),
                    inputs=[state, chatbot, beams, max_new_tokens],
                    outputs=[chatbot],
                )
                beams.render()
                max_new_tokens.render()
                clear_button = gr.Button(value="Clear")
                clear_button.click(
                    lambda: (State(), "", []),
                    outputs=[state, chat_input, chatbot],
                )

        with gr.Row():
            examples = gr.Dataset(
                label="Examples",
                components=[gr.Textbox(visible=False)],
                samples=[
                    ["Example from paper"],
                    ["Bicycle fixing and motorcycle riding"],
                ],
            )
            examples.click(
                partial(click_example, processor, video_path_handler),
                inputs=[examples],
                outputs=[state, chatbot],
            )
        with gr.Row():
            gr.Markdown(
                "Example videos are from [Ego4D](https://ego4d-data.org/) and Hamzah Tariq ([Pexels](https://www.pexels.com/@hamzah-tariq-28798546/), [Instagram](https://www.instagram.com/hamzah_tariq/))"  # noqa: E501
            )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kpyu/eilev-blip2-opt-2.7b")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--queue", action="store_true", default=False)
    parser.add_argument("--max-size", type=int, default=10)
    args = parser.parse_args()

    model = VideoBlipForConditionalGeneration.from_pretrained(args.model).to(
        args.device
    )
    if args.processor is None:
        args.processor = args.model
    processor = Blip2Processor.from_pretrained(args.processor)
    demo = construct_demo(model, processor, VideoPathHandler())
    if args.queue:
        demo.queue(api_open=False, max_size=args.max_size)
    demo.launch()
