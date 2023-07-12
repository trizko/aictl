import argparse
import PIL
import requests
import calendar
from collections import namedtuple
from datetime import datetime


# define named tuple for the size of result image
Size = namedtuple("Size", "width height")


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def load_image_from_path(path):
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def t2i(args):
    import torch
    from diffusers import StableDiffusionPipeline

    # check if on mac and mps is available, fallback to cuda then cpu
    is_mac = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        is_mac = True
        device = torch.device("mps")
        print("MPS device detected. Using MPS.")

    # load models and configure pipeline settings
    print("### loading models")
    model_type = torch.float32 if is_mac else torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=model_type,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = args.scheduler.from_config(pipe.scheduler.config)
    if is_mac:
        pipe.enable_attention_slicing()
    else:
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)

    print("### performing inference with args:")
    print("# args: ", args)
    output = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        negative_prompt=args.negative_prompt,
        width=args.resolution.width,
        height=args.resolution.height,
        guidance_scale=args.cfg,
        guidance_rescale=args.denoiser,
        num_images_per_prompt=args.batch_size,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )

    print("### saving image files")
    output.images[0].save(args.output_path)


def ip2p(args):
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline

    # check if on mac and mps is available, fallback to cuda then cpu
    is_mac = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        is_mac = True
        device = torch.device("mps")
        print("MPS device detected. Using MPS.")

    # load models and configure pipeline settings
    print("### loading models")
    model_type = torch.float32 if is_mac else torch.float16
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model,
        torch_dtype=model_type,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = args.scheduler.from_config(pipe.scheduler.config)
    if is_mac:
        pipe.enable_attention_slicing()
    else:
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)

    if args.image is None:
        image = download_image(args.image_url)
    else:
        image = load_image_from_path(args.image)

    print("### performing inference with args:")
    print("# args: ", args)
    output = pipe(
        prompt=args.prompt,
        image=image,
        num_inference_steps=args.steps,
        negative_prompt=args.negative_prompt,
        image_guidance_scale=args.cfg,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )

    print("### saving image files")
    output.images[0].save(args.output_path)


def t2v(args):
    import torch
    from diffusers import DiffusionPipeline
    from diffusers.utils import export_to_video

    # check if on mac and mps is available, fallback to cuda then cpu
    is_mac = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        # TODO: remove this block of when https://github.com/pytorch/pytorch/pull/99246 is merged
        print(
            "ERROR: MPS currently does not support text-to-video pipelines. This will only work when the following PR is merged and released: https://github.com/pytorch/pytorch/pull/99246"
        )
        return
        # TODO END

        is_mac = True
        device = torch.device("mps")
        print("MPS device detected. Using MPS.")

    model_type = torch.float32 if is_mac else torch.float16
    variant = "fp32" if is_mac else "fp16"
    pipe = DiffusionPipeline.from_pretrained(
        args.model, torch_dtype=model_type, variant=variant
    )

    # memory optimization
    if is_mac:
        pipe.enable_attention_slicing()
    else:
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)

    print(args.prompt)
    video_frames = pipe(args.prompt, num_frames=args.frames).frames
    video_path = export_to_video(video_frames, output_video_path=args.output_path)
    print(video_path)


def resolution_validator(x):
    x = x.split("x")
    return Size(int(x[0]), int(x[1]))


def scheduler_validator(sampler):
    from diffusers import (
        LMSDiscreteScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        PNDMScheduler,
        DDPMScheduler,
        EulerAncestralDiscreteScheduler,
    )

    if sampler == "lms":
        return LMSDiscreteScheduler
    elif sampler == "ddim":
        return DDIMScheduler
    elif sampler == "dpm":
        return DPMSolverMultistepScheduler
    elif sampler == "euler":
        return EulerDiscreteScheduler
    elif sampler == "pndm":
        return PNDMScheduler
    elif sampler == "ddpm":
        return DDPMScheduler
    elif sampler == "eulera":
        return EulerAncestralDiscreteScheduler

    return EulerAncestralDiscreteScheduler


def main():
    parser = argparse.ArgumentParser(
        description="A command-line interface for ai models"
    )

    subparsers = parser.add_subparsers()

    # Set the timestamp
    today_time = datetime.now()
    utc_time = calendar.timegm(today_time.utctimetuple())

    t2i_parser = subparsers.add_parser("t2i", help="the text-to-image subcommand")
    t2i_parser.add_argument(
        "-m",
        "--model",
        default="runwayml/stable-diffusion-v1-5",
        help="the model id to use",
    )
    t2i_parser.add_argument(
        "-p",
        "--prompt",
        default="a photo of an astronaut riding a horse on mars",
        help="the prompt to use",
    )
    t2i_parser.add_argument(
        "-x",
        "--seed",
        default="420",
        help="seed for pinning random generations",
        type=int,
    )
    t2i_parser.add_argument(
        "-s", "--steps", default="20", help="number of generation steps", type=int
    )
    t2i_parser.add_argument(
        "-n", "--negative-prompt", default="", help="prompt keywords to be excluded"
    )
    t2i_parser.add_argument(
        "-y",
        "--scheduler",
        default="ddim",
        help="available schedulers are: lms, ddim, dpm, euler, pndm, ddpm, and eulera",
        type=scheduler_validator,
    )
    t2i_parser.add_argument(
        "-r",
        "--resolution",
        default="512x512",
        help="the resolution of the image delimited by an 'x' (e.g. 512x512)",
        type=resolution_validator,
    )
    t2i_parser.add_argument(
        "-c",
        "--cfg",
        default="7.5",
        help="higher values tell the image gen to follow the prompt more closely (default=7.5)",
        type=float,
    )
    t2i_parser.add_argument(
        "-d",
        "--denoiser",
        default="0.7",
        help="modulate the influence of guidance images on the denoising process (default=0.7)",
        type=float,
    )
    t2i_parser.add_argument(
        "-b",
        "--batch-size",
        default="1",
        help="number of images per generation",
        type=int,
    )
    t2i_parser.add_argument(
        "-o",
        "--output-path",
        default=f"output_t2i{utc_time}.png",
        help="path for image output when generation is complete",
    )
    t2i_parser.set_defaults(func=t2i)

    ip2p_parser = subparsers.add_parser("ip2p", help="the instruct-pix2pix subcommand")
    ip2p_parser.add_argument(
        "-m",
        "--model",
        default="timbrooks/instruct-pix2pix",
        help="the model id to use",
    )
    ip2p_parser.add_argument(
        "-p",
        "--prompt",
        default="turn him into cyborg",
        help="the instruction prompt to use",
    )
    ip2p_parser.add_argument(
        "-i", "--image", default=None, help="the local image file to edit"
    )
    ip2p_parser.add_argument(
        "-u",
        "--image-url",
        default="https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg",
        help="the url of the image to edit",
    )
    ip2p_parser.add_argument(
        "-x",
        "--seed",
        default="420",
        help="seed for pinning random generations",
        type=int,
    )
    ip2p_parser.add_argument(
        "-s", "--steps", default="10", help="number of generation steps", type=int
    )
    ip2p_parser.add_argument(
        "-n", "--negative-prompt", default="", help="prompt keywords to be excluded"
    )
    ip2p_parser.add_argument(
        "-y",
        "--scheduler",
        default="eulera",
        help="available schedulers are: lms, ddim, dpm, euler, pndm, ddpm, and eulera",
        type=scheduler_validator,
    )
    ip2p_parser.add_argument(
        "-c",
        "--cfg",
        default="1.0",
        help="higher values tell the image gen to follow the prompt more closely (default=7.5)",
        type=float,
    )
    ip2p_parser.add_argument(
        "-o",
        "--output-path",
        default=f"output_ip2p{utc_time}.png",
        help="path for image output when generation is complete",
    )
    ip2p_parser.set_defaults(func=ip2p)

    t2v_parser = subparsers.add_parser("t2v", help="the text-to-video subcommand")
    t2v_parser.add_argument(
        "-m",
        "--model",
        default="damo-vilab/text-to-video-ms-1.7b",
        help="the model id to use",
    )
    t2v_parser.add_argument(
        "-p", "--prompt", default="Darth Vader surfing a wave", help="the prompt to use"
    )
    t2v_parser.add_argument(
        "-f", "--frames", default="16", help="number of frames generated", type=int
    )
    t2v_parser.add_argument(
        "-o",
        "--output-path",
        default=f"output_t2v{utc_time}.mp4",
        help="the path for video when generation is complete",
    )
    t2v_parser.set_defaults(func=t2v)

    args = parser.parse_args()

    # Call the function associated with the chosen subcommand
    if "func" in args:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
