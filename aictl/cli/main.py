import argparse
import PIL
import requests
import calendar
from collections import namedtuple
from datetime import datetime

from aictl.common.config import SystemConfig


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


def classify(args):
    from transformers import ViTImageProcessor, ViTForImageClassification

    if args.image is None:
        image = download_image(args.image_url)
    else:
        image = load_image_from_path(args.image)

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])


def segment(args):
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    if args.image is None:
        image = download_image(args.image_url)
    else:
        image = load_image_from_path(args.image)

    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    print(logits)


def t2i(args):
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    # get config for device type
    cfg = SystemConfig()

    # set sd pipeline type
    sd_pipeline_type = StableDiffusionPipeline
    if args.sdxl:
        sd_pipeline_type = StableDiffusionXLPipeline

    # load models and configure pipeline settings
    print("### loading models")
    pipe = sd_pipeline_type.from_pretrained(
        args.model,
        torch_dtype=cfg.model_type,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = args.scheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(cfg.device)
    cfg.set_pipeline_options(pipe)

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
    for i, image in enumerate(output.images):
        image.save(f"{args.output_path.split('.png')[0]}_{i}.png")


def ip2p(args):
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline

    # get config for device type
    cfg = SystemConfig()

    # load models and configure pipeline settings
    print("### loading models")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model,
        torch_dtype=cfg.model_type,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = args.scheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(cfg.device)
    cfg.set_pipeline_options(pipe)

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

    # get config for device type
    cfg = SystemConfig()

    # TODO: remove this block of when https://github.com/pytorch/pytorch/pull/99246 is merged
    if cfg.device == torch.device("mps"):
        print(
            "ERROR: MPS currently does not support text-to-video pipelines. This will only work when the following PR is merged and released: https://github.com/pytorch/pytorch/pull/99246"
        )
        return
        # TODO END

    variant = "fp32" if cfg.device == torch.device("mps") else "fp16"
    pipe = DiffusionPipeline.from_pretrained(
        args.model, torch_dtype=cfg.model_type, variant=variant
    )

    # memory optimization
    pipe = pipe.to(cfg.device)

    cfg.set_pipeline_options(pipe)
    print(args.prompt)
    video_frames = pipe(args.prompt, num_frames=args.frames).frames
    video_path = export_to_video(video_frames, output_video_path=args.output_path)
    print(video_path)


# Text to Audio
def t2a(args):
    import torch
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write

    if torch.backends.mps.is_available():
        print("ERROR: MPS currently does not support text-to-audio pipelines.")
        return

    model = MusicGen.get_pretrained(args.model)

    # Set the duration
    model.set_generation_params(duration=int(args.duration))

    # Set the description, for now one at a time
    wav = model.generate([args.prompt])
    # Write to file
    audio_write(
        args.output_path,
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True,
    )


def upscale(args):
    from RealESRGAN import RealESRGAN
    from diffusers import StableDiffusionUpscalePipeline

    # get config for device type
    cfg = SystemConfig()

    # Get the image
    if args.image is None:
        image = download_image(args.image_url)
    else:
        image = load_image_from_path(args.image)

    # SDX4 Upscaler
    if args.model == "sdx4":
        # load models and configure pipeline settings
        print("### loading models")
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, torch_dtype=cfg.model_type
        )
        pipe.scheduler = args.scheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(cfg.device)
        cfg.set_pipeline_options(pipe)
        upscaled_image = pipe(prompt=args.prompt, image=image).images[0]
        upscaled_image.save(args.output_path)
    elif args.model == "esrgan":
        model = RealESRGAN(cfg.device, scale=args.scale)
        model.load_weights(f"weights/RealESRGAN_x{args.scale}.pth", download=True)
        upscaled_image = model.predict(image)
        upscaled_image.save(args.output_path)


# Text to Text
def t2t(args):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    # get config for device type
    cfg = SystemConfig()

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model, device_map="auto", torch_dtype=cfg.model_type
    )

    input_text = args.prompt
    print(f"Input: {args.prompt}")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(cfg.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    output_text = tokenizer.decode(outputs.squeeze(), skip_special_tokens=True)
    print(f"Output: {output_text}")


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

    # fmt: off
    t2i_parser = subparsers.add_parser("t2i", help="a text-to-image subcommand")
    t2i_parser.add_argument("-m", "--model", default="runwayml/stable-diffusion-v1-5", help="the model id to use")
    t2i_parser.add_argument("-l", "--sdxl", action="store_true", help="use SDXL model and pipelines instead of SD 1.5")
    t2i_parser.add_argument("-p", "--prompt", default="a photo of an astronaut riding a horse on mars", help="the prompt to use")
    t2i_parser.add_argument("-x", "--seed", default="420", help="seed for pinning random generations", type=int)
    t2i_parser.add_argument("-s", "--steps", default="20", help="number of generation steps", type=int)
    t2i_parser.add_argument("-n", "--negative-prompt", default="", help="prompt keywords to be excluded")
    t2i_parser.add_argument("-y", "--scheduler", default="ddim", help="available schedulers are: lms, ddim, dpm, euler, pndm, ddpm, and eulera", type=scheduler_validator)
    t2i_parser.add_argument("-r", "--resolution", default="512x512", help="the resolution of the image delimited by an 'x' (e.g. 512x512)", type=resolution_validator)
    t2i_parser.add_argument("-c", "--cfg", default="7.5", help="higher values tell the image gen to follow the prompt more closely (default=7.5)", type=float)
    t2i_parser.add_argument("-d", "--denoiser", default="0.7", help="modulate the influence of guidance images on the denoising process (default=0.7)", type=float)
    t2i_parser.add_argument("-b", "--batch-size", default="1", help="number of images per generation", type=int)
    t2i_parser.add_argument("-o", "--output-path", default=f"output_t2i{utc_time}.png", help="path for image output when generation is complete")
    t2i_parser.set_defaults(func=t2i)

    classify_parser = subparsers.add_parser("classify", help="an image classification subcommand")
    classify_parser.add_argument("-m","--model", default="timbrooks/instruct-pix2pix", help="the model id to use")
    classify_parser.add_argument("-i", "--image", default=None, help="the local image file to classify")
    classify_parser.add_argument("-u","--image-url", default="https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg", help="the url of the image to classify")
    classify_parser.set_defaults(func=classify)

    segment_parser = subparsers.add_parser("segment", help="an image segmentation subcommand")
    segment_parser.add_argument("-m","--model", default="timbrooks/instruct-pix2pix", help="the model id to use")
    segment_parser.add_argument("-i", "--image", default=None, help="the local image file to segment")
    segment_parser.add_argument("-u","--image-url", default="https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg", help="the url of the image to segment")
    segment_parser.set_defaults(func=segment)

    ip2p_parser = subparsers.add_parser("ip2p", help="an instruct-pix2pix subcommand")
    ip2p_parser.add_argument("-m","--model", default="timbrooks/instruct-pix2pix", help="the model id to use")
    ip2p_parser.add_argument("-p","--prompt", default="turn him into cyborg", help="the instruction prompt to use")
    ip2p_parser.add_argument("-i", "--image", default=None, help="the local image file to edit")
    ip2p_parser.add_argument("-u","--image-url", default="https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg", help="the url of the image to edit")
    ip2p_parser.add_argument("-x","--seed", default="420", help="seed for pinning random generations", type=int)
    ip2p_parser.add_argument("-s", "--steps", default="10", help="number of generation steps", type=int)
    ip2p_parser.add_argument("-n", "--negative-prompt", default="", help="prompt keywords to be excluded")
    ip2p_parser.add_argument("-y","--scheduler", default="eulera", help="available schedulers are: lms, ddim, dpm, euler, pndm, ddpm, and eulera", type=scheduler_validator)
    ip2p_parser.add_argument("-c","--cfg", default="1.0", help="higher values tell the image gen to follow the prompt more closely (default=7.5)", type=float)
    ip2p_parser.add_argument("-o","--output-path", default=f"output_ip2p{utc_time}.png", help="path for image output when generation is complete")
    ip2p_parser.set_defaults(func=ip2p)

    t2v_parser = subparsers.add_parser("t2v", help="a text-to-video subcommand")
    t2v_parser.add_argument("-m","--model", default="damo-vilab/text-to-video-ms-1.7b", help="the model id to use")
    t2v_parser.add_argument("-p", "--prompt", default="Darth Vader surfing a wave", help="the prompt to use")
    t2v_parser.add_argument("-f", "--frames", default="16", help="number of frames generated", type=int)
    t2v_parser.add_argument("-o","--output-path", default=f"output_t2v{utc_time}.mp4", help="the path for video when generation is complete")
    t2v_parser.set_defaults(func=t2v)

    t2a_parser = subparsers.add_parser("t2a", help="a text-to-audio subcommand")
    t2a_parser.add_argument("-p", "--prompt", default="Greek folk", help="the prompt to use")
    t2a_parser.add_argument("-m","--model", default="small", help="the MusicGen model to use (options: small, medium, large, melody)")
    t2a_parser.add_argument("-o","--output-path", default=f"output_t2a{utc_time}", help="the path for audio when generation is complete")
    t2a_parser.add_argument("-d", "--duration", default="8", help="how long the audio lasts in seconds")
    t2a_parser.set_defaults(func=t2a)

    upscale_parser = subparsers.add_parser("upscale", help="an image upscaling subcommand")
    upscale_parser.add_argument("-p", "--prompt", default="", help="the prompt to use, only works with sdx4")
    upscale_parser.add_argument("-i", "--image", default=None, help="the local image file to edit")
    upscale_parser.add_argument("-u", "--image-url", default="https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg", help="the url of the image to edit")
    upscale_parser.add_argument("-y", "--scheduler", default="euler", help="available schedulers are: lms, ddim, dpm, euler, pndm, ddpm, and eulera", type=scheduler_validator)
    upscale_parser.add_argument("-o", "--output-path", default=f"output_upscale{utc_time}.png", help="the path for audio when generation is complete")
    upscale_parser.add_argument("-m", "--model", default="esrgan", help="the upscale model (x4) to use (options: esrgan,sdx4)")
    upscale_parser.add_argument("-s", "--scale", default="4", help="the scale factor for the upscale", type=int)
    upscale_parser.set_defaults(func=upscale)
    
    # other options for text generation can be found here: https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationConfig
    t2t_parser = subparsers.add_parser("t2t", help="a text generation subcommand")
    t2t_parser.add_argument("-p", "--prompt", default="What color is the sky?", help="the prompt to use, ask a question")
    t2t_parser.add_argument("-m", "--model", default="google/flan-t5-base", help="The model to use")
    t2t_parser.add_argument("-n", "--max-new-tokens", default="256", help="maximum numbers of tokens to generate (not including prompt)", type=int)
    t2t_parser.add_argument("-t", "--temp", default="1.0", help="value used to modulate the next token probabilities.", type=float)
    t2t_parser.add_argument("-k", "--top-k", default="50", help="number of highest probability vocabulary tokens to keep for top-k-filtering", type=int)
    t2t_parser.add_argument("-b", "--top-p", default="1.0", help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation", type=float)
    t2t_parser.set_defaults(func=t2t)
    # fmt: on

    args = parser.parse_args()

    # Call the function associated with the chosen subcommand
    if "func" in args:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
