import argparse
from collections import namedtuple

# define named tuple for the size of result image
Size = namedtuple('Size', 'width height')

def sd(args):
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
        torch_dtype=model_type
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

def resolution_validation(x):
    x = x.split('x')
    return Size(int(x[0]), int(x[1]))

def scheduler_validation(sampler):
    from diffusers import \
        LMSDiscreteScheduler, \
        DDIMScheduler, \
        DPMSolverMultistepScheduler, \
        EulerDiscreteScheduler, \
        PNDMScheduler, \
        DDPMScheduler, \
        EulerAncestralDiscreteScheduler

    if sampler == 'lms':
        return LMSDiscreteScheduler
    elif sampler == 'ddim':
        return DDIMScheduler
    elif sampler == 'dpm':
        return DPMSolverMultistepScheduler
    elif sampler == 'euler':
        return EulerDiscreteScheduler
    elif sampler == 'pndm':
        return PNDMScheduler
    elif sampler == 'ddpm':
        return DDPMScheduler
    elif sampler == 'eulera':
        return EulerAncestralDiscreteScheduler

    return EulerAncestralDiscreteScheduler


def main():
    parser = argparse.ArgumentParser(description="A command-line interface for ai models")

    subparsers = parser.add_subparsers()

    sd_parser = subparsers.add_parser('sd', help='the stable diffusion subcommand')
    sd_parser.add_argument('-m', '--model', default='runwayml/stable-diffusion-v1-5', help='the model id to use')
    sd_parser.add_argument('-p', '--prompt', default='a photo of an astronaut riding a horse on mars', help='the prompt to use')
    sd_parser.add_argument('-x', '--seed', default='420', help='seed for pinning random generations', type=int)
    sd_parser.add_argument('-s', '--steps', default='20', help='number of generation steps', type=int)
    sd_parser.add_argument('-n', '--negative-prompt', default='', help='prompt keywords to be excluded')
    sd_parser.add_argument('-y', '--scheduler', default='ddim', help='available schedulers are: lms, ddim, dpm, euler, pndm, ddpm, and eulera', type=scheduler_validation)
    sd_parser.add_argument('-r', '--resolution', default='512x512', help='the resolution of the image delimited by an \'x\' (e.g. 512x512)', type=resolution_validation)
    sd_parser.add_argument('-c', '--cfg', default='7.5', help='higher values tell the image gen to follow the prompt more closely (default=7.5)', type=float)
    sd_parser.add_argument('-d', '--denoiser', default='0.7', help='modulate the influence of guidance images on the denoising process (default=0.7)', type=float)
    sd_parser.add_argument('-b', '--batch-size', default='1', help='number of images per generation', type=int)
    sd_parser.add_argument('-o', '--output-path', default='output_sd_15.png', help='path for image output when generation is complete')
    sd_parser.set_defaults(func=sd)

    args = parser.parse_args()

    # Call the function associated with the chosen subcommand
    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
