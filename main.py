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
    if is_mac:
        pipe.enable_attention_slicing()
    else:
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)

    print("### performing inference with args:")
    print("# model id: ", args.model)
    print("# prompt: ", args.prompt)
    print("# steps: ", args.steps)
    output = pipe(
        prompt=args.prompt,
        num_inference_steps=int(args.steps)
    )

    print("### saving image files")
    output.images[0].save(args.output_path)

def resolution_validation(x):
    x = x.split('x')
    return Size(int(x[0]), int(x[1]))

def main():
    parser = argparse.ArgumentParser(description="A command-line interface for ai models")

    subparsers = parser.add_subparsers()

    sd_parser = subparsers.add_parser('sd', help='the stable diffusion subcommand')
    sd_parser.add_argument('-m', '--model', default='runwayml/stable-diffusion-v1-5', help='the model id to use')
    sd_parser.add_argument('-p', '--prompt', default='a photo of an astronaut riding a horse on mars', help='the prompt to use')
    sd_parser.add_argument('-s', '--steps', default='20', help='number of generation steps')
    sd_parser.add_argument('-r', '--resolution', default='512x512', help='the resolution of the image delimited by an \'x\' (e.g. 512x512)', type=resolution_validation)
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
