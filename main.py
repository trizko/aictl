import argparse

import torch
from diffusers import StableDiffusionPipeline

def sd(args):
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
    output = pipe(args.prompt)

    print("### saving image files")
    output.images[0].save("output_sd_15.png")

def main():
    parser = argparse.ArgumentParser(description="A command-line interface for ai models")

    subparsers = parser.add_subparsers()

    sd_parser = subparsers.add_parser('sd', help='the stable diffusion subcommand')
    sd_parser.add_argument('-m', '--model', default="runwayml/stable-diffusion-v1-5", help='the model id to use')
    sd_parser.add_argument('-p', '--prompt', default="a photo of an astronaut riding a horse on mars", help='the prompt to use')
    sd_parser.set_defaults(func=sd)

    args = parser.parse_args()

    # Call the function associated with the chosen subcommand
    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
