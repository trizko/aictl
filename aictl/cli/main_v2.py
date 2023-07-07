import argparse
import base64
import json
import requests
from collections import namedtuple

from aictl.common.types import T2IConfig

# define named tuple for the size of result image
Size = namedtuple('Size', 'width height')

def sd(args):
    payload = T2IConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        width=args.resolution.width,
        height=args.resolution.height,
        cfg=args.cfg,
        denoiser=args.denoiser,
        batch_size=args.batch_size,
        seed=args.seed
    )
    response = requests.post('http://localhost:8000/generate/', json=payload.dict())
    image_b64 = json.loads(response.text)['image']

    with open(args.output_path, 'wb') as file:
        file.write(base64.b64decode(image_b64))

    print(f'Image successfully written to {args.output_path}')


def resolution_validation(x):
    x = x.split('x')
    return Size(int(x[0]), int(x[1]))

def start_server(args):
    print(args)
    pass

def stop_server(args):
    print(args)
    pass

def main():
    parser = argparse.ArgumentParser(description="A command-line interface for ai models")

    subparsers = parser.add_subparsers()

    # cli parser
    sd_parser = subparsers.add_parser('sd', help='the stable diffusion subcommand')
    sd_parser.add_argument('-m', '--model', default='runwayml/stable-diffusion-v1-5', help='the model id to use')
    sd_parser.add_argument('-p', '--prompt', default='a photo of an astronaut riding a horse on mars', help='the prompt to use')
    sd_parser.add_argument('-x', '--seed', default='420', help='seed for pinning random generations', type=int)
    sd_parser.add_argument('-s', '--steps', default='20', help='number of generation steps', type=int)
    sd_parser.add_argument('-n', '--negative-prompt', default='', help='prompt keywords to be excluded')
    sd_parser.add_argument('-y', '--scheduler', default='ddim', help='available schedulers are: lms, ddim, dpm, euler, pndm, ddpm, and eulera')
    sd_parser.add_argument('-r', '--resolution', default='512x512', help='the resolution of the image delimited by an \'x\' (e.g. 512x512)', type=resolution_validation)
    sd_parser.add_argument('-c', '--cfg', default='7.5', help='higher values tell the image gen to follow the prompt more closely (default=7.5)', type=float)
    sd_parser.add_argument('-d', '--denoiser', default='0.7', help='modulate the influence of guidance images on the denoising process (default=0.7)', type=float)
    sd_parser.add_argument('-b', '--batch-size', default='1', help='number of images per generation', type=int)
    sd_parser.add_argument('-o', '--output-path', default='output_sd_15.png', help='path for image output when generation is complete')
    sd_parser.set_defaults(func=sd)

    # server parser
    server_parser = subparsers.add_parser('server', help='the server subcommand')
    server_subparsers = server_parser.add_subparsers()

    start_parser = server_subparsers.add_parser('start', help='start the server')
    start_parser.set_defaults(func=start_server)

    stop_parser = server_subparsers.add_parser('stop', help='stop the server')
    stop_parser.set_defaults(func=stop_server)

    # parse args for all parsers
    args = parser.parse_args()

    # Call the function associated with the chosen subcommand
    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
