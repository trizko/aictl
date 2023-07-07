import argparse
import base64
import json
import os
import requests
import signal
import subprocess
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
    if args.daemon:
        with open('aictl-server.log', 'a') as log_file:
            server = subprocess.Popen(
                'PYTORCH_ENABLE_MPS_FALLBACK=1 python -m uvicorn aictl.server.main:app',
                stdout=log_file,
                stderr=log_file,
                preexec_fn=os.setsid,
                shell=True
            )
            with open('aictl-server.pid', 'w') as pid_file:
                pid_file.write(str(server.pid))
    else:
        try:
            subprocess.run(
                'PYTORCH_ENABLE_MPS_FALLBACK=1 python -m uvicorn aictl.server.main:app',
                check=True,
                shell=True
            )
        except subprocess.CalledProcessError:
            print("Server failed to start.")
        except KeyboardInterrupt:
            print("\nServer stopped by user.")

def status_server(args):
    try:
        response = requests.get('http://localhost:8000/health-check/')
        if response.ok:
            print('Server is up and running.')
        else:
            print('Server has not started.')
    except requests.exceptions.ConnectionError:
        print('Server has not started.')


def stop_server(args):
    try:
        with open('aictl-server.pid', 'r') as pid_file:
            pid = int(pid_file.read())
            os.killpg(pid, signal.SIGTERM)
            print(f'Stopped server with PID {pid}')
    except FileNotFoundError:
        print('PID file not found. Is the server running?')
    except ValueError:
        print('PID file is empty. Is the server running?')
    except ProcessLookupError:
        print(f'No process with PID {pid}.')

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
    start_parser.add_argument('-d', '--daemon', action='store_true', help='starts the server as a background process')
    start_parser.set_defaults(func=start_server)

    status_parser = server_subparsers.add_parser('status', help='get the status of the server')
    status_parser.set_defaults(func=status_server)

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
