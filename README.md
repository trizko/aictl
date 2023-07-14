# aictl
![ci-python](https://github.com/trizko/aictl/actions/workflows/ci.yml/badge.svg)

`aictl` is a commandline tool for running ai models on mac and linux.

## Installation
Depending on your platform, you will need to follow the following steps to install this tool:
```bash
# set up your virtualenv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

# on Mac
pip install --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu .

# on Linux (if you have an nvidia gpu)
pip install --extra-index-url https://download.pytorch.org/whl/cu116 .[xformers]
```

## Usage
Like the installation, usage of this tool will also depend on your platform:
```bash
# on Mac
PYTORCH_ENABLE_MPS_FALLBACK=1 aictl --help

# on Linux
aictl --help
```

## Audio and Video output models
For the `t2v`, `t2a`, and other future commands that output audio/video, Mac is currently not supported and so we don't include the dependencies by default. If you would like to use audio commands, you will have to install those specific dependencies by running the install with the `audio` feature, like so:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu116 .[audio]
```
(NOTE: For video, the dependencies are all the same and the only thing that's holding back Mac from performing text-to-video functions is [this PR](https://github.com/pytorch/pytorch/pull/99246))

## Development
To install `aictl` for development, you will want to add the `dev` flag in the install command. This flag will download development dependencies such as the linter and formatter. To install with this flag, run one of the following commands:
```bash
# on Mac
pip install --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu .[dev]

# on Linux (if you have an nvidia gpu)
pip install --extra-index-url https://download.pytorch.org/whl/cu116 .[dev,xformers]
```
To test your changes, run the code with the following commands:
```bash
# on Mac
PYTORCH_ENABLE_MPS_FALLBACK=1 python aictl/cli/main.py <args>

# on Linux
python aictl/cli/main.py <args>
```
Before commiting changes to the repo, please run the linter and formatter:
```bash
black aictl
ruff aictl
```
Check the output of these commands and make sure these commands do not output any warnings or errors.

## Roadmap
The following items are either in progress or will be in future iterations:
- [ ] Ability to upscale images/video
- [ ] Text-to-audio/video model support for Mac
- [ ] Frontend webapp
- [ ] Server to support the frontend
- [ ] Support for fine-tuning and training models
- [ ] Ability to add ControlNet into pipelines
- [ ] Segmentation model support
- [ ] SDXL support (when model weights are available)