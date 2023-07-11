# aictl

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

# on Linux
pip install --extra-index-url https://download.pytorch.org/whl/cu116 .
```

## Usage
Like the installation, usage of this tool will also depend on your platform:
```bash
# on Mac
PYTORCH_ENABLE_MPS_FALLBACK=1 aictl --help

# on Linux
aictl --help
```

## Development
If you would like to contribute to the project, you could use the above methods for installation and usage to test your changes. However, it's much faster to skip `pip install` every time you make changes. You should only need to run `pip install` when you change anything in the dependencies. To test your changes, run the code with the following commands:
```bash
# on Mac
PYTORCH_ENABLE_MPS_FALLBACK=1 python aictl/cli/main.py <args>

# on Linux
python aictl/cli/main.py <args>
```

## Roadmap
The following items are either in progress or will be in future iterations:
- [ ] Ability to upscale images/video
- [ ] Text-to-audio model support
- [ ] Frontend webapp
- [ ] Server to support the frontend
- [ ] Support for fine-tuning and training models
- [ ] Ability to add ControlNet into pipelines
- [ ] Segmentation model support
- [ ] SDXL support (when model weights are available)