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