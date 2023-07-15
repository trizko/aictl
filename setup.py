from setuptools import setup, find_packages

setup(
    name='aictl',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'accelerate',
        'controlnet_aux',
        'diffusers',
        'fastapi',
        'mediapipe',
        'opencv-contrib-python',
        'pdf2image',
        'Pillow',
        'pytesseract',
        'torch',
        'torchvision',
        'torchaudio',
        'transformers',
        'uvicorn',
        'upscaler': [
            'RealESRGAN @ git+https://github.com/ai-forever/Real-ESRGAN.git',
        ],
    ],
    extras_require={
       'audio': [
            'audiocraft @ git+https://github.com/facebookresearch/audiocraft.git'
        ],
       'dev':  [
            'black',
            'pytest',
            'ruff',
        ],
        'xformers':  [
            'xformers',
        ],
    },
    entry_points={
        'console_scripts': [
            'aictl = aictl.cli.main:main',
            'aictl2 = aictl.cli.main_v2:main',
        ],
    },
)
