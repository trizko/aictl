import asyncio
import base64
import io
import json
import re
import time

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

import logging
from logging.config import dictConfig
from .config import LogConfig

from aictl.common.types import T2IConfig

# Initialize logger
dictConfig(LogConfig().dict())
logger = logging.getLogger("aictl")

# Initialize the FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a class that will hold our model and lock
class Model:
    def __init__(self):
        self.pipeline = None
        self.lock = asyncio.Lock()

# Create a single instance of our model holder
model = Model()

@app.on_event("startup")
async def load_model():
    logger.info('Loading models and configuration.')
    is_mac = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        is_mac = True
        device = torch.device("mps")
        logger.info("MPS device detected. Using MPS.")

    model_type = torch.float32 if is_mac else torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', 
        torch_dtype=model_type
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if is_mac:
        pipe.enable_attention_slicing()
    else:
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
    logger.info('Model finished loading.')

    # assign pipeline to model instance
    model.pipeline = pipe


# Dependency function that returns the model instance
def get_model():
    return model

@app.get("/health-check/")
async def health_check():
    return { 'status': 'OK' }

@app.post("/generate/")
async def analyze_image(data: T2IConfig, model_resource: Model = Depends(get_model)):
    # Get preprocessor, pipeline and lock
    pipe = model_resource.pipeline
    lock = model_resource.lock
    async with lock:
        logger.info('Performing inference.')
        output = pipe(
            prompt=data.prompt,
            num_inference_steps=data.steps,
            negative_prompt=data.negative_prompt,
            width=data.width,
            height=data.height,
            guidance_scale=data.cfg,
            guidance_rescale=data.denoiser,
            num_images_per_prompt=data.batch_size,
            generator=torch.Generator(device="cpu").manual_seed(data.seed),
        )
        logger.info('Inference complete.')

        buffer = io.BytesIO()
        output.images[0].save(buffer, format="JPEG")
        return {
            "image": base64.b64encode(buffer.getvalue())
        }

