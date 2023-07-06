import asyncio
import base64
import io
import json
import logging
import re
import time

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import Optional

import torch
from controlnet_aux import HEDdetector
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

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
    # Load the model during startup
    is_mac = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        is_mac = True
        device = torch.device("mps")
        print("MPS device detected. Using MPS.")

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

    # assign pipeline to model instance
    model.pipeline = pipe

    print('model loaded and server listening...')

# Dependency function that returns the model instance
def get_model():
    return model

class T2IConfig(BaseModel):
    prompt: Optional[str] = 'a photo of an astronaut riding a horse on mars'
    negative_prompt: Optional[str] = ''
    steps: Optional[int] = 20
    width: Optional[int] = 512
    height: Optional[int] = 512
    cfg: Optional[float] = 7.5
    denoiser: Optional[float] = 0.7
    batch_size: Optional[int] = 1
    seed: Optional[int] = 420

@app.post("/generate/")
async def analyze_image(data: T2IConfig, model_resource: Model = Depends(get_model)):
    # Get preprocessor, pipeline and lock
    pipe = model_resource.pipeline
    lock = model_resource.lock
    async with lock:
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

        buffer = io.BytesIO()
        output.images[0].save(buffer, format="JPEG")
        return {
            "image": base64.b64encode(buffer.getvalue())
        }

