import asyncio
from datetime import datetime
import hashlib
import os
import secrets

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

# Initialize static file server
images_path = "./frontend/public/images/"
if not os.path.exists(images_path):
    os.mkdir(images_path)
app.mount("/build", StaticFiles(directory="frontend/public/build/"), name="static")
app.mount("/images", StaticFiles(directory="frontend/public/images/"), name="images")


# Create a class that will hold our model and lock
class Model:
    def __init__(self):
        self.pipeline = None
        self.lock = asyncio.Lock()


# Create a single instance of our model holder
model = Model()


@app.on_event("startup")
async def load_model():
    logger.info("Loading models and configuration.")
    is_mac = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        is_mac = True
        device = torch.device("mps")
        logger.info("MPS device detected. Using MPS.")

    model_type = torch.float32 if is_mac else torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=model_type
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if is_mac:
        pipe.enable_attention_slicing()
    else:
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
    logger.info("Model finished loading.")

    # assign pipeline to model instance
    model.pipeline = pipe


# Dependency function that returns the model instance
def get_model():
    return model


@app.get("/")
async def read_root():
    return FileResponse("frontend/public/index.html")


@app.get("/health-check/")
async def health_check():
    return {"status": "OK"}


@app.post("/generate/")
async def analyze_image(data: T2IConfig, model_resource: Model = Depends(get_model)):
    # Get preprocessor, pipeline and lock
    pipe = model_resource.pipeline
    lock = model_resource.lock
    async with lock:
        logger.info("Performing inference.")
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
        logger.info("Inference complete.")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_data = secrets.token_bytes(16)
        short_hash = hashlib.sha256(random_data).hexdigest()[:5]
        filename = f"{timestamp}-{short_hash}.png"
        output.images[0].save(f"frontend/public/images/{filename}")
        return {"path": f"images/{filename}"}
