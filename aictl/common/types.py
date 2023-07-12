from pydantic import BaseModel
from typing import Optional


class T2IConfig(BaseModel):
    prompt: Optional[str] = "a photo of an astronaut riding a horse on mars"
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = 20
    width: Optional[int] = 512
    height: Optional[int] = 512
    cfg: Optional[float] = 7.5
    denoiser: Optional[float] = 0.7
    batch_size: Optional[int] = 1
    seed: Optional[int] = 420
