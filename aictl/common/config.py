import torch
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

@dataclass
class SystemConfig():
    device: torch.device = torch.device("cpu")
    device_type: DeviceType = DeviceType.CPU
    model_type: torch.dtype = torch.float16

    def __init__(self, **kwargs):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.device_type = DeviceType.CUDA
            self.model_type = torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_type = DeviceType.MPS
            self.model_type = torch.float32
        else:
            self.device = torch.device("cpu")
            self.device_type = DeviceType.CPU
            self.model_type = torch.float16
    
    def set_pipeline_options(self, pipeline):
        if self.device_type == DeviceType.CUDA:
            pipeline.enable_model_cpu_offload()
            pipeline.enable_xformers_memory_efficient_attention()
        elif self.device_type == DeviceType.MPS:
            pipeline.enable_attention_slicing()

