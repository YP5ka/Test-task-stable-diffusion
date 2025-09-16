from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel, PeftConfig
from PIL import Image

import argparse

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_enable_xformers(pipe: StableDiffusionPipeline) -> None:
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass


def load_pipe(
        lora_path,
        device,
        base_model = "runwayml/stable-diffusion-v1-5"):
    logger.info("start load pipeline")
    base_model = "runwayml/stable-diffusion-v1-5"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe = pipe.to(device)
    _safe_enable_xformers(pipe)

    if lora_path:
        print("lora_path exist")
        config = PeftConfig.from_pretrained(lora_path)
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path, config=config)
        pipe.unet.eval()
    logger.warning("load pipeline done")
    return pipe


def parse_args():
    p = argparse.ArgumentParser(description="Generate images with (optionally) LoRA adapter")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--lora_path", type=str, help="Path to folder with LoRA weights (peft model)")
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="outputs/gen")
    return p.parse_args()