from typing import List, Optional, Any, Dict
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel, PeftConfig
from PIL import Image

import argparse
import yaml

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
    cfg = read_config()

    # Resolve base model from config if available
    base_model = (
        cfg.get("model", {}).get("base_model")
        if isinstance(cfg, dict) else base_model
    ) or base_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve dtype from config if available (fallback to device-based)
    cfg_dtype = None
    try:
        cfg_dtype_name = cfg.get("model", {}).get("dtype") if isinstance(cfg, dict) else None
        if cfg_dtype_name:
            cfg_dtype_name = str(cfg_dtype_name).lower()
            if cfg_dtype_name in ("float16", "fp16", "half"):
                cfg_dtype = torch.float16
            elif cfg_dtype_name in ("float32", "fp32"):
                cfg_dtype = torch.float32
            elif cfg_dtype_name in ("bfloat16", "bf16"):
                cfg_dtype = torch.bfloat16
    except Exception:
        cfg_dtype = None

    dtype = cfg_dtype or (torch.float16 if device == "cuda" else torch.float32)

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
    p.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], default=None)
    return p.parse_args()


def read_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Read YAML config from project root (config.yaml) unless a path is provided.

    Returns an empty dict if the file is missing or cannot be parsed.
    """
    config_path = Path(path) if path else Path(__file__).resolve().parents[1] / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
        return {}