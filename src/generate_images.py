from __future__ import annotations
from pathlib import Path
from typing import List
import torch
from PIL import Image
from utils import load_pipe
from utils import parse_args

if __name__ == "__main__":
    args = parse_args()

    # Basic validations for generation inputs
    prompt = (args.prompt or "").strip()
    if not prompt:
        raise ValueError("--prompt must be a non-empty string")

    if args.num_images is None or args.num_images < 1:
        raise ValueError("--num_images must be >= 1")

    if args.lora_path:
        lp = Path(args.lora_path)
        if not lp.exists():
            raise FileNotFoundError(f"LoRA path not found: {lp}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available")

    pipe = load_pipe(args.lora_path, args.device)

    images: List[Image.Image] = []
    for _ in range(args.num_images):
        img = pipe(args.prompt, num_inference_steps=35).images[0]
        images.append(img)



    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(out_dir / f"img_{i}.png")
