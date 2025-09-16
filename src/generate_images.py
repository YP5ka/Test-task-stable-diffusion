from __future__ import annotations
from pathlib import Path
from typing import List
import torch
from PIL import Image
from utils import load_pipe, parse_args

if __name__ == "__main__":
    args = parse_args()

    pipe = load_pipe(args.lora_path)

    images: List[Image.Image] = []
    for _ in range(args.num_images):
        img = pipe(args.prompt, num_inference_steps=35).images[0]
        images.append(img)



    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(out_dir / f"img_{i}.png")
