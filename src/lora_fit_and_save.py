from __future__ import annotations
from ImageCaptionDataset import ImageDataset

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils_fit_lora import parse_args
from utils import _safe_enable_xformers, read_config

if __name__ == "__main__":
    print("Начало обучения модели stable diffusion")
    args = parse_args()
    print(f"Args parsed successfully: {args=}")
    logger.info(f"Starting training with args: {args}")

    # Load configuration (optional, values can be overridden by CLI in future)
    cfg = read_config()
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    effective_resolution = int(model_cfg.get("resolution", args.resolution))

    # Basic validations for training inputs
    train_dir = Path(args.train_data_dir)
    if not train_dir.exists() or not train_dir.is_dir():
        raise FileNotFoundError(f"--train_data_dir not found or not a directory: {train_dir}")

    if args.output_folder is None or str(args.output_folder).strip() == "":
        raise ValueError("--output_folder must be provided")

    if args.resolution is None or args.resolution <= 0:
        raise ValueError("--resolution must be a positive integer")

    if args.batch_size is None or args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer")

    if args.epochs is None or args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer")

    if args.learning_rate is None or args.learning_rate <= 0:
        raise ValueError("--learning_rate must be a positive float")

    if args.checkpointing_steps is None or args.checkpointing_steps <= 0:
        raise ValueError("--checkpointing_steps must be a positive integer")

    print("Создание директории")
    logs_dir = Path(args.output_folder) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Accelerator...")
    project_config = ProjectConfiguration(project_dir=str(logs_dir))
    accelerator = Accelerator(log_with="tensorboard", project_config=project_config)
    device = accelerator.device
    logger.warning(f"Accelerator initialized on device: {device}")

    try:
        dataset = ImageDataset(args.train_data_dir, resolution=effective_resolution)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)
    except Exception as e:
        print(str(e))
        raise e


    # Load model & tokenizer
    logger.info("Загрузка Stable Diffusion ")
    print("Загрузка stable-diffusion")
    model_id = model_cfg.get("base_model", "runwayml/stable-diffusion-v1-5")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    dtype_name = str(model_cfg.get("dtype", "")).lower()
    if dtype_name in ("float16", "fp16", "half"):
        dtype = torch.float16
    elif dtype_name in ("float32", "fp32"):
        dtype = torch.float32
    elif dtype_name in ("bfloat16", "bf16"):
        dtype = torch.bfloat16

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    _safe_enable_xformers(pipe)
    tokenizer: AutoTokenizer = pipe.tokenizer

    print("Конфиг модели")
    lora_cfg = (cfg.get("lora", {}) if isinstance(cfg, dict) else {})
    lora_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 2)),
        bias="none",
        target_modules=list(lora_cfg.get("target_modules", ["to_k", "to_q", "to_v"])),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.15)),
    )
    unet = get_peft_model(pipe.unet, lora_config)
    unet.train()

    training_cfg = (cfg.get("training", {}) if isinstance(cfg, dict) else {})
    effective_batch_size = int(training_cfg.get("batch_size", args.batch_size))
    effective_epochs = int(training_cfg.get("epochs", args.epochs))
    effective_lr = float(training_cfg.get("learning_rate", args.learning_rate))
    effective_ckpt_steps = int(training_cfg.get("checkpointing_steps", args.checkpointing_steps))
    effective_num_workers = int(training_cfg.get("num_workers", 20))

    # Rebuild dataloader if num_workers overridden by config
    if effective_num_workers != 20:
        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, num_workers=effective_num_workers)
    else:
        # If batch size overridden but workers not, ensure consistent dataloader
        if effective_batch_size != args.batch_size:
            dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, num_workers=20)

    optimizer = torch.optim.Adam(unet.parameters(), lr=effective_lr)
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    print(f"Starting training for {effective_epochs} epochs...")
    logger.info(f"Starting training for {effective_epochs} epochs...")
    global_step = 0
    for epoch in range(effective_epochs):
        print(f"Эпоха = {epoch + 1}/{effective_epochs}")
        logger.info(f"epoch = {epoch + 1}/{effective_epochs}")
        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=device
                ).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                tokenized = tokenizer(
                    batch["caption"],
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                input_ids = tokenized.input_ids.to(device)
                encoder_hidden_states = pipe.text_encoder(input_ids)[0]

                model_predict = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_predict.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % effective_ckpt_steps == 0:
                ckpt_dir = Path(args.output_folder) / f"step_{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(ckpt_dir)
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
            global_step += 1
        logger.info("Epoch %d complete, loss %.4f", epoch + 1, loss.item())

    print("Сохраняем итоговую модель")
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(args.output_folder)
    pipe.save_pretrained(args.output_folder, safe_serialization=False)
    logger.info(f"Training complete! Model saved to {args.output_folder}")


