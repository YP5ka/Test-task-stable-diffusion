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


def _safe_enable_xformers(pipe: StableDiffusionPipeline) -> None:
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as _:
        pass


if __name__ == "__main__":
    print("Начало обучения модели stable diffusion")
    args = parse_args()
    print(f"Args parsed successfully: {args=}")
    logger.info(f"Starting training with args: {args}")

    print("Создание директории")
    logs_dir = Path(args.output_folder) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Accelerator...")
    project_config = ProjectConfiguration(project_dir=str(logs_dir))
    accelerator = Accelerator(log_with="tensorboard", project_config=project_config)
    device = accelerator.device
    logger.warning(f"Accelerator initialized on device: {device}")

    try:
        dataset = ImageDataset(args.train_data_dir, resolution=args.resolution)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)
    except Exception as e:
        print(str(e))
        raise e


    # Load model & tokenizer
    logger.info("Загрузка Stable Diffusion ")
    print("Загрузка stable-diffusion")
    model_id = "runwayml/stable-diffusion-v1-5"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    _safe_enable_xformers(pipe)
    tokenizer: AutoTokenizer = pipe.tokenizer

    print("Конфиг модели")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=2,
        bias="none",
        target_modules=["to_k", "to_q", "to_v"],
        lora_dropout=0.15,
    )
    unet = get_peft_model(pipe.unet, lora_config)
    unet.train()

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    print(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Starting training for {args.epochs} epochs...")
    global_step = 0
    for epoch in range(args.epochs):
        print(f"Эпоха = {epoch + 1}/{args.epochs}")
        logger.info(f"epoch = {epoch + 1}/{args.epochs}")
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

            if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
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


