import subprocess
from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option('--data-dir', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--output-dir', required=True, type=click.Path(file_okay=False))
def train(data_dir, output_dir):
    """Запуск обучения LoRA на Stable Diffusion."""
    cmd = [
        'python', 'src/lora_fit_and_save.py',
        '--train_data_dir', str(Path(data_dir)),
        '--output_folder', str(Path(output_dir)),
    ]
    raise SystemExit(subprocess.call(cmd))


@cli.command()
@click.option('--lora-path', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--prompt', required=True, type=str)
@click.option('--num-images', default=4, type=int)
@click.option('--output-dir', default='outputs/gen', type=click.Path(file_okay=False))
@click.option('--device', type=click.Choice(['cuda', 'cpu', 'mps']), default=None)
def generate(lora_path, prompt, num_images, output_dir, device):
    """Запуск генерации изображений с обученной LoRA."""
    cmd = [
        'python', 'src/generate_images.py',
        '--prompt', prompt,
        '--lora_path', str(Path(lora_path)),
        '--num_images', str(num_images),
        '--output_dir', str(Path(output_dir)),
    ]
    if device:
        cmd += ['--device', device]
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    cli()


