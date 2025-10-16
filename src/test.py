import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

