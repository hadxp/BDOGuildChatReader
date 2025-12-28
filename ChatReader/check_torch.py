import torch
print(f"CUDA is available: {torch.cuda.is_available()}")
print("Device: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
