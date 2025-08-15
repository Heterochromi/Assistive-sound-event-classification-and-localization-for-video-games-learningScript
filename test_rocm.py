import torch

device = torch.cuda.get_device_name(0)

print(device)
print(torch.cuda.get_device_capability(0))