import torch
def move2gpu(x):
    return torch.tensor(x,dtype=torch.complex128).to(device='cuda')
