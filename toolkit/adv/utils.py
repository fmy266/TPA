import torch
import torchvision
import numpy as np

identity = lambda x, *args: x

def nothing(**kwargs):
    pass

def clip_based_on_inf_norm_(distortion, epsilon, device):
    distortion.data = torch.sign(distortion.data) * torch.min(distortion.data.abs(),
                                                              torch.full_like(distortion, epsilon, device=device))


def clip_based_on_2_norm_(distortion, epsilon, device):
    batch_size = distortion.size(0)
    distortion_norm = distortion.data.view(batch_size, -1).norm(p=2, dim=1)
    factor = torch.min(epsilon / distortion_norm, torch.ones_like(distortion_norm)).view(-1, 1, 1, 1)
    distortion.data = distortion.data * factor

