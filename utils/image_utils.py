#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from torchvision import transforms


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def apply_colormap(img, colormap='jet'):
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img = img.squeeze()
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    img = np.squeeze(img)
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = cm.get_cmap(colormap)(img)
    img = transforms.ToTensor()(img)
    return img[:3,:,:][None]