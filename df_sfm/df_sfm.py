"""Utilities for Structure from Motion using Gradient Descent"""

from __future__ import annotations
import os
from typing import List
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Scientific Computing
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math

# Machine Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F

# Kornia
import kornia
from kornia.geometry import resize
from kornia.core import Module, Tensor, pad, stack, tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters import filter2d, gaussian_blur2d
from kornia.geometry.transform.pyramid import _get_pyramid_gaussian_kernel

from kornia.geometry.transform.image_registrator import Similarity

from kornia.geometry.transform.homography_warper import HomographyWarper

from copy import deepcopy

class MyHomography(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.homography = nn.Parameter(torch.Tensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.eye_(self.homography)

    def forward(self) -> torch.Tensor:
        return torch.unsqueeze(self.homography, dim=0)  # 1x3x3
    
def get_gaussian_pyramid(img: torch.Tensor, num_levels: int) -> List[torch.Tensor]:
    r"""Utility function to compute a gaussian pyramid."""
    pyramid = []
    pyramid.append(img)
    for _ in range(num_levels - 1):
        img_curr = pyramid[-1]
        img_down = kornia.geometry.pyrdown(img_curr)
        pyramid.append(img_down)
    return pyramid

def compute_scale_loss(
                        img_src: torch.Tensor,
                        img_dst: torch.Tensor,
                        dst_homo_src: nn.Module,
                        optimizer: torch.optim,
                        num_iterations: int,
                        error_tol: float,
                        ) -> torch.Tensor:
    """Main optimization loop"""
    assert len(img_src.shape) == len(img_dst.shape), (img_src.shape, img_dst.shape)

    # init loop parameters
    loss_tol = torch.tensor(error_tol)
    loss_prev = torch.finfo(img_src.dtype).max

    for i in range(num_iterations):
        # create homography warper
        src_homo_dst: torch.Tensor = torch.inverse(dst_homo_src)

        _height, _width = img_src.shape[-2:]
        warper = kornia.geometry.HomographyWarper(_height, _width)
        img_src_to_dst = warper(img_src, src_homo_dst)

        # compute and mask loss
        loss = F.l1_loss(img_src_to_dst, img_dst, reduction="none")  # 1x3xHxW

        ones = warper(torch.ones_like(img_src), src_homo_dst)
        loss = loss.masked_select(ones > 0.9).mean()

        # compute gradient and update optimizer parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def get_gaussian_down(input: Tensor,
                      border_type: str = 'reflect',
                      align_corners: bool = False,
                      factor: float = 2.0,
                      blur: bool = True) -> Tensor:
    """
    Get gaussian pyramids for an image
    
    Args:
        img
        num_levels
        blur: whether or not to apply a gaussian blur

    Returns:
        Downsampled img
    """

    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    kernel: Tensor = _get_pyramid_gaussian_kernel()
    _, _, height, width = input.shape

    if blur:
        # blur image
        x_blur: Tensor = filter2d(input, kernel, border_type)
    else:
        x_blur = input

    # downsample.
    out: Tensor = kornia.geometry.transform.resize(
        x_blur,
        size=(int(float(height) / factor), int(float(width) // factor)),
        interpolation='bilinear',
        align_corners=align_corners,
    )
    return out

class MySimilarity(kornia.geometry.transform.image_registrator.ImageRegistrator):
    def register(
        self, src_img: Tensor, dst_img: Tensor, verbose: bool = False, output_intermediate_models: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        r"""Estimate the tranformation' which warps src_img into dst_img by gradient descent. The shape of the
        tensors is not checked, because it may depend on the model, e.g. volume registration.

        Args:
            src_img: Input image tensor.
            dst_img: Input image tensor.
            verbose: if True, outputs loss every 10 iterations.
            output_intermediate_models: if True with intermediate models

        Returns:
            the transformation between two images, shape depends on the model,
            typically [1x3x3] tensor for string model_types.
        """
        self.reset_model()
        # ToDo: better parameter passing to optimizer
        _opt_args: Dict[str, Any] = {}
        _opt_args['lr'] = self.lr
        opt = self.optimizer(self.model.parameters(), **_opt_args)

        # compute the gaussian pyramids
        # [::-1] because we have to register from coarse to fine
        img_src_pyr = get_gaussian_pyramid2(src_img, self.pyramid_levels,blur=True,factor=1.5)[::-1]
        img_dst_pyr = get_gaussian_pyramid2(dst_img, self.pyramid_levels,blur=False,factor=1.5)[::-1]
        prev_loss = 1e10
        aux_models = []
        losses = []
        if len(img_dst_pyr) != len(img_src_pyr):
            raise ValueError("Cannot register images of different sizes")
        for itr,(img_src_level, img_dst_level) in enumerate(zip(img_src_pyr, img_dst_pyr)):
            for i in range(self.num_iterations):
                # compute gradient and update optimizer parameters
                opt.zero_grad()
                loss = self.get_single_level_loss(img_src_level, img_dst_level, self.model())
                # loss += self.get_single_level_loss(img_dst_level, img_src_level, self.model.forward_inverse())
                current_loss = loss.item()
                if abs(current_loss - prev_loss) < self.tolerance:
                    print(f"Pyramid {itr}, Loss = {current_loss:.4f}, iter={i}")
                    break
                prev_loss = current_loss
                loss.backward()
                if verbose and (i % 25 == 0):
                # if verbose:
                    print(f"Pyramid {itr}, Loss = {current_loss:.4f}, iter={i}")
                opt.step()
            if output_intermediate_models:
                aux_models.append(self.model().clone().detach())
                losses.append(current_loss)
        print(f"Pyramid {itr}, Loss = {current_loss:.4f}, iter={i}")
        if output_intermediate_models:
            return self.model(), aux_models,losses
        return self.model()


if __name__ == "__main__":

    # Load images 
    # s2_mutli = 
    # l8_multi = 

    # Config
    learning_rate: float = 8e-4  # the gradient optimisation update step
    num_iterations: int = 500  # the number of iterations until convergence
    num_levels: int = 10  # the total number of image pyramid levels
    error_tol: float = 1e-8  # the optimisation error tolerance

    log_interval: int = 100  # print log every N iterations
    device = kornia.utils.get_cuda_or_mps_device_if_available()
    # device = 'cpu'
    print("Using ", device)

    s2_mutli, l8_multi = s2_mutli.to(device), l8_multi.to(device)

    # Setup homography, optimizer
    dst_homo_src = MyHomography().to(device)
    # sim_model = Similarity(True, True, True).to(device)
    # registrator = kornia.geometry.transform.image_registrator.ImageRegistrator("similarity", loss_fn=F.mse_loss, lr=8e-4, pyramid_levels=3, num_iterations=500).to(device)
    registrator = MySimilarity(warper = HomographyWarper, model_type = Similarity(True, False, True),
        loss_fn=F.mse_loss, lr=learning_rate, pyramid_levels=num_levels, num_iterations=num_iterations,tolerance=error_tol).to(device)
    optimizer = optim.Adam(dst_homo_src.parameters(), lr=learning_rate)

    # # Gaussian-pyramid the data
    s2_mutli_pyr: List[torch.Tensor] = get_gaussian_pyramid2(s2_mutli, num_levels,blur=True,factor=1.5)
    l8_multi_pyr: List[torch.Tensor] = get_gaussian_pyramid2(l8_multi, num_levels,blur=False,factor=1.5) # Don't blur Landsat data

    for iter_idx in range(num_levels):
    # get current pyramid data
    scale: int = (num_levels - 1) - iter_idx
    img_src = img_src_pyr[scale]
    img_dst = img_dst_pyr[scale]

    # compute scale loss
    compute_scale_loss(img_src, img_dst, dst_homo_src(), optimizer, num_iterations, error_tol)

    print(f"Optimization iteration: {iter_idx}/{num_levels}")

    # merge warped and target image for visualization
    h, w = img_src.shape[-2:]
    warper = kornia.geometry.HomographyWarper(h, w)
    img_src_to_dst = warper(img_src, torch.inverse(dst_homo_src()))
    img_src_to_dst_merge = 0.65 * img_src_to_dst + 0.35 * img_dst