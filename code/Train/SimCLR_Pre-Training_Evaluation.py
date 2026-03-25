"""
Script for pre-training a backbone with SimCLR on spectrogram datasets and downstream evaluation.
Uses the MINERVA framework: https://github.com/discovery-unicamp/Minerva

This script supports:
- Pre-training with SimCLR
- Linear readout evaluation
- Full fine-tuning evaluation
- Gini impurity calculation for cluster purity analysis
- UMAP visualization of learned representations
"""
import os
import re
import csv
import argparse
import random
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import copy

from tqdm import tqdm
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from typing import Optional, Dict, Any, Union, List, Tuple

# MINERVA imports[](https://github.com/discovery-unicamp/Minerva)
from minerva.data.readers.png_reader import PNGReader
from minerva.data.datasets.base import SimpleDataset
from minerva.data.data_modules.base import MinervaDataModule
from minerva.transforms.transform import _Transform, ContrastiveTransform
from minerva.models.nets.image.deeplabv3 import DeepLabV3
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.losses.xtent_loss import NTXentLoss
from minerva.optimizers.lars import LARS

# Torchvision transforms
from torchvision import transforms
from torchvision.transforms.v2 import (
    Compose, ToImage, ToDtype,
    RandomHorizontalFlip, RandomResizedCrop,
    RandomApply, ColorJitter, RandomGrayscale,
    GaussianBlur, Normalize, RandomVerticalFlip, GaussianNoise
)

from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torchmetrics.classification import JaccardIndex

# Visualization
import umap
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.core.module")
warnings.filterwarnings(
    "ignore",
    message="Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option.",
    category=UserWarning
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# =============================================================================
# Custom transforms
# =============================================================================

class Identity_2:
    """
    Transform that converts a NumPy array to a PyTorch tensor and applies ImageNet normalization.
    """
    
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """
        Apply the transformation pipeline to the input array.

        Parameters
        ----------
        x : np.ndarray
            Input image as a NumPy array. Expected to be in a format
            compatible with torchvision transforms.

        Returns
        -------
        torch.Tensor
            Transformed tensor with dtype float32 and ImageNet normalization
        """
        transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(x)

    def __str__(self) -> str:
        """
        Return a string description of the transform.

        Returns
        -------
        str
            Human-readable description of the transformation pipeline.
        """
        return "Identity_2 (numpy → tensor → ImageNet norm)"

    
class Format_label_img(_Transform):
    """
    Transform for segmentation labels.

    This transform converts a NumPy array of segmentation labels into a
    PyTorch tensor, adjusts label values if they start from 1, and resizes
    the tensor to a fixed target size using nearest-neighbor interpolation.
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the transform with a target size.

        Parameters
        ----------
        target_size : Tuple[int, int], optional
            Desired output size (height, width). Default is (256, 256).
        """
        self.target_size = target_size

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """
        Apply the transformation to the input label array.

        Steps:
        - Squeeze extra dimensions if present.
        - Convert NumPy array to a long tensor.
        - Subtract 1 if labels start from 1.
        - Ensure the tensor is 2D.
        - Resize to the target size using nearest-neighbor interpolation.

        Parameters
        ----------
        x : np.ndarray
            Input segmentation label array.

        Returns
        -------
        torch.Tensor
            Transformed label tensor of shape `target_size`, dtype long.

        Raises
        ------
        ValueError
            If the label tensor is not 2D after squeezing.
        """
        if x.ndim == 3:
            x = x.squeeze()

        label_tensor = torch.from_numpy(x).long()

        if label_tensor.min() >= 1:
            label_tensor -= 1

        if label_tensor.dim() != 2:
            label_tensor = label_tensor.squeeze()

        if label_tensor.dim() != 2:
            raise ValueError(f"Label is not 2D after squeeze: {label_tensor.shape}")

        if label_tensor.shape != self.target_size:
            label_tensor = F.interpolate(
                label_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=self.target_size,
                mode='nearest'
            ).squeeze().long()

        return label_tensor

class ImageFeatureDataset(Dataset):
    """
    Custom dataset for loading images and their labels from a CSV file.

    The dataset reads image file names and labels from a CSV file, loads
    the corresponding images from a directory, and applies optional
    transformations.
    """

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Initialize the dataset.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file containing image file names and labels.
        img_dir : str
            Directory where the image files are stored.
        transform : callable, optional
            Optional transform to be applied on the images. Default is None.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples.
        """
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Retrieve the image and label at the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor or PIL.Image.Image
            The loaded image, optionally transformed.
        label : int
            The label corresponding to the image.
        """
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.loc[idx, 'label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
# =============================================================================
# SimCLR Model
# =============================================================================

class SimCLR(L.LightningModule):
    """
    SimCLR model for self-supervised contrastive learning.
    
    Implements the SimCLR framework using NT-Xent loss for contrastive pre-training.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projection_head: nn.Module,
        flatten: bool = True,
        temperature: float = 0.5,
        lr: float = 1e-3,
    ):
        """
        Initialize the SimCLR model.
        
        Parameters
        ----------
        backbone : nn.Module
            Feature extractor backbone (e.g., ResNet50).
        projection_head : nn.Module
            Projection head (usually an MLP) to map features to lower-dimensional space.
        flatten : bool, optional, default=True
            Whether to flatten the backbone output before projection.
        temperature : float, optional, default=0.5
            Temperature parameter for NT-Xent loss scaling.
        lr : float, optional, default=1e-3
            Learning rate for the optimizer.
        """
        super().__init__()
        self.backbone = backbone
        self.projector = projection_head
        self.flatten = flatten
        self.temperature = temperature
        self.lr = lr
        self.loss_fn = NTXentLoss(temperature=temperature)

    def _unwrap_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recursively unwraps nested lists/tuples until the two augmented views are obtained.

        Parameters
        ----------
        batch : Any
            Input batch, potentially nested.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Two augmented views of the input image.

        Raises
        ------
        ValueError
            If batch does not contain exactly two views.
        """
        while isinstance(batch, (list, tuple)) and len(batch) == 1:
            batch = batch[0]

        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise ValueError(f"Expected two views, got: {batch}")

        return batch

    def forward(self, x: Any, flatten_override: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: extracts features from two views and projects them.

        Parameters
        ----------
        x : Any
            Input batch containing two augmented views.
        flatten_override : bool, optional
            Override self.flatten, by default None.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Projected features (z0, z1) for the two views.
        """
        x0, x1 = self._unwrap_batch(x)

        feat0 = self.backbone(x0)
        feat1 = self.backbone(x1)

        use_flatten = flatten_override if flatten_override is not None else self.flatten
        if use_flatten:
            feat0 = torch.flatten(feat0, start_dim=1)
            feat1 = torch.flatten(feat1, start_dim=1)

        z0 = self.projector(feat0)
        z1 = self.projector(feat1)

        return z0, z1

    def _single_step(self, batch: Any) -> torch.Tensor:
        """
        Computes contrastive loss for a single batch.

        Parameters
        ----------
        batch : Any
            Input batch containing two views.

        Returns
        -------
        Tensor
            NT-Xent loss value.
        """
        z0, z1 = self.forward(batch)
        loss = self.loss_fn(z0, z1)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step: computes loss and logs it.

        Parameters
        ----------
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
            Computed loss.
        """
        loss = self._single_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Validation step: computes loss and logs it.

        Parameters
        ----------
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
            Computed loss.
        """
        loss = self._single_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict step: returns projected features for both views.

        Parameters
        ----------
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : Optional[int], optional
            Dataloader index.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Projected features (z0, z1).
        """
        return self.forward(batch)

    def configure_optimizers(self) -> LARS:
        """
        Configures LARS optimizer for training.

        Returns
        -------
        LARS
            Optimizer instance.
        """
        return LARS(self.parameters(), lr=self.lr)

# =============================================================================
# SimCLR Transforms
# =============================================================================

class AdditiveGaussianNoise(_Transform):
    """
    Additive Gaussian noise augmentation.

    Adds Gaussian noise to the input image with a given probability.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.0, p: float = 0.5):
        """
        Initialize the additive Gaussian noise transform.
        
        Parameters
        ----------
        mean : float, default=0.0
            Mean of the Gaussian noise (0.0 is neutral).
        std : float, default=0.0
            Standard deviation of the noise (in pixel intensity scale, typically 0–255).
        p : float, default=0.5
            Probability of applying the noise transformation.
        """
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply additive Gaussian noise to the input image.

        Parameters
        ----------
        x : np.ndarray
            Input image array (HWC or HW, typically uint8).

        Returns
        -------
        np.ndarray
            Image with noise applied (or original if not applied).
        """
        if np.random.rand() < self.p:
            noise = np.random.normal(self.mean, self.std, size=x.shape).astype(np.float32)
            noisy = x.astype(np.float32) + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)
        return x

class HorizontalLineMask(_Transform):
    """
    Horizontal line mask augmentation.

    Randomly draws horizontal lines of fixed width to occlude parts of the image.
    """

    def __init__(self, min_lines: int = 3, max_lines: int = 5, line_width: int = 16, seed: int = None):
        """
        Initialize the horizontal line mask transform.
        
        Parameters
        ----------
        min_lines : int, default=3
            Minimum number of lines to draw.
        max_lines : int, default=5
            Maximum number of lines to draw.
        line_width : int, default=16
            Height of each horizontal line in pixels.
        seed : int, optional
            Random seed for reproducibility.
        """
        if line_width <= 0:
            raise ValueError("line_width must be positive")
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.line_width = line_width
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        """Apply horizontal line mask to input.

        Parameters
        ----------
        x : PIL.Image.Image or torch.Tensor or np.ndarray
            Input image (PIL, tensor CHW, or numpy HWC/HW).

        Returns
        -------
        np.ndarray or torch.Tensor
            Masked image in the original format.
        """
        if isinstance(x, Image.Image):
            x = np.array(x)
        elif isinstance(x, torch.Tensor):
            x = x.permute(1, 2, 0).cpu().numpy()  # CHW → HWC

        if not isinstance(x, np.ndarray):
            raise TypeError(f"Unsupported input type: {type(x)}")

        # Determine format
        if x.ndim == 2:
            h, w = x.shape
            c = 1
            format_ = 'HW'
        elif x.ndim == 3:
            if x.shape[2] in [1, 3, 4]:
                h, w, c = x.shape
                format_ = 'HWC'
            else:
                c, h, w = x.shape
                format_ = 'CHW'
                x = np.transpose(x, (1, 2, 0))  # CHW → HWC
        else:
            raise ValueError(f"Unsupported shape: {x.shape}")

        max_start = h - self.line_width
        if max_start <= 0:
            raise ValueError(f"Height {h} too small for line width {self.line_width}")

        max_possible = max_start // self.line_width
        num_lines = min(self.rng.integers(self.min_lines, self.max_lines + 1), max_possible)

        starts = np.arange(0, max_start + 1, self.line_width)
        positions = self.rng.choice(starts, size=num_lines, replace=False)

        x_out = x.copy()
        fill = 0 if x.dtype == np.uint8 else 0.0

        for start in positions:
            end = start + self.line_width
            if format_ in ['HW', 'HWC']:
                x_out[start:end, :] = fill
            else:
                x_out[start:end, :, :] = fill

        if format_ == 'CHW':
            x_out = np.transpose(x_out, (2, 0, 1))

        return x_out


class VerticalLineMask(_Transform):
    """
    Vertical line mask augmentation.

    Randomly draws vertical lines of fixed width to occlude parts of the image.
    """

    def __init__(self, min_lines: int = 3, max_lines: int = 5, line_width: int = 16, seed: int = None):
        """
        Initialize the vertical line mask transform.
        
        Parameters
        ----------
        min_lines : int, default=3
            Minimum number of lines to draw.
        max_lines : int, default=5
            Maximum number of lines to draw.
        line_width : int, default=16
            Width of each vertical line in pixels.
        seed : int, optional
            Random seed for reproducibility.
        """
        if line_width <= 0:
            raise ValueError("line_width must be positive")
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.line_width = line_width
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        """Apply vertical line mask to input.

        Parameters
        ----------
        x : PIL.Image.Image or torch.Tensor or np.ndarray
            Input image (PIL, tensor CHW, or numpy HWC/HW).

        Returns
        -------
        np.ndarray or torch.Tensor
            Masked image in the original format.
        """
        if isinstance(x, Image.Image):
            x = np.array(x)
        elif isinstance(x, torch.Tensor):
            x = x.permute(1, 2, 0).cpu().numpy()  # CHW → HWC

        if not isinstance(x, np.ndarray):
            raise TypeError(f"Unsupported input type: {type(x)}")

        # Determine format
        if x.ndim == 2:
            h, w = x.shape
            c = 1
            format_ = 'HW'
        elif x.ndim == 3:
            if x.shape[2] in [1, 3, 4]:
                h, w, c = x.shape
                format_ = 'HWC'
            else:
                c, h, w = x.shape
                format_ = 'CHW'
                x = np.transpose(x, (1, 2, 0))  # CHW → HWC

        else:
            raise ValueError(f"Unsupported shape: {x.shape}")

        max_start = w - self.line_width
        if max_start <= 0:
            raise ValueError(f"Width {w} too small for line width {self.line_width}")

        max_possible = max_start // self.line_width
        num_lines = min(self.rng.integers(self.min_lines, self.max_lines + 1), max_possible)

        starts = np.arange(0, max_start + 1, self.line_width)
        positions = self.rng.choice(starts, size=num_lines, replace=False)

        x_out = x.copy()
        fill = 0 if x.dtype == np.uint8 else 0.0

        for start in positions:
            end = start + self.line_width
            if format_ in ['HW', 'HWC']:
                x_out[:, start:end] = fill
            else:
                x_out[:, :, start:end] = fill

        if format_ == 'CHW':
            x_out = np.transpose(x_out, (2, 0, 1))

        return x_out
    
def get_contrastive_transform_original(variant: str = "base"):
    """
    Returns a contrastive augmentation pipeline for SimCLR pre-training.

    The pipeline applies strong random augmentations to generate two views of the same image,
    following the standard SimCLR augmentation strategy.

    Returns
    -------
    Callable
        ContrastiveTransform object that returns two augmented views.
    """
    base_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        RandomResizedCrop(size=IMG_SIZE, scale=(0.2, 1.0)),
        RandomHorizontalFlip(p=0.5),
        RandomApply([ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        Normalize(mean=MEAN, std=STD),
    ])
    return ContrastiveTransform(base_transform)

def get_contrastive_transform(variant: str = "base"):
    """
    Returns a contrastive augmentation pipeline for SimCLR pre-training.

    Variants allow ablation studies (e.g., with/without crop, jitter, masks, etc.).

    Parameters
    ----------
    variant : str
        Name of the augmentation variant. See list below.

    Returns
    -------
    ContrastiveTransform
        Transform that generates two augmented views.
    """
    # Common components (used in most variants)
    crop = RandomResizedCrop(size=IMG_SIZE, scale=(0.2, 1.0))
    flip = RandomHorizontalFlip(p=0.5)
    jitter_strong = RandomApply([ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8)
    grayscale = RandomGrayscale(p=0.2)
    blur_strong = GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    noise = AdditiveGaussianNoise(std=0.05, p=0.5)
    h_mask = HorizontalLineMask(min_lines=3, max_lines=5, line_width=16)
    v_mask = VerticalLineMask(min_lines=3, max_lines=5, line_width=16)
    normalize = Normalize(mean=MEAN, std=STD)

    # Base components that go FIRST and LAST
    base_start = [
        ToImage(),
        ToDtype(torch.float32, scale=True),
    ]

    base_end = [normalize]

    if variant == "base":  # Full classic SimCLR
        base_transform = Compose([
            *base_start,
            crop,
            flip,
            jitter_strong,
            grayscale,
            blur_strong,
            *base_end
        ])

    elif variant == "no_crop":
        base_transform = Compose([
            *base_start,
            flip,
            jitter_strong,
            grayscale,
            blur_strong,
            *base_end
        ])

    elif variant == "crop_only":
        base_transform = Compose([
            *base_start,
            crop,
            flip,
            *base_end
        ])

    elif variant == "no_jitter":
        base_transform = Compose([
            *base_start,
            crop,
            flip,
            grayscale,
            blur_strong,
            *base_end
        ])

    elif variant == "jitter_only":
        base_transform = Compose([
            *base_start,
            crop,
            flip,
            jitter_strong,
            *base_end
        ])

    elif variant == "no_blur":
        base_transform = Compose([
            *base_start,
            crop,
            flip,
            jitter_strong,
            grayscale,
            *base_end
        ])

    elif variant == "blur_only":
        base_transform = Compose([
            *base_start,
            crop,
            flip,
            blur_strong,
            *base_end
        ])

    elif variant == "masks_only":
        base_transform = Compose([
            h_mask,
            v_mask,
            *base_start,
            *base_end
        ])
        
    elif variant == "masks_weak":
        base_transform = Compose([
            h_mask,
            v_mask,
            *base_start,
            crop,
            flip,
            *base_end
        ])
        
    elif variant == "horizontal_weak":
        base_transform = Compose([
            h_mask,
            *base_start,
            crop,
            flip,
            *base_end
        ])

    elif variant == "vertical_weak":
        base_transform = Compose([
            v_mask,
            *base_start,
            crop,
            flip,
            *base_end
        ])
        
    elif variant == "masks_strong":
        base_transform = Compose([
            h_mask,
            v_mask,
            *base_start,
            crop,
            flip,
            jitter_strong,
            *base_end
        ])
        
    elif variant == "masks_noise":
        base_transform = Compose([
            h_mask,
            v_mask,
            noise,
            *base_start,
            crop,
            flip,
            *base_end
        ])
        
    elif variant == "noise_only":
        base_transform = Compose([
            noise,
            *base_start,
            crop,
            flip,
            *base_end
        ])

    elif variant == "noise_strong":
        base_transform = Compose([
            noise,
            *base_start,
            crop,
            flip,
            jitter_strong,
            blur_strong,
            *base_end
        ])

    elif variant == "no_grayscale":
        base_transform = Compose([
            *base_start,
            crop,
            flip,
            jitter_strong,
            blur_strong,
            *base_end
        ])

    elif variant == "full":
        base_transform = Compose([
            noise,       
            h_mask, 
            v_mask,   
            *base_start,
            crop,
            flip,
            jitter_strong,
            grayscale,
            blur_strong,
            *base_end
        ])

    elif variant == "minimal":
        base_transform = Compose([
            *base_start,
            crop,
            flip,
            *base_end
        ])

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return ContrastiveTransform(base_transform)

# =============================================================================
# Prediction Head
# =============================================================================

class SimpleSegmentationHead(nn.Sequential):
    """
    Simple segmentation head with 3x3 convolution + ReLU + 1x1 convolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels from the backbone.
    num_classes : int
        Number of segmentation classes.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
# =============================================================================
# Linear Head
# =============================================================================
class ProjectionHead(nn.Sequential):
    """
    Projection head module for feature representation.

    Architecture
    ------------
    - AdaptiveAvgPool2d((1, 1)): Global average pooling to reduce spatial dimensions.
    - Flatten(start_dim=1): Flatten pooled features into a vector.
    - MLP: Fully connected layers with ReLU activation and batch normalization.

    Output dimension: 256
    """
    
    def __init__(self) -> None:
            """
        Initialize the projection head.

        The MLP consists of:
        - Input size: 768
        - Hidden size: 4096 (with BatchNorm1d and ReLU)
        - Output size: 256
        """
        
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            MLP(
                layer_sizes=[768, 4096, 256],
                activation_cls=nn.ReLU,
                intermediate_ops=[nn.BatchNorm1d(4096), None],
            ),
        )
        
class LinearCustomHead(nn.Module):
    """
    Simple linear head for segmentation readout.
    """

    def __init__(self, in_channels: int, num_classes: int, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the linear head.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels from the backbone.
        num_classes : int
            Number of segmentation classes.
        target_size : Tuple[int, int], optional
            Target spatial size for upsampling, by default (256, 256)
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear head.
        """
        
        logits = self.conv(x)
        logits = F.interpolate(
            logits,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )
        return logits

# =============================================================================
# mIoU evaluation function
# =============================================================================
def model_mIoU(model, trainer, datamodule):
    """
    Computes mean Intersection over Union (mIoU) on the test set using trainer.predict.

    Parameters
    ----------
    model : L.LightningModule
        Trained model with predict_step implemented.
    trainer : Trainer
        PyTorch Lightning Trainer instance.
    datamodule : MinervaDataModule
        DataModule containing test_dataset.

    Returns
    -------
    float
        mIoU score in percentage.
    """
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        datamodule.test_dataset,
        batch_size=BATCH_SIZE_FINE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    predictions = trainer.predict(model, test_loader)
    predictions = torch.cat(predictions, dim=0).cpu()

    gt = []
    for idx, (_, y) in enumerate(datamodule.test_dataset):
        y = y.long().cpu()

        if y.dim() == 3:
            if y.shape[0] == 1:
                y = y.squeeze(0)
        elif y.dim() == 4:
            if y.shape[1] == 1:
                y = y.squeeze(1)
        if y.dim() != 2:
            print(f"Mask error {idx}: {y.shape}")
            continue
        y = y.unsqueeze(0)  # [1, H, W]
        gt.append(y)
    gt = torch.cat(gt, dim=0)  # [N, H, W]

    jaccard = JaccardIndex(task="multiclass", num_classes=3, average="macro")
    score = jaccard(predictions, gt)
    miou = score.item() * 100

    print(f"The mIoU of the model is {miou:.2f}%")
    return miou

# =============================================================================
# Global Configuration
# =============================================================================

IMG_SIZE = 192 # 192 for SimCLR
BATCH_SIZE = 256
NUM_WORKERS = 4
MAX_EPOCHS = 50
TEMPERATURE = 0.5
LR = 1e-4

BATCH_SIZE_FINE = 32
MAX_EPOCHS_FINE = 50

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

# Root directories
EXPERIMENTS_ROOT = Path("/data/spectograms")
DATASETS_ROOT = Path("/data/Dataset")
SUPERVISED_DIR = DATASETS_ROOT / "dataset_spectogram"  # train/val/test for downstream
FRAGMENTED_DIR = DATASETS_ROOT / "fragmented_data"     # iteration_XX
FRAGMENTED2_DIR = DATASETS_ROOT / "fragmented_data_2"  # XX_percent

MODEL_PREFIX = "SIMCLR"

# Available datasets
DATASETS = {
    "spectogram_rgb": "spectogram_rgb",
    "dataset_no_panoradio_rgb": "dataset_no_panoradio_rgb",
    "dataset_no_powder_rgb": "dataset_no_powder_rgb",
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
L.seed_everything(SEED, workers=True)

# =============================================================================
# Utility Functions: Directory Management
# =============================================================================

def get_pretext_checkpoints_dir(dataset_name: str) -> Path:
    """
    Returns the directory for pre-training checkpoints.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    Path
        Path to the checkpoints directory.
    """
    return EXPERIMENTS_ROOT / MODEL_PREFIX / dataset_name / "pretext" / "checkpoints"


def get_pretext_csv_dir(dataset_name: str) -> Path:
    """
    Returns the directory for pre-training CSV logs.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    Path
        Path to the CSV directory.
    """
    return EXPERIMENTS_ROOT / MODEL_PREFIX / dataset_name / "pretext" / "csv"


def get_downstream_checkpoints_dir(dataset_name: str) -> Path:
    """
    Returns the directory for downstream checkpoints.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    Path
        Path to the checkpoints directory.
    """
    return EXPERIMENTS_ROOT / MODEL_PREFIX / dataset_name / "downstream" / "checkpoints"


def get_downstream_csv_dir(dataset_name: str) -> Path:
    """
    Returns the directory for downstream CSV results.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    Path
        Path to the CSV directory.
    """
    return EXPERIMENTS_ROOT / MODEL_PREFIX / dataset_name / "downstream" / "csv"


def ensure_dirs_for_dataset(dataset_name: str) -> None:
    """
    Creates (if necessary) all required directories for a given dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Raises
    ------
    ValueError
        If dataset_name is not recognized.
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    folders = [
        get_pretext_checkpoints_dir(dataset_name),
        get_pretext_csv_dir(dataset_name),
        get_downstream_checkpoints_dir(dataset_name),
        get_downstream_csv_dir(dataset_name),
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Directories checked/created for dataset '{dataset_name}':")
    for f in folders:
        print(f"  - {f}")


def get_dataset_path(dataset_name: str) -> Path:
    """
    Returns the full path to the raw dataset folder.

    Parameters
    ----------
    dataset_name : str
        Key from DATASETS dictionary.

    Returns
    -------
    Path
        Path to the dataset folder.
    """
    return DATASETS_ROOT / DATASETS[dataset_name]

# =============================================================================
# SimCLR Pre-training and Evaluation Functions
# ============================================================================

def load_datamodule(dataset_name: str, transform_variant: str = "base") -> MinervaDataModule:
    """
    Loads a DataModule for SimCLR pre-training on a specific dataset.

    Parameters
    ----------
    dataset_name : str
        Key from the DATASETS dictionary (e.g., "spectogram_rgb").
    transform_variant : str, optional
        Variant of contrastive transform to use, by default "base".

    Returns
    -------
    MinervaDataModule
        Configured DataModule ready for training.

    Raises
    ------
    ValueError
        If dataset_name is unknown.
    FileNotFoundError
        If the dataset path does not exist.
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    # Full path to raw dataset folder
    data_path = get_dataset_path(dataset_name)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_path}")

    # Image reader
    reader = PNGReader(path=data_path)

    # Contrastive transform (two views)
    contrastive_transform = get_contrastive_transform(transform_variant)

    # Dataset
    train_dataset = SimpleDataset(
        readers=[reader],
        transforms=[contrastive_transform],
        return_single=False  # Important: returns (view1, view2)
    )

    # DataModule
    dm = MinervaDataModule(
        name=f"SimCLR_{dataset_name}",
        train_dataset=train_dataset,
        val_dataset=None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle_train=True
    )

    print(f"DataModule loaded successfully for dataset: {dataset_name}")
    print(f"Full dataset path: {data_path}")
    return dm


def pretrain_backbone(dataset_name: str = "spectogram_rgb", transform_variant: str = "base", backbone_name: str = "resnet50") -> None:
    """
    Runs SimCLR pre-training for a given dataset.

    Parameters
    ----------
    dataset_name : str, default="spectogram_rgb"
        Dataset key to use for pre-training.
    transform_variant : str, optional
        Variant of contrastive transform to use, by default "base".
    backbone_name : str, default="resnet50"
        Name of the backbone network to use.
    """
    print(f"Starting SimCLR pre-training for dataset: {dataset_name}")

    # 1. Ensure output directories exist
    ensure_dirs_for_dataset(dataset_name)

    # 2. Load DataModule
    dm = load_datamodule(dataset_name, transform_variant)

    # 3. Backbone + Projection Head
    if backbone_name == "resnet50":
        backbone = resnet50(weights=None)
        feat_dim = 2048
    elif backbone_name == "resnet18":
        backbone = resnet18(weights=None)
        feat_dim = 512
    else:
        raise ValueError(f"Backbone no soportado: {backbone_name}. Opciones: resnet50, resnet18, resnet34")
    
    backbone.fc = nn.Identity()

    proj_head = nn.Sequential(
        nn.Linear(feat_dim, feat_dim),
        nn.ReLU(inplace=True),
        nn.Linear(feat_dim, 128)
    )

    # 4. Model
    model = SimCLR(
        backbone=backbone,
        projection_head=proj_head,
        temperature=TEMPERATURE,
        lr=LR,
        flatten=True
    )

    # 5. Specific paths for this dataset
    pretext_ckpt_dir = get_pretext_checkpoints_dir(dataset_name)
    pretext_csv_dir = get_pretext_csv_dir(dataset_name)

    # 6. Loggers (TensorBoard + CSV)
    tb_logger = TensorBoardLogger(
        save_dir=str(pretext_ckpt_dir.parent),  # .../pretext
        name="checkpoints"                      # creates checkpoints subfolder
    )
    csv_logger = CSVLogger(
        save_dir=str(pretext_csv_dir.parent),   # .../pretext
        name="csv"                              # creates csv subfolder
    )

    print(f"🔥 Transform: {transform_variant}")
    print(f"📁 CSV version: {csv_logger.version}")
    print(f"📂 CSV path: {csv_logger.log_dir}")

    # 7. Checkpoint callbacks
    callbacks_list = [
        ModelCheckpoint(
            save_weights_only=True,
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-epoch"
        ),
        ModelCheckpoint(
            filename=f"simclr-{{epoch:03d}}-{{train_loss:.4f}}",
            save_weights_only=True,
            monitor="train_loss",
            mode="min",
            save_top_k=2,
            every_n_epochs=10,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # 8. Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=callbacks_list,
        logger=[tb_logger, csv_logger],
        default_root_dir=str(pretext_ckpt_dir),
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    # 9. Start training
    trainer.fit(model, datamodule=dm)

    print(f"Pre-training completed for {dataset_name}")
    print(f"Checkpoints saved to: {pretext_ckpt_dir}")
    print(f"CSV logs saved to: {pretext_csv_dir}")

    return model

# =============================================================================
# Backbone Loading
# =============================================================================

def get_last_checkpoint(dataset_name: str, version: Optional[int] = None) -> Optional[str]:
    """
    Searches for the latest checkpoint in the pretext/checkpoints folder of the dataset.
    Priority: most recent best-*.ckpt > last.ckpt.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    version : int, optional
        Specific version number to use (e.g. 5 for version_5).
        If None, uses the latest version.

    Returns
    -------
    Optional[str]
        Path to the latest checkpoint, or None if none found.
    """
    ckpt_base = get_pretext_checkpoints_dir(dataset_name)

    if not ckpt_base.exists():
        print(f"No checkpoints folder found: {ckpt_base}")
        return None

    # Find Lightning version folders (version_X)
    version_dirs = [d for d in os.listdir(ckpt_base) if re.match(r"version_\d+", d)]
    if not version_dirs:
        print("No version_X folders found in checkpoints")
        return None

    if version is not None:
        target = f"version_{version}"
        if target not in version_dirs:
            print(f"Version {target} not found. Available: {sorted(version_dirs)}")
            return None
        last_version = target
    else:
        # Sort by version number (highest first)
        version_dirs.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
        last_version = version_dirs[0]

    ckpt_dir = ckpt_base / last_version / "checkpoints"

    if not ckpt_dir.exists():
        print(f"No checkpoints folder in version {last_version}")
        return None

    # Priority 1: most recent best-*.ckpt (by modification time)
    best_ckpts = [f for f in ckpt_dir.iterdir() if f.name.startswith("best") and f.suffix == ".ckpt"]
    if best_ckpts:
        latest_best = max(best_ckpts, key=lambda p: p.stat().st_mtime)
        print(f"Latest best checkpoint found: {latest_best}")
        return str(latest_best)

    # Priority 2: last.ckpt
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        print(f"Using last.ckpt: {last_ckpt}")
        return str(last_ckpt)

    print("No valid checkpoint found")
    return None


def load_backbone(
    dataset_name: str = "spectogram_rgb",
    from_scratch: bool = False,
    pretrained: str = "none",
    version: int = None,
    backbone_name: str = "resnet50"
) -> nn.Module:
    """
    Loads a ResNet50 backbone with different initialization strategies.

    Parameters
    ----------
    dataset_name : str, default="spectogram_rgb"
        Name of the dataset (used for SimCLR checkpoint loading).
    from_scratch : bool, default=False
        If True, loads random weights backbone (baseline).
    pretrained : str, default="none"
        Pretraining type: "none" (SimCLR checkpoint), "imagenet", "coco".
    version : int, optional
        Specific version number to load for SimCLR checkpoint. If None, loads latest.
    backbone_name : str, default="resnet50"
        Backbone architecture to use ("resnet50" or "resnet18").

    Returns
    -------
    nn.Module
        Backbone model (up to layer4).

    Raises
    ------
    FileNotFoundError
        If SimCLR checkpoint is not found.
    """
    dilation_cfg = [False, False, True]  # Output stride 16 (like BYOL)

    if pretrained == "imagenet":
        print("Loading ResNet50 pretrained on ImageNet with layer4 dilations")
        rn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, replace_stride_with_dilation=dilation_cfg)

    elif pretrained == "coco":
        print("Loading DeepLabV3-ResNet50 backbone pretrained on COCO")
        model = deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1")
        return model.backbone

    elif from_scratch:
        print("Loading ResNet50 from scratch (no dilations, standard backbone)")
        print(f"Backbone: {backbone_name}")
        if backbone_name == "resnet18":
            rn = resnet18(weights=None)
        elif backbone_name == "resnet50":
            print("Backbone: ResNet50")
            rn = resnet50(weights=None, replace_stride_with_dilation=[False, False, False])

        backbone = nn.Sequential(*list(rn.children())[:-2])
        return backbone

    else:  # SimCLR checkpoint
        print("Loading SimCLR checkpoint")
        if version is None:
            ckpt_path = get_last_checkpoint(dataset_name)
        else:
            print(f"Looking for version_{version} checkpoint...")
            ckpt_path = get_last_checkpoint(dataset_name, version=version)
            
        if not ckpt_path:   
            raise FileNotFoundError(f"No checkpoint found for dataset: {dataset_name}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Instanciar arquitectura correcta según backbone_name
        if backbone_name == "resnet18":
            rn = resnet18(weights=None)
        elif backbone_name == "resnet50":
            rn = resnet50(weights=None, replace_stride_with_dilation=dilation_cfg)
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    # Filter keys for backbone (up to layer4)
    filtered = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
    missing, unexpected = rn.load_state_dict(filtered, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # Standard cut: remove avgpool + fc
    backbone = nn.Sequential(*list(rn.children())[:-2])

    return backbone


# =============================================================================
# Downstream Evaluation: Linear Readout
# =============================================================================

def linear_readout_evaluation(
    dataset_name: str,
    experiment_name: str,
    pretrain: str = "none",
    from_scratch: bool = False,
) -> None:
    """
    Performs linear readout evaluation: frozen backbone + trainable linear head.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    experiment_name : str
        Experiment identifier for output files.
    pretrain : str, optional
        Pretraining type ("none" for SimCLR checkpoint, "imagenet", "coco"), default "none".
    from_scratch : bool, optional
        Use random weights backbone if True, default False.
    """
    backbone = load_backbone(dataset_name, from_scratch=from_scratch, pretrained=pretrain)

    csv_suffix = "baseline_scratch_linear" if from_scratch else f"{pretrain}_linear"
    csv_path = get_downstream_csv_dir(dataset_name) / f"{csv_suffix}_{experiment_name}.csv"
    print(f"Saving linear readout results to: {csv_path}")

    # Fixed val and test sets
    val_dataset = SimpleDataset(
        readers=[
            PNGReader(SUPERVISED_DIR / "val" / "data"),
            PNGReader(SUPERVISED_DIR / "val" / "label")
        ],
        transforms=[Identity_2(), Format_label_img()],
    )

    test_dataset = SimpleDataset(
        readers=[
            PNGReader(SUPERVISED_DIR / "test" / "data"),
            PNGReader(SUPERVISED_DIR / "test" / "label")
        ],
        transforms=[Identity_2(), Format_label_img()],
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["percent", "mIoU_list", "mIoU_mean"])

    percentages = [0.2, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
    low_percent_map = {0.2: "01", 1.0: "06", 5.0: "31", 10.0: "61"}
    high_percent_map = {25.0: "25_percent", 50.0: "50_percent", 75.0: "75_percent"}

    for percent in percentages:
        miou_list = []

        print(f"\nLinear readout {percent}% ...")
        # bb_copy = copy.deepcopy(backbone)

        # model = DeepLabV3(
        #     backbone=bb_copy,
        #     learning_rate=1e-2,
        #     num_classes=3
        # )
        # Freeze backbone parameters
        # def freeze_module(module):
        #     for param in module.parameters():
        #         param.requires_grad = False
        #     for child in module.children():
        #         freeze_module(child)

        # freeze_module(model.backbone)
        
        # Copy backbone
        bb_copy = copy.deepcopy(backbone)
        
        model = SimpleSupervisedModel(
            backbone=bb_copy,
            fc=LinearCustomHead(2048, 3),
            loss_fn=nn.CrossEntropyLoss(),
            learning_rate=1e-2,
            flatten=False,    
            freeze_backbone=True,
            train_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=3)},
            val_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=3)},
            test_metrics={"mIoU": JaccardIndex(task="multiclass", num_classes=3)},
        )

        # Verify only head parameters are trainable
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params (head only): {trainable}")

        # Load train data
        if percent == 100.0:
            data_path = SUPERVISED_DIR / "train" / "data"
            label_path = SUPERVISED_DIR / "train" / "label"
        elif percent in low_percent_map:
            code = low_percent_map[percent]
            data_path = DATASETS_ROOT / "fragmented_data" / f"iteration_{code}" / "data"
            label_path = DATASETS_ROOT / "fragmented_data" / f"iteration_{code}" / "label"
        elif percent in high_percent_map:
            folder = high_percent_map[percent]
            data_path = DATASETS_ROOT / "fragmented_data_2" / folder / "data"
            label_path = DATASETS_ROOT / "fragmented_data_2" / folder / "label"
        else:
            continue

        if not data_path.exists():
            print(f"Path not found: {data_path} → skipping")
            continue

        train_dataset = SimpleDataset(
            readers=[PNGReader(data_path), PNGReader(label_path)],
            transforms=[Identity_2(), Format_label_img()],
        )

        dm = MinervaDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=BATCH_SIZE_FINE,
            num_workers=NUM_WORKERS,
        )

        trainer = Trainer(
            max_epochs=50,
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(model, dm)

        miou = model_mIoU(model, trainer, dm)
        miou_list.append(miou)

        mean_miou = sum(miou_list) / len(miou_list) if miou_list else 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([percent, miou_list, mean_miou])
        print(f"Linear readout {percent}% → mIoU en TEST: {miou:.2f}%")

    print(f"Finalizado linear readout. Resultados: {csv_path}")

def full_finetune_evaluation(
    dataset_name: str,
    experiment_name: str,
    pretrain: str = "none",
    from_scratch: bool = False,
    version: int = None,
    backbone_name: str = "resnet50"
) -> None:
    """
    Performs full fine-tuning evaluation on downstream segmentation tasks.

    Trains the full model (backbone + segmentation head) on varying amounts of supervised data
    and evaluates mIoU on a fixed test set.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    experiment_name : str
        Experiment identifier for output files.
    pretrain : str, optional
        Pretraining type ("none" for SimCLR checkpoint, "imagenet", "coco"), default "none".
    from_scratch : bool, optional
        Use random weights backbone if True, default False.
    version : int, optional
        Specific SimCLR checkpoint version to use (e.g. 5 for version_5).
    backbone : str, optional
        Backbone architecture to use ("resnet50" or "resnet18"), default "resnet50".
    """
    # Load initial backbone
    print(f"Loading backbone for full fine-tuning evaluation: dataset={dataset_name}, pretrain={pretrain}, from_scratch={from_scratch}, version={version}")
    backbone = load_backbone(dataset_name, from_scratch=from_scratch, pretrained=pretrain, version=version, backbone_name=backbone_name)

    if from_scratch:
        csv_suffix = "baseline_scratch"
    elif pretrain == "imagenet":
        csv_suffix = "imagenet_fullft"
    elif pretrain == "coco":
        csv_suffix = "coco_fullft"
    else:
        csv_suffix = "simclr_fullft"

    csv_path = get_downstream_csv_dir(dataset_name) / f"{csv_suffix}_{experiment_name}.csv"
    print(f"Saving results to: {csv_path}")

    # Load fixed validation and test sets
    val_data_reader = PNGReader(path=SUPERVISED_DIR / "val" / "data")
    val_label_reader = PNGReader(path=SUPERVISED_DIR / "val" / "label")
    val_dataset = SimpleDataset(
        readers=[val_data_reader, val_label_reader],
        transforms=[Identity_2(), Format_label_img()],
    )

    test_data_reader = PNGReader(path=SUPERVISED_DIR / "test" / "data")
    test_label_reader = PNGReader(path=SUPERVISED_DIR / "test" / "label")
    test_dataset = SimpleDataset(
        readers=[test_data_reader, test_label_reader],
        transforms=[Identity_2(), Format_label_img()],
    )

    # Prepare CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["percent", "mIoU_list", "mIoU_mean"])

    percentages = [0.2, 1.0, 10.0, 50.0, 100.0]
    low_percent_map = {0.2: "01", 1.0: "06", 5.0: "31", 10.0: "61"}
    high_percent_map = {25.0: "25_percent", 50.0: "50_percent", 75.0: "75_percent"}

    for percent in percentages:
        miou_list = []
        
        # bb_copy = copy.deepcopy(backbone)
        # model = DeepLabV3(
        #     backbone=bb_copy,
        #     pred_head=SimpleSegmentationHead(2048, 3),
        #     learning_rate=1e-4,
        #     num_classes=3
        # )
        # # Freeze backbone parameters
        # def freeze_module(module):
        #     for param in module.parameters():
        #         param.requires_grad = False
        #     for child in module.children():
        #         freeze_module(child)

        # freeze_module(model.backbone)
        
        # Copy backbone and create new model
        bb_copy = copy.deepcopy(backbone)
        if backbone_name == "resnet18":
            print("Using ResNet18 backbone → head input channels = 512")
            model = DeepLabV3(
                backbone=bb_copy,
                pred_head=SimpleSegmentationHead(512, 3),
                learning_rate=1e-4,
                num_classes=3
            )
        else:
            model = DeepLabV3(
                backbone=bb_copy,
                pred_head=SimpleSegmentationHead(2048, 3),
                learning_rate=1e-4,
                num_classes=3
            )

        # Select train data based on percentage
        if percent == 100.0:
            data_path = SUPERVISED_DIR / "train" / "data"
            label_path = SUPERVISED_DIR / "train" / "label"
        elif percent in low_percent_map:
            code = low_percent_map[percent]
            data_path = DATASETS_ROOT / "fragmented_data" / f"iteration_{code}" / "data"
            label_path = DATASETS_ROOT / "fragmented_data" / f"iteration_{code}" / "label"
        elif percent in high_percent_map:
            folder = high_percent_map[percent]
            data_path = DATASETS_ROOT / "fragmented_data_2" / folder / "data"
            label_path = DATASETS_ROOT / "fragmented_data_2" / folder / "label"
        else:
            continue

        if not data_path.exists():
            continue

        train_reader_data = PNGReader(path=data_path)
        train_reader_label = PNGReader(path=label_path)

        train_dataset = SimpleDataset(
            readers=[train_reader_data, train_reader_label],
            transforms=[Identity_2(), Format_label_img()],
        )

        # DataModule: train changes, val and test are fixed
        dm = MinervaDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,          # fixed
            test_dataset=test_dataset,        # fixed - key for real mIoU
            batch_size=BATCH_SIZE_FINE,
            num_workers=NUM_WORKERS,
            name=f"fullft_{percent}",
        )

        trainer = Trainer(
            max_epochs=MAX_EPOCHS_FINE,
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(model, dm)

        # mIoU on TEST (as in BYOL)
        miou = model_mIoU(model, trainer, dm)
        miou_list.append(miou)

        mean_miou = sum(miou_list) / len(miou_list) if miou_list else 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([percent, miou_list, mean_miou])
        print(f"{percent}% → mIoU on TEST: {miou:.2f}%")

    print(f"Finished. Results saved to: {csv_path}")
    
      
# =============================================================================
# Visualization Functions
# =============================================================================
def calculate_cluster_gini(true_labels, cluster_assignments):
    """
    Calculates Gini Impurity for each cluster based on ground truth labels.
    Gini = 1 - sum(pi^2)
    0.0 = Perfect purity (only one signal type in the cluster).
    Higher values = High impurity (mixed signals).
    """
    unique_clusters = np.unique(cluster_assignments)
    cluster_ginis = {}

    for cluster_id in unique_clusters:
        # Get the ground truth labels for items in this specific cluster
        mask = (cluster_assignments == cluster_id)
        labels_in_cluster = np.array(true_labels)[mask]
        
        if len(labels_in_cluster) == 0:
            cluster_ginis[cluster_id] = 0
            continue

        # Calculate probabilities (pi) for each class (NR, LTE, noise)
        _, counts = np.unique(labels_in_cluster, return_counts=True)
        probabilities = counts / len(labels_in_cluster)
        
        # Apply the Gini formula
        gini = 1 - np.sum(probabilities**2)
        cluster_ginis[cluster_id] = gini
        
        print(f"Cluster {cluster_id} Gini Impurity: {gini:.4f} (n={len(labels_in_cluster)})")
    
    return cluster_ginis

def Kmeans_true_label(title, name, y_true, X):
    """
    Perform K-means clustering and visualize true labels with t-SNE embedding.

    This function applies K-means clustering (k=3) to the input data,
    computes a t-SNE embedding for visualization, and plots the data points
    colored by their true labels. The figure is saved as a PNG file.

    Parameters
    ----------
    title : str
        Title for the plot.
    name : str
        Base name for the output file.
    y_true : np.ndarray
        Ground truth labels for the samples.
    X : np.ndarray
        Input feature matrix.

    Returns
    -------
    None
        Saves the plot as '{name}_true_labels.png'.
    """

    kmeans = KMeans(n_init=10, n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    label_names = {
            0: 'Noise + LTE',
            1: 'Noise + NR', 
            2: 'Noise + LTE + NR'
        }
    colors = ['#E69F00', '#56B4E9', '#009E73']
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    
    for label in np.unique(y_true):
        mask = (y_true == label)
        plt.scatter(
            X_embedded[mask, 0], 
            X_embedded[mask, 1], 
            c=colors[label], 
            label=label_names[label], 
            alpha=0.7, 
            edgecolors='white', 
            linewidth=0.5,
            s=60
        )
    
    plt.title(f"K-means Clustering ($k=3$): {title}", fontsize=26)
    plt.legend(title="Spectogram Types", title_fontsize='20', loc='best', frameon=True, fontsize=20,
              borderpad=0.1,
              labelspacing=0.2,
              handletextpad=0.1,
              borderaxespad=0.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'{name}_true_labels.png', dpi=600, bbox_inches='tight', format='png')

def Kmeans_3_clusters(title, name, y_pred, X_embedded):
    """
    Visualize K-means clustering results with t-SNE embedding.

    This function plots the samples colored by their predicted cluster
    assignments (k=3) using a precomputed embedding (e.g., t-SNE).
    The figure is saved as a PNG file.

    Parameters
    ----------
    title : str
        Title for the plot.
    name : str
        Base name for the output file.
    y_pred : np.ndarray
        Predicted cluster labels from K-means.
    X_embedded : np.ndarray
        2D embedding of the input data (e.g., t-SNE output).

    Returns
    -------
    None
        Saves the plot as '{name}_kmeans_clusters.png' and displays it.
    """

    cluster_colors = ['#D55E00', '#CC79A7', '#F0E442']
    unique_clusters = [0, 1, 2]

    plt.figure(figsize=(10, 6))
    for i, cluster_id in enumerate(unique_clusters):
        mask = (y_pred == cluster_id)
        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=cluster_colors[i],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            s=60
        )
    plt.title(f"K-means Clustering ($k=3$): {title}", fontsize=26)
    plt.legend(title="Spectogram Types", title_fontsize='20', loc='best', frameon=True, fontsize=20,
               borderpad=0.1, labelspacing=0.2, handletextpad=0.1, borderaxespad=0.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'{name}_kmeans_clusters.png', dpi=600, bbox_inches='tight', format='png')
    plt.show()


def UMAP_plot(title, name, y_true, X):
    """
    Perform UMAP dimensionality reduction and visualize true labels.

    This function applies UMAP to reduce the input data to 2D, then plots
    the samples colored by their true labels. The figure is saved as a PNG file.

    Parameters
    ----------
    title : str
        Title for the plot.
    name : str
        Base name for the output file.
    y_true : np.ndarray
        Ground truth labels for the samples.
    X : np.ndarray
        Input feature matrix.

    Returns
    -------
    None
        Saves the plot as '{name}_umap.png' and displays it.
    """

    label_names = {
        0: 'Noise + LTE',
        1: 'Noise + NR', 
        2: 'Noise + LTE + NR'
    }
    paper_colors = ['#E69F00', '#56B4E9', '#009E73']
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    
    for label in np.unique(y_true):
        mask = (y_true == label)
        plt.scatter(
            X_umap[mask, 0], 
            X_umap[mask, 1], 
            c=paper_colors[label], 
            label=label_names[label],
            alpha=0.6,
            edgecolors='white',
            linewidth=0.5,
            s=60,
        )
    plt.legend(title="Spectogram Types", title_fontsize='20', loc='best', frameon=True, fontsize=20,
              borderpad=0.1,
              labelspacing=0.2,
              handletextpad=0.1,
              borderaxespad=0.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"UMAP: {title}", fontsize=26)
    plt.savefig(f'{name}_umap.png', dpi=600, bbox_inches='tight', format='png')
    plt.show()

def run_plot_evaluation(
    dataset_name: str,
    pretrain: str = "simclr",
    from_scratch: bool = False,
    backbone_name: str = "resnet50",
    version: int = None,
    test_split: str = "test",
    n_clusters: int = 3,
    title: str = "SimCLR",
    name: str = "simclr_eval",
):
    """
    Full evaluation pipeline for a SimCLR pre-trained backbone on spectrograms:
    1. Extract features
    2. KMeans clustering
    3. Gini Impurity per cluster
    4. t-SNE and UMAP visualizations
    """
    # ── Backbone ──────────────────────────────────────────────────────
    backbone = load_backbone(dataset_name, from_scratch=from_scratch, pretrained=pretrain, version=version, backbone_name=backbone_name)
    backbone.eval()
    backbone.to(device)

    base = SUPERVISED_DIR / test_split
    csv_file = base / "label_tsne" / "label_tsne.csv"
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFeatureDataset(csv_file=csv_file,
                                img_dir=base / "data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # ── Feature extraction ────────────────────────────────────────────
    print(f"Extracting features from {test_split} set ({len(dataset)} samples)...")
    all_features = []
    all_labels   = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            feat   = backbone(images)
            feat   = torch.mean(feat, dim=[2, 3])
            all_features.append(feat.cpu().numpy())
            all_labels.append(labels.numpy())

    X      = np.vstack(all_features)
    X      = normalize(X, norm='l2')
    y_true = np.concatenate(all_labels)

    print("Clases únicas en y_true:", np.unique(y_true))

    # ── KMeans ────────────────────────────────────────────────────────
    print("Running K-Means clustering...")
    kmeans = KMeans(n_init=10, n_clusters=n_clusters, random_state=42)
    y_pred = kmeans.fit_predict(X)

    # ── Gini Impurity ─────────────────────────────────────────────────
    print("Calculating Gini Impurity per cluster...")
    gini_results = calculate_cluster_gini(y_true, y_pred)
    
    csv_path = Path(f"gini_impurity_results_simclr_{version}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "cluster_id", "gini_impurity"])  # header opcional
        for cluster_id, gini in gini_results.items():
            writer.writerow(["simclr_pretrained", cluster_id, f"{gini:.4f}"])

    print(f"Saved to {csv_path}")

    # ── Proyecciones (calcular una sola vez) ──────────────────────────
    name = f"{name}_{version}" if version is not None else name
    title = f"RF-SimCLR_{version}"
    Kmeans_true_label(title, name, y_true, X)
    Kmeans_3_clusters(title, name, y_pred, X)
    UMAP_plot(title, name, y_true, X)

    return gini_results, y_true, y_pred, X

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕒 Date and time of execution: {now}")

    parser = argparse.ArgumentParser(description="SimCLR Pre-training & Downstream Evaluation")
    parser.add_argument("--mode", choices=["pretrain", "linear", "full", "gini", "tsne", "plot"],
                        default="pretrain", help="Mode: pretrain, linear, full, gini, tsne, plot")
    parser.add_argument("--dataset", default="spectogram_rgb",
                        help="Dataset key")
    parser.add_argument("--name", default="simclr_spectogram",
                        help="Experiment name")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Use random weights backbone")
    parser.add_argument("--pretrain", choices=["none", "imagenet", "coco"], default="none",
                        help="Pretraining type for downstream")
    parser.add_argument("--transform", choices=["base", "no_crop", "crop_only", "no_jitter", "jitter_only", "no_blur", "masks_noise",
                                                "blur_only", "masks_only", "masks_weak", "masks_strong", "horizontal_weak", "vertical_weak", "noise_only", "noise_strong", 
                                                "no_grayscale", "full", "minimal"], default="base",
                        help="Contrastive transform variant for pre-training")
    parser.add_argument("--backbone", choices=["resnet50", "resnet18"], default="resnet50",
                        help="Backbone architecture for pre-training")
    parser.add_argument("--version", type=int, default=None,
                        help="Specific SimCLR checkpoint version to use (e.g. 5 for version_5)")
    args = parser.parse_args()

    if args.mode == "pretrain":
        pretrain_backbone(args.dataset, transform_variant=args.transform, backbone_name=args.backbone)
        
    elif args.mode == "linear":
        linear_readout_evaluation(
            dataset_name=args.dataset,
            experiment_name=args.name,
            from_scratch=args.from_scratch,
            pretrain=args.pretrain
        )

    elif args.mode == "full":
        full_finetune_evaluation(
            dataset_name=args.dataset,
            experiment_name=args.name,
            from_scratch=args.from_scratch,
            pretrain=args.pretrain,
            version=args.version,
            backbone_name=args.backbone
        )
        
    elif args.mode == "plot":
        run_plot_evaluation(
            dataset_name=args.dataset, 
            pretrain=args.pretrain, 
            from_scratch=args.from_scratch, 
            version=args.version, 
            backbone_name=args.backbone, 
            title=args.name, 
            name=args.name
        )
        