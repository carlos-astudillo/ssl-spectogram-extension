"""
Script for pre-training a backbone with SimCLR on spectrogram datasets and downstream evaluation.
Uses the MINERVA framework: https://github.com/discovery-unicamp/Minerva

This script supports:
- Pre-training with SimCLR
- Linear readout evaluation (frozen backbone + linear head)
- Full fine-tuning evaluation
"""
import os
import re
import csv
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import copy
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

# Torchvision v2 transforms
from torchvision.transforms.v2 import (
    Compose, ToImage, ToDtype,
    RandomHorizontalFlip, RandomResizedCrop,
    RandomApply, ColorJitter, RandomGrayscale,
    GaussianBlur, Normalize
)

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchmetrics.classification import JaccardIndex

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.core.module")

# =============================================================================
# Custom transforms
# =============================================================================
class Identity_2:
    """
    Transform that converts numpy array to tensor and applies ImageNet normalization.
    Used for input images in downstream tasks.
    """

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(x)

    def __str__(self) -> str:
        return "Identity_2 (numpy → tensor → ImageNet norm)"


class Format_label_img(_Transform):
    """
    Transform for segmentation labels.
    - Converts numpy array to long tensor
    - Subtracts 1 if labels start from 1
    - Resizes to fixed target size with nearest interpolation
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    def __call__(self, x: np.ndarray) -> torch.Tensor:
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

# =============================================================================
# SimCLR Model
# =============================================================================
class SimCLR(L.LightningModule):
    """
    SimCLR model for self-supervised contrastive learning.

    Implements the SimCLR framework using NT-Xent loss for contrastive pre-training.

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

    def __init__(
        self,
        backbone: nn.Module,
        projection_head: nn.Module,
        flatten: bool = True,
        temperature: float = 0.5,
        lr: float = 1e-3,
    ):
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
def get_contrastive_transform():
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
        RandomApply(
            [ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)],
            p=0.8
        ),
        RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        Normalize(mean=MEAN, std=STD),
    ])

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
class LinearCustomHead(nn.Module):
    """
    Simple linear head for segmentation readout.

    Parameters
    ----------
    in_channels : int
        Number of input channels from the backbone.
    num_classes : int
        Number of segmentation classes.
    target_size : Tuple[int, int], optional
        Target spatial size for upsampling, by default (256, 256)
    """

    def __init__(self, in_channels: int, num_classes: int, target_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            print(f"Error máscara {idx}: {y.shape}")
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

IMG_SIZE = 256
BATCH_SIZE = 512
NUM_WORKERS = 4
MAX_EPOCHS = 100
TEMPERATURE = 0.5
LR = 1e-4

BATCH_SIZE_FINE = 32
MAX_EPOCHS_FINE = 50

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

# Root directories
EXPERIMENTS_ROOT = Path("/Experiments")
DATASETS_ROOT = Path("/Dataset")
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

def load_datamodule(dataset_name: str) -> MinervaDataModule:
    """
    Loads a DataModule for SimCLR pre-training on a specific dataset.

    Parameters
    ----------
    dataset_name : str
        Key from the DATASETS dictionary (e.g., "spectogram_rgb").

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
    contrastive_transform = get_contrastive_transform()

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


def pretrain_backbone(dataset_name: str = "spectogram_rgb") -> None:
    """
    Runs SimCLR pre-training for a given dataset.

    Parameters
    ----------
    dataset_name : str, default="spectogram_rgb"
        Dataset key to use for pre-training.
    """
    print(f"Starting SimCLR pre-training for dataset: {dataset_name}")

    # 1. Ensure output directories exist
    ensure_dirs_for_dataset(dataset_name)

    # 2. Load DataModule
    dm = load_datamodule(dataset_name)

    # 3. Backbone + Projection Head
    backbone = resnet50(weights=None)
    backbone.fc = nn.Identity()

    proj_head = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 128)
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

def get_last_checkpoint(dataset_name: str) -> Optional[str]:
    """
    Searches for the latest checkpoint in the pretext/checkpoints folder of the dataset.
    Priority: most recent best-*.ckpt > last.ckpt.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

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
    pretrained: str = "none"
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

        # class COCOBackboneWrapper(nn.Module):
        #     """Wrapper to extract the deepest feature map from DeepLabV3 backbone."""

        #     def __init__(self, backbone):
        #         super().__init__()
        #         self.backbone = backbone

        #     def forward(self, x):
        #         out = self.backbone(x)
        #         if isinstance(out, dict):
        #             # Prefer 'layer4' or 'out', fallback to last key
        #             if 'layer4' in out:
        #                 return out['layer4']
        #             elif 'out' in out:
        #                 return out['out']
        #             else:
        #                 return out[list(out.keys())[-1]]
        #         return out

        # wrapper = COCOBackboneWrapper(backbone)
        
        # return wrapper

    elif from_scratch:
        print("Loading ResNet50 from scratch (standard backbone, no dilations)")
        rn = resnet50(weights=None, replace_stride_with_dilation=[False, False, False])

        # Explicit construction up to layer4 (no residual avgpool)
        backbone = nn.Sequential(
            rn.conv1,
            rn.bn1,
            rn.relu,
            rn.maxpool,
            rn.layer1,
            rn.layer2,
            rn.layer3,
            rn.layer4
        )
        return backbone

    else:  # SimCLR checkpoint
        print("Loading SimCLR checkpoint")
        ckpt_path = get_last_checkpoint(dataset_name)
        if not ckpt_path:
            raise FileNotFoundError(f"No checkpoint found for dataset: {dataset_name}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        rn = resnet50(weights=None, replace_stride_with_dilation=dilation_cfg)

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
    """
    # Load initial backbone
    backbone = load_backbone(dataset_name, from_scratch=from_scratch, pretrained=pretrain)

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

    percentages = [0.2, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
    low_percent_map = {0.2: "01", 1.0: "06", 5.0: "31", 10.0: "61"}
    high_percent_map = {25.0: "25_percent", 50.0: "50_percent", 75.0: "75_percent"}

    for percent in percentages:
        miou_list = []

        # Copy backbone and create new model
        bb_copy = copy.deepcopy(backbone)
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
# Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Pre-training & Downstream Evaluation")
    parser.add_argument("--mode", choices=["pretrain", "linear", "full"],
                        default="pretrain", help="Mode: pretrain, linear, full")
    parser.add_argument("--dataset", default="spectogram_rgb",
                        help="Dataset key")
    parser.add_argument("--name", default="simclr_spectogram",
                        help="Experiment name")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Use random weights backbone")
    parser.add_argument("--pretrain", choices=["none", "imagenet", "coco"], default="none",
                        help="Pretraining type for downstream")
    args = parser.parse_args()

    if args.mode == "pretrain":
        pretrain_backbone(args.dataset)

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
            pretrain=args.pretrain
        )