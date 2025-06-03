from pathlib import Path
from PIL import Image
import cv2
import random
import wget
import numpy as np
from minerva.data.readers.png_reader import PNGReader
from minerva.data.readers.patched_array_reader import NumpyArrayReader
from minerva.data.data_modules.base import MinervaDataModule

from minerva.transforms.transform import _Transform, Identity
from minerva.data.datasets.base import SimpleDataset

from minerva.models.ssl.byol import BYOL
from torchvision import transforms
import lightning as L
import torch
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
import torchvision.models as models

class VerticalLineMask(_Transform):
    def __init__(self, min_lines: int = 2, max_lines: int = 5, line_width: int = 16, seed: int = None):
        """
        Creates a mask with random vertical lines of fixed width.

        Parameters
        ----------
        min_lines : int
            Minimum number of vertical lines to draw.
        max_lines : int
            Maximum number of vertical lines to draw.
        line_width : int
            Width of each vertical line in pixels.
        seed : int, optional
            Seed for reproducibility.
        """
        assert min_lines > 0 and max_lines >= min_lines, "Invalid line count range."
        assert line_width > 0, "Line width must be positive."
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.line_width = line_width
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            h, w = x.shape
            c = 1
            format = 'HW'
        elif x.ndim == 3:
            if x.shape[0] == 1 or x.shape[0] == 3:
                c, h, w = x.shape
                format = 'CHW'
            else:
                h, w, c = x.shape
                format = 'HWC'
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        max_start = w - self.line_width
        if max_start <= 0:
            raise ValueError(f"Image width {w} too small for line width {self.line_width}")

        max_possible_lines = max_start // self.line_width
        num_lines = min(self.rng.integers(self.min_lines, self.max_lines + 1), max_possible_lines)

        possible_starts = np.arange(0, max_start + 1, self.line_width)
        start_positions = self.rng.choice(possible_starts, size=num_lines, replace=False)

        x_out = x.copy()

        if x.dtype == np.uint8:
            noise_max = 255
        else:
            noise_max = 1.0

        for start in start_positions:
            end = start + self.line_width
            if format == 'HW':
                x_out[:, start:end] = 0
                #x_out[:, start:end] = self.rng.random((h, self.line_width)) * noise_max
            elif format == 'HWC':
                x_out[:, start:end] = 0
                #x_out[:, start:end, :] = self.rng.random((h, self.line_width, c)) * noise_max
            elif format == 'CHW':
                x_out[:, start:end] = 0
                #x_out[:, :, start:end] = self.rng.random((c, h, self.line_width)) * noise_max

        #img_chw = np.transpose(x_out, (2, 0, 1))
        transform = transforms.ToTensor()

        return transform(x_out)

class Identity_2(_Transform):
    """This class is a dummy transform that does nothing. It is useful when
    you want to skip a transform in a pipeline.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        transform = transforms.ToTensor()
        return transform(x)

    def __str__(self) -> str:
        return "Identity()"

class ColorJitter(_Transform):
    def __init__(
        self,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        hue: float = 0.0,
    ):
        """
        Applies fixed adjustments to brightness, contrast, saturation, and hue to an input image.

        Parameters
        ----------
        brightness : float, optional
            Fixed factor for brightness adjustment. A value of 1.0 means no change. Defaults to 1.0.
        contrast : float, optional
            Fixed factor for contrast adjustment. A value of 1.0 means no change. Defaults to 1.0.
        saturation : float, optional
            Fixed factor for saturation adjustment. A value of 1.0 means no change. Defaults to 1.0.
        hue : float, optional
            Fixed degree shift for hue adjustment, in the range [-180, 180]. Defaults to 0.0.

        Returns
        -------
        np.ndarray
            The transformed image with fixed brightness, contrast, saturation, and hue adjustments applied.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Convert to HSV for hue/saturation adjustment
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Brightness adjustment
        image[..., 2] = np.clip(image[..., 2] * self.brightness, 0, 255)

        # Saturation adjustment
        image[..., 1] = np.clip(image[..., 1] * self.saturation, 0, 255)

        # Contrast adjustment
        mean = image[..., 2].mean()
        image[..., 2] = np.clip((image[..., 2] - mean) * self.contrast + mean, 0, 255)

        # Hue adjustment
        image[..., 0] = (image[..., 0] + self.hue) % 180

        transform = transforms.ToTensor()

        return transform(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB))

    def __str__(self) -> str:
        return f"ColorJitter(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue})"

def main():
    train_data_reader = PNGReader(
        path=Path('spectogram_rgb')
    )

    #transform = transforms.ToTensor()

    train_dataset_0 = SimpleDataset(
        readers=[train_data_reader, train_data_reader],
        transforms=[VerticalLineMask(), ColorJitter(brightness=2, contrast=3, saturation=2.5, hue=50)],
        return_single = False
    )

    data_module = MinervaDataModule(
        train_dataset=train_dataset_0,
        batch_size=32,
        num_workers=2,
        name="Spectogram Dataset",
    )

    resnet50_pretrained = models.resnet50(weights='DEFAULT')

    model = BYOL(
        backbone = torch.nn.Sequential(*list(resnet50_pretrained.children())[:-1])
    )

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, data_module)

    trainer.save_checkpoint("byol_pretrained_0_resnet50.ckpt")

    """byol_pretrained_0_resnet50 - loss: -0.988"""
    """byol_pretrained_0_resnet18 - loss: -0.996"""

if __name__ == "__main__":
    main()
