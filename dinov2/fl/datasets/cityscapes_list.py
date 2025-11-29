# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Cityscapes dataset loader from text file listing.

Reads train_fine.txt format:
    <image_path> <label_path>
    
Each line contains space-separated paths to the RGB image and its label mask.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# DINOv2 normalization constants (ImageNet)
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_dinov2_transform(
    resize_size: int = 518,  # Multiple of 14 for ViT-14 models
    center_crop: bool = True,
) -> T.Compose:
    """Create transform pipeline for DINOv2 inference.
    
    Args:
        resize_size: Size to resize images (should be multiple of 14)
        center_crop: Whether to center crop after resize
        
    Returns:
        torchvision Compose transform
    """
    transforms_list = [
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
    ]
    if center_crop:
        transforms_list.append(T.CenterCrop(resize_size))
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return T.Compose(transforms_list)


class CityscapesListDataset(Dataset):
    """Dataset for Cityscapes images from a text file listing.
    
    Reads a text file where each line contains:
        <image_path> <label_path>
    
    Only loads images (not labels) for embedding extraction.
    
    Args:
        list_file: Path to train_fine.txt or similar listing file
        base_path: Optional base path to remap original paths
                   e.g., remap '/content/drive/MyDrive/...' to '/kaggle/input/...'
        transform: Transform to apply to images
        original_prefix: The prefix in the list file to replace
    """
    
    def __init__(
        self,
        list_file: str,
        base_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        original_prefix: str = "/content/drive/MyDrive/Datasets/Cityscapes/Fine",
    ):
        self.list_file = list_file
        self.base_path = base_path
        self.original_prefix = original_prefix
        self.transform = transform or make_dinov2_transform()
        
        # Parse the listing file
        self.image_paths: List[str] = []
        self.label_paths: List[str] = []
        self._parse_list_file()
        
    def _parse_list_file(self) -> None:
        """Parse the listing file to extract image and label paths."""
        if not os.path.exists(self.list_file):
            raise FileNotFoundError(f"List file not found: {self.list_file}")
            
        with open(self.list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid line format: {line}")
                    
                img_path, label_path = parts
                
                # Remap paths if base_path is provided
                if self.base_path is not None:
                    img_path = self._remap_path(img_path)
                    label_path = self._remap_path(label_path)
                    
                self.image_paths.append(img_path)
                self.label_paths.append(label_path)
                
        print(f"Loaded {len(self.image_paths)} image paths from {self.list_file}")
    
    def _remap_path(self, original_path: str) -> str:
        """Remap a path from original prefix to base_path.
        
        Example:
            original: /content/drive/MyDrive/Datasets/Cityscapes/Fine/leftImg8bit/train/aachen/...
            base_path: /kaggle/input/cityscapes
            result: /kaggle/input/cityscapes/leftImg8bit/train/aachen/...
        """
        if self.base_path is None:
            return original_path
            
        if original_path.startswith(self.original_prefix):
            relative = original_path[len(self.original_prefix):]
            # Remove leading slash if present
            relative = relative.lstrip("/")
            return os.path.join(self.base_path, relative)
        else:
            # If prefix doesn't match, return as-is
            return original_path
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, str]:
        """Get an image by index.
        
        Returns:
            Tuple of (index, image_tensor, image_path)
        """
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
            
        if self.transform is not None:
            image = self.transform(image)
            
        return idx, image, img_path
    
    def get_image_path(self, idx: int) -> str:
        """Get image path by index."""
        return self.image_paths[idx]
    
    def get_label_path(self, idx: int) -> str:
        """Get label path by index."""
        return self.label_paths[idx]


def create_dataloader(
    dataset: CityscapesListDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the Cityscapes dataset.
    
    Args:
        dataset: CityscapesListDataset instance
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    def collate_fn(batch):
        """Custom collate to handle (idx, image, path) tuples."""
        indices = [item[0] for item in batch]
        images = torch.stack([item[1] for item in batch])
        paths = [item[2] for item in batch]
        return indices, images, paths
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
