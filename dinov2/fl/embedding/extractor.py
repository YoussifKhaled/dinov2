# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
DINOv2 embedding extractor for FL data heterogeneity pipeline.

Extracts [CLS] token embeddings from images using frozen DINOv2 backbone.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from ..config import FLConfig, get_model_config
from ..datasets.cityscapes_list import CityscapesListDataset, create_dataloader


class DINOv2Extractor:
    """Extract embeddings using DINOv2 backbone.
    
    Loads a pre-trained DINOv2 model and extracts [CLS] token embeddings
    for scene-level representation.
    
    Args:
        model_name: Name of DINOv2 model (e.g., 'dinov2_vitl14')
        device: Device to run inference on
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = get_model_config(model_name)
        self.embed_dim = self.model_config["embed_dim"]
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self) -> nn.Module:
        """Load pre-trained DINOv2 model from torch hub."""
        print(f"Loading {self.model_name} from torch hub...")
        
        model = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            pretrained=True,
        )
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        print(f"Model loaded on {self.device}, embed_dim={self.embed_dim}")
        return model
    
    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for a batch of images.
        
        Args:
            images: Batch of images [B, 3, H, W]
            
        Returns:
            CLS token embeddings [B, embed_dim]
        """
        images = images.to(self.device)
        
        # Get features including CLS token
        # forward_features returns dict with 'x_norm_clstoken'
        features = self.model.forward_features(images)
        cls_tokens = features["x_norm_clstoken"]  # [B, embed_dim]
        
        return cls_tokens.cpu()
    
    def extract_all(
        self,
        dataloader: torch.utils.data.DataLoader,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[int], List[str]]:
        """Extract embeddings for entire dataset.
        
        Args:
            dataloader: DataLoader yielding (indices, images, paths)
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of:
                - embeddings: Tensor [N, embed_dim]
                - indices: List of sample indices
                - paths: List of image paths
        """
        all_embeddings = []
        all_indices = []
        all_paths = []
        
        iterator = tqdm(dataloader, desc="Extracting embeddings") if show_progress else dataloader
        
        for indices, images, paths in iterator:
            embeddings = self.extract_batch(images)
            
            all_embeddings.append(embeddings)
            all_indices.extend(indices)
            all_paths.extend(paths)
            
        # Concatenate all embeddings
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        
        return embeddings_tensor, all_indices, all_paths


def extract_embeddings(
    config: FLConfig,
    save: bool = True,
) -> Dict:
    """Run embedding extraction phase.
    
    Args:
        config: FLConfig with all settings
        save: Whether to save results to file
        
    Returns:
        Dictionary with embeddings, indices, and paths
    """
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    print(f"Loading dataset from {config.dataset_list_file}")
    dataset = CityscapesListDataset(
        list_file=config.dataset_list_file,
        base_path=config.base_path,
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,  # Preserve order
    )
    
    # Create extractor and run
    extractor = DINOv2Extractor(
        model_name=config.model_name,
    )
    
    embeddings, indices, paths = extractor.extract_all(dataloader)
    
    # Prepare output
    result = {
        "embeddings": embeddings,  # [N, embed_dim]
        "indices": indices,        # List[int]
        "image_paths": paths,      # List[str]
        "model_name": config.model_name,
        "embed_dim": extractor.embed_dim,
        "n_samples": len(indices),
    }
    
    # Save if requested
    if save:
        print(f"Saving embeddings to {config.embeddings_path}")
        torch.save(result, config.embeddings_path)
        print(f"Saved {result['n_samples']} embeddings with dim {result['embed_dim']}")
    
    return result


def load_embeddings(path: str) -> Dict:
    """Load embeddings from file.
    
    Args:
        path: Path to embeddings.pth file
        
    Returns:
        Dictionary with embeddings and metadata
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    
    data = torch.load(path)
    print(f"Loaded {data['n_samples']} embeddings from {path}")
    return data
