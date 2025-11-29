# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Configuration module for FL Data Heterogeneity Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import argparse


@dataclass
class FLConfig:
    """Configuration for FL embedding-based data heterogeneity pipeline.
    
    Attributes:
        dataset_list_file: Path to train_fine.txt with image-label pairs
        base_path: Base path to remap dataset paths (e.g., /kaggle/input/cityscapes)
        output_dir: Directory to save all outputs (embeddings, clusters, splits)
        
        model_name: DINOv2 model variant to use
        embed_dim: Embedding dimension (must match model)
        batch_size: Batch size for inference
        num_workers: DataLoader workers
        
        n_clusters: Number of K-Means clusters (scene categories)
        clustering_random_state: Random seed for reproducibility
        
        n_clients: Number of FL clients
        alpha: Dirichlet concentration parameter (lower = more heterogeneity)
        min_samples_per_client: Minimum images per client
        seed: Global random seed
    """
    # Data paths
    dataset_list_file: str = "train_fine.txt"
    base_path: Optional[str] = None  # Set to remap paths for Kaggle/Colab
    output_dir: str = "./fl_outputs"
    
    # Model settings
    model_name: str = "dinov2_vitl14"
    embed_dim: int = 1024  # vitl14 dimension
    batch_size: int = 32
    num_workers: int = 4
    
    # Clustering settings
    n_clusters: int = 16
    clustering_random_state: int = 42
    
    # Partitioning settings
    n_clients: int = 10
    alpha: float = 0.5  # Lower = more non-IID
    min_samples_per_client: int = 10
    seed: int = 42
    
    # Derived paths (set in __post_init__)
    embeddings_path: str = field(init=False)
    clusters_path: str = field(init=False)
    splits_path: str = field(init=False)
    
    def __post_init__(self):
        """Set derived paths based on output_dir."""
        output = Path(self.output_dir)
        self.embeddings_path = str(output / "embeddings.pth")
        self.clusters_path = str(output / "clusters.pth")
        self.splits_path = str(output / "client_splits.pth")
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FLConfig":
        """Create config from parsed command-line arguments."""
        return cls(
            dataset_list_file=getattr(args, "dataset_list_file", cls.dataset_list_file),
            base_path=getattr(args, "base_path", None),
            output_dir=getattr(args, "output_dir", cls.output_dir),
            model_name=getattr(args, "model_name", cls.model_name),
            batch_size=getattr(args, "batch_size", cls.batch_size),
            num_workers=getattr(args, "num_workers", cls.num_workers),
            n_clusters=getattr(args, "n_clusters", cls.n_clusters),
            n_clients=getattr(args, "n_clients", cls.n_clients),
            alpha=getattr(args, "alpha", cls.alpha),
            seed=getattr(args, "seed", cls.seed),
        )
    
    @staticmethod
    def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add common arguments to any phase's argument parser."""
        parser.add_argument(
            "--dataset_list_file", type=str, default="train_fine.txt",
            help="Path to dataset list file (train_fine.txt format)"
        )
        parser.add_argument(
            "--base_path", type=str, default=None,
            help="Base path to remap dataset paths for Kaggle/Colab"
        )
        parser.add_argument(
            "--output_dir", type=str, default="./fl_outputs",
            help="Directory for all pipeline outputs"
        )
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Random seed for reproducibility"
        )
        return parser


# Model configurations for reference
MODEL_CONFIGS = {
    "dinov2_vits14": {"embed_dim": 384, "patch_size": 14},
    "dinov2_vitb14": {"embed_dim": 768, "patch_size": 14},
    "dinov2_vitl14": {"embed_dim": 1024, "patch_size": 14},
    "dinov2_vitg14": {"embed_dim": 1536, "patch_size": 14},
}


def get_model_config(model_name: str) -> dict:
    """Get model configuration by name."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]
