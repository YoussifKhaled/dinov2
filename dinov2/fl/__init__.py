# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Federated Learning Data Heterogeneity Pipeline using DINOv2 Embeddings.

This module implements embedding-based data heterogeneity for FL on semantic
segmentation tasks (Cityscapes), following Borazjani et al.'s methodology.

Pipeline Phases:
    1. Extract embeddings: Use DINOv2 to extract CLS token embeddings
    2. Cluster embeddings: Apply K-Means to group similar scenes
    3. Partition data: Use Dirichlet distribution to create non-IID client splits

Usage (from command line):
    # Run full pipeline
    python -m dinov2.fl.scripts.run_pipeline \\
        --dataset_list_file train_fine.txt \\
        --base_path /kaggle/input/cityscapes \\
        --output_dir ./fl_outputs \\
        --n_clusters 16 --n_clients 10 --alpha 0.5
    
    # Or run phases separately
    python -m dinov2.fl.scripts.run_extraction ...
    python -m dinov2.fl.scripts.run_clustering ...
    python -m dinov2.fl.scripts.run_partitioning ...

Usage (from Python):
    from dinov2.fl import FLConfig
    from dinov2.fl.embedding import extract_embeddings
    from dinov2.fl.clustering import cluster_embeddings
    from dinov2.fl.partitioning import partition_data
    
    config = FLConfig(
        dataset_list_file="train_fine.txt",
        base_path="/kaggle/input/cityscapes",
        n_clusters=16,
        n_clients=10,
        alpha=0.5,
    )
    
    extract_embeddings(config)
    cluster_embeddings(config)
    partition_data(config)
"""

from .config import FLConfig, get_model_config, MODEL_CONFIGS
from .datasets import CityscapesListDataset
from .embedding import DINOv2Extractor, extract_embeddings
from .clustering import cluster_embeddings, load_clusters
from .partitioning import partition_data, load_client_splits, DirichletPartitioner

__all__ = [
    # Config
    "FLConfig",
    "get_model_config",
    "MODEL_CONFIGS",
    # Dataset
    "CityscapesListDataset",
    # Embedding
    "DINOv2Extractor",
    "extract_embeddings",
    # Clustering
    "cluster_embeddings",
    "load_clusters",
    # Partitioning
    "partition_data",
    "load_client_splits",
    "DirichletPartitioner",
]
