# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
K-Means clustering for embedding-based scene grouping.

Groups images by semantic similarity based on DINOv2 embeddings.
Cluster assignments serve as pseudo-labels for Dirichlet partitioning.
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans

from ..config import FLConfig
from ..embedding.extractor import load_embeddings


def cluster_embeddings(
    config: FLConfig,
    embeddings_path: Optional[str] = None,
    save: bool = True,
) -> Dict:
    """Run K-Means clustering on embeddings.
    
    Args:
        config: FLConfig with clustering settings
        embeddings_path: Path to embeddings file (uses config default if None)
        save: Whether to save results to file
        
    Returns:
        Dictionary with cluster assignments and metadata
    """
    # Load embeddings
    emb_path = embeddings_path or config.embeddings_path
    emb_data = load_embeddings(emb_path)
    
    embeddings = emb_data["embeddings"]  # [N, embed_dim]
    
    # Convert to numpy for sklearn
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = embeddings
    
    print(f"Clustering {len(embeddings_np)} embeddings into {config.n_clusters} clusters...")
    
    # Run K-Means
    kmeans = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.clustering_random_state,
        n_init=10,
        max_iter=300,
        verbose=0,
    )
    
    cluster_labels = kmeans.fit_predict(embeddings_np)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    # Compute cluster statistics
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
    
    print(f"Clustering complete. Inertia: {inertia:.2f}")
    print(f"Cluster sizes: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")
    
    # Prepare output
    result = {
        "cluster_labels": cluster_labels,      # np.ndarray [N]
        "centroids": centroids,                # np.ndarray [K, embed_dim]
        "n_clusters": config.n_clusters,
        "cluster_sizes": cluster_sizes,        # Dict[int, int]
        "inertia": inertia,
        "indices": emb_data["indices"],        # Preserve original indices
        "image_paths": emb_data["image_paths"],
        "n_samples": len(cluster_labels),
    }
    
    # Save if requested
    if save:
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"Saving clusters to {config.clusters_path}")
        torch.save(result, config.clusters_path)
    
    return result


def load_clusters(path: str) -> Dict:
    """Load cluster assignments from file.
    
    Args:
        path: Path to clusters.pth file
        
    Returns:
        Dictionary with cluster labels and metadata
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clusters file not found: {path}")
    
    data = torch.load(path)
    print(f"Loaded {data['n_samples']} cluster assignments ({data['n_clusters']} clusters)")
    return data


def get_cluster_distribution(cluster_labels: np.ndarray) -> np.ndarray:
    """Compute the empirical distribution over clusters.
    
    Args:
        cluster_labels: Array of cluster assignments
        
    Returns:
        Probability distribution over clusters [K]
    """
    n_clusters = len(np.unique(cluster_labels))
    counts = np.bincount(cluster_labels, minlength=n_clusters)
    return counts / counts.sum()


def get_samples_by_cluster(
    cluster_labels: np.ndarray,
    indices: list,
) -> Dict[int, list]:
    """Group sample indices by cluster.
    
    Args:
        cluster_labels: Array of cluster assignments
        indices: List of sample indices
        
    Returns:
        Dictionary mapping cluster_id -> list of sample indices
    """
    cluster_to_samples = {}
    
    for idx, cluster_id in zip(indices, cluster_labels):
        cluster_id = int(cluster_id)
        if cluster_id not in cluster_to_samples:
            cluster_to_samples[cluster_id] = []
        cluster_to_samples[cluster_id].append(idx)
    
    return cluster_to_samples
