# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Dirichlet-based data partitioning for Federated Learning.

Creates non-IID data distributions across FL clients by sampling
from a Dirichlet distribution over embedding clusters.

Lower alpha values create more heterogeneous (non-IID) distributions,
while higher values approach IID partitioning.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from ..config import FLConfig
from ..clustering.kmeans import load_clusters, get_cluster_distribution, get_samples_by_cluster


class DirichletPartitioner:
    """Partition data across FL clients using Dirichlet distribution.
    
    For each cluster k, samples proportions for N clients from Dir(alpha * p_k),
    where p_k is the prior probability of cluster k.
    
    Args:
        n_clients: Number of FL clients
        alpha: Dirichlet concentration parameter
        min_samples_per_client: Minimum samples each client must receive
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_clients: int = 10,
        alpha: float = 0.5,
        min_samples_per_client: int = 10,
        seed: int = 42,
    ):
        self.n_clients = n_clients
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
        self.seed = seed
        
        np.random.seed(seed)
        
    def partition(
        self,
        cluster_labels: np.ndarray,
        indices: List[int],
    ) -> Dict[int, List[int]]:
        """Partition samples across clients based on cluster assignments.
        
        Args:
            cluster_labels: Array of cluster assignments for each sample
            indices: List of sample indices corresponding to cluster_labels
            
        Returns:
            Dictionary mapping client_id -> list of sample indices
        """
        n_samples = len(indices)
        n_clusters = len(np.unique(cluster_labels))
        
        # Get cluster distribution (prior)
        cluster_dist = get_cluster_distribution(cluster_labels)
        
        # Group samples by cluster
        cluster_to_samples = get_samples_by_cluster(cluster_labels, indices)
        
        # Initialize client data
        client_data: Dict[int, List[int]] = {i: [] for i in range(self.n_clients)}
        
        # For each cluster, distribute its samples across clients
        for cluster_id in range(n_clusters):
            if cluster_id not in cluster_to_samples:
                continue
                
            cluster_samples = cluster_to_samples[cluster_id]
            n_cluster_samples = len(cluster_samples)
            
            # Sample proportions from Dirichlet(alpha * p_k)
            # Using uniform prior scaled by alpha
            dir_params = np.repeat(self.alpha, self.n_clients)
            proportions = np.random.dirichlet(dir_params)
            
            # Convert proportions to sample counts
            counts = (proportions * n_cluster_samples).astype(int)
            
            # Distribute any remaining samples due to rounding
            remainder = n_cluster_samples - counts.sum()
            if remainder > 0:
                # Add remainder to clients with highest proportions
                top_clients = np.argsort(proportions)[-remainder:]
                counts[top_clients] += 1
            
            # Shuffle samples within cluster
            shuffled_samples = np.random.permutation(cluster_samples).tolist()
            
            # Assign samples to clients
            start_idx = 0
            for client_id in range(self.n_clients):
                end_idx = start_idx + counts[client_id]
                client_data[client_id].extend(shuffled_samples[start_idx:end_idx])
                start_idx = end_idx
        
        # Validate and potentially rebalance
        client_data = self._ensure_minimum_samples(client_data)
        
        return client_data
    
    def _ensure_minimum_samples(
        self,
        client_data: Dict[int, List[int]],
    ) -> Dict[int, List[int]]:
        """Ensure each client has minimum required samples.
        
        Moves samples from clients with excess to those with deficit.
        """
        # Check which clients need more samples
        deficit_clients = [
            c for c in range(self.n_clients)
            if len(client_data[c]) < self.min_samples_per_client
        ]
        
        if not deficit_clients:
            return client_data
            
        # Find clients with excess samples
        surplus_clients = [
            c for c in range(self.n_clients)
            if len(client_data[c]) > self.min_samples_per_client * 2
        ]
        
        for deficit_client in deficit_clients:
            needed = self.min_samples_per_client - len(client_data[deficit_client])
            
            for surplus_client in surplus_clients:
                if needed <= 0:
                    break
                    
                available = len(client_data[surplus_client]) - self.min_samples_per_client
                if available <= 0:
                    continue
                    
                transfer = min(needed, available)
                # Move samples
                transferred = client_data[surplus_client][:transfer]
                client_data[surplus_client] = client_data[surplus_client][transfer:]
                client_data[deficit_client].extend(transferred)
                needed -= transfer
        
        return client_data
    
    def compute_statistics(
        self,
        client_data: Dict[int, List[int]],
        cluster_labels: np.ndarray,
        indices: List[int],
    ) -> Dict:
        """Compute statistics about the partitioning.
        
        Args:
            client_data: Client -> sample indices mapping
            cluster_labels: Original cluster labels
            indices: Original sample indices
            
        Returns:
            Dictionary with partition statistics
        """
        # Create index -> cluster mapping
        idx_to_cluster = {idx: int(cluster_labels[i]) for i, idx in enumerate(indices)}
        
        stats = {
            "n_clients": self.n_clients,
            "alpha": self.alpha,
            "samples_per_client": {},
            "clusters_per_client": {},
            "client_cluster_distribution": {},
        }
        
        for client_id, samples in client_data.items():
            stats["samples_per_client"][client_id] = len(samples)
            
            # Get cluster distribution for this client
            client_clusters = [idx_to_cluster[idx] for idx in samples if idx in idx_to_cluster]
            unique_clusters = np.unique(client_clusters)
            stats["clusters_per_client"][client_id] = len(unique_clusters)
            
            # Detailed distribution
            cluster_counts = {}
            for c in client_clusters:
                cluster_counts[c] = cluster_counts.get(c, 0) + 1
            stats["client_cluster_distribution"][client_id] = cluster_counts
        
        # Summary statistics
        sample_counts = list(stats["samples_per_client"].values())
        stats["total_samples"] = sum(sample_counts)
        stats["min_samples"] = min(sample_counts)
        stats["max_samples"] = max(sample_counts)
        stats["mean_samples"] = np.mean(sample_counts)
        stats["std_samples"] = np.std(sample_counts)
        
        return stats


def partition_data(
    config: FLConfig,
    clusters_path: Optional[str] = None,
    save: bool = True,
) -> Dict:
    """Run Dirichlet partitioning phase.
    
    Args:
        config: FLConfig with partitioning settings
        clusters_path: Path to clusters file (uses config default if None)
        save: Whether to save results to file
        
    Returns:
        Dictionary with client splits and statistics
    """
    # Load clusters
    clust_path = clusters_path or config.clusters_path
    cluster_data = load_clusters(clust_path)
    
    cluster_labels = cluster_data["cluster_labels"]
    indices = cluster_data["indices"]
    image_paths = cluster_data["image_paths"]
    
    print(f"Partitioning {len(indices)} samples across {config.n_clients} clients")
    print(f"Alpha (heterogeneity): {config.alpha} (lower = more non-IID)")
    
    # Create partitioner and run
    partitioner = DirichletPartitioner(
        n_clients=config.n_clients,
        alpha=config.alpha,
        min_samples_per_client=config.min_samples_per_client,
        seed=config.seed,
    )
    
    client_data = partitioner.partition(cluster_labels, indices)
    
    # Compute statistics
    stats = partitioner.compute_statistics(client_data, cluster_labels, indices)
    
    # Print summary
    print(f"\nPartitioning complete:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Samples per client: {stats['min_samples']} - {stats['max_samples']} "
          f"(mean: {stats['mean_samples']:.1f}, std: {stats['std_samples']:.1f})")
    
    # Create path mapping for convenience
    idx_to_path = {idx: path for idx, path in zip(indices, image_paths)}
    client_paths = {
        client_id: [idx_to_path[idx] for idx in sample_indices]
        for client_id, sample_indices in client_data.items()
    }
    
    # Prepare output
    result = {
        "client_data": client_data,        # Dict[int, List[int]] - indices
        "client_paths": client_paths,      # Dict[int, List[str]] - paths
        "statistics": stats,
        "config": {
            "n_clients": config.n_clients,
            "alpha": config.alpha,
            "n_clusters": cluster_data["n_clusters"],
            "seed": config.seed,
        },
    }
    
    # Save if requested
    if save:
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"\nSaving client splits to {config.splits_path}")
        torch.save(result, config.splits_path)
    
    return result


def load_client_splits(path: str) -> Dict:
    """Load client splits from file.
    
    Args:
        path: Path to client_splits.pth file
        
    Returns:
        Dictionary with client data and statistics
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Client splits file not found: {path}")
    
    data = torch.load(path)
    n_clients = len(data["client_data"])
    total = sum(len(v) for v in data["client_data"].values())
    print(f"Loaded splits for {n_clients} clients ({total} total samples)")
    return data
