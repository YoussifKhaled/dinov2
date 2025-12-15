# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
City-based data partitioning for Federated Learning.

Creates partitions where each city is clustered separately into K clusters,
then distributed across 5 clients per city using Dirichlet distribution.

This creates a natural geographic + semantic heterogeneity:
- Geographic: Each city group is isolated
- Semantic: Within each city, clients specialize in different scene types
"""

import os
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans

from ..config import FLConfig
from ..embedding.extractor import load_embeddings
from .dirichlet import DirichletPartitioner


def extract_city_from_path(image_path: str) -> str:
    """Extract city name from Cityscapes image path.
    
    Example:
        leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png -> aachen
    """
    parts = image_path.split("/")
    for i, part in enumerate(parts):
        if part == "train" and i + 1 < len(parts):
            return parts[i + 1]
    
    # Fallback: extract from filename
    filename = os.path.basename(image_path)
    city = filename.split("_")[0]
    return city


def cluster_city_embeddings(
    embeddings: np.ndarray,
    indices: List[int],
    n_clusters: int,
    seed: int = 42,
) -> np.ndarray:
    """Cluster embeddings for a single city.
    
    Args:
        embeddings: Array of embeddings for this city
        indices: Indices of samples in this city
        n_clusters: Number of clusters for this city
        seed: Random seed
        
    Returns:
        Array of cluster labels (local to this city)
    """
    if len(embeddings) < n_clusters:
        # If city has fewer samples than clusters, assign unique labels
        return np.arange(len(embeddings))
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        max_iter=300,
    )
    
    labels = kmeans.fit_predict(embeddings)
    return labels


def partition_city_based(
    embeddings_path: str,
    n_clients_per_city: Optional[int] = None,
    clusters_per_city: Optional[int] = None,
    alpha: float = 0.5,
    min_samples_per_client: int = 5,
    seed: int = 42,
    output_dir: Optional[str] = None,
    save: bool = True,
) -> Dict:
    """Partition data by city with per-city clustering.
    
    For each city:
    1. Extract embeddings for that city
    2. Cluster into K clusters (automatically determined if not specified)
    3. Distribute across N clients using Dirichlet(alpha)
    
    Args:
        embeddings_path: Path to embeddings.pth file
        n_clients_per_city: Number of clients per city (default: 5)
        clusters_per_city: Number of clusters per city (auto if None)
        alpha: Dirichlet concentration parameter
        min_samples_per_client: Minimum samples per client
        seed: Random seed
        output_dir: Output directory for saving
        save: Whether to save results
        
    Returns:
        Dictionary with city-based partition data
    """
    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}")
    emb_data = load_embeddings(embeddings_path)
    embeddings = emb_data["embeddings"].numpy()
    image_paths = emb_data["image_paths"]
    indices = list(range(len(image_paths)))
    
    print(f"Total samples: {len(indices)}")
    
    # Group samples by city
    city_to_indices = defaultdict(list)
    city_to_paths = defaultdict(list)
    
    for idx, path in zip(indices, image_paths):
        city = extract_city_from_path(path)
        city_to_indices[city].append(idx)
        city_to_paths[city].append(path)
    
    cities = sorted(city_to_indices.keys())
    n_cities = len(cities)
    
    print(f"\nFound {n_cities} cities:")
    for city in cities:
        print(f"  {city}: {len(city_to_indices[city])} samples")
    
    # Process each city separately
    global_client_id = 0
    all_client_data = {}
    all_client_paths = {}
    city_statistics = {}

    # Keep the originally requested value; recompute adaptively per city without overwriting it
    requested_n_clients_per_city = n_clients_per_city
    
    np.random.seed(seed)
    
    for city_idx, city in enumerate(cities):
        print(f"\n{'='*60}")
        print(f"Processing City: {city} ({city_idx+1}/{n_cities})")
        print(f"{'='*60}")
        
        city_indices = city_to_indices[city]
        city_embs = embeddings[city_indices]
        n_samples = len(city_indices)
        
        requested_display = (
            str(requested_n_clients_per_city)
            if requested_n_clients_per_city is not None
            else "adaptive"
        )
        print(f"number of clients per city: {requested_display}")

        if requested_n_clients_per_city is None:
            local_n_clients = max(1, min(5, round(n_samples / 50)))
        else:
            local_n_clients = requested_n_clients_per_city

        # Determine number of clusters for this city
        if clusters_per_city is None:
            # Auto: roughly 1 cluster per 30 samples, min 2, max 8
            n_clusters = max(2, min(8, n_samples // 30))
        else:
            n_clusters = min(clusters_per_city, n_samples)
        
        print(f"  Samples: {n_samples}")
        print(f"  Clusters: {n_clusters}")
        print(f"  Clients: {local_n_clients}")
        
        # Cluster this city's embeddings
        city_cluster_labels = cluster_city_embeddings(
            city_embs, 
            city_indices, 
            n_clusters,
            seed=seed + city_idx
        )
        
        # Partition across clients using Dirichlet
        partitioner = DirichletPartitioner(
            n_clients=local_n_clients,
            alpha=alpha,
            min_samples_per_client=min_samples_per_client,
            seed=seed + city_idx,
        )
        
        city_client_data = partitioner.partition(city_cluster_labels, city_indices)
        
        # Add to global client mapping
        for local_client_id in range(local_n_clients):
            client_indices = city_client_data[local_client_id]
            
            all_client_data[global_client_id] = client_indices
            all_client_paths[global_client_id] = [
                image_paths[idx] for idx in client_indices
            ]
            
            print(f"  Client {global_client_id} (City: {city}, Local: {local_client_id}): "
                  f"{len(client_indices)} samples")
            
            global_client_id += 1
        
        # Store city statistics
        city_statistics[city] = {
            "n_samples": n_samples,
            "n_clusters": n_clusters,
            "n_clients": local_n_clients,
            "cluster_distribution": {
                int(k): int(v) for k, v in 
                zip(*np.unique(city_cluster_labels, return_counts=True))
            },
        }
    
    # Compute overall statistics
    total_clients = len(all_client_data)
    total_samples = sum(len(v) for v in all_client_data.values())
    samples_per_client = {cid: len(indices) for cid, indices in all_client_data.items()}
    
    statistics = {
        "n_cities": n_cities,
        "n_clients_per_city": requested_n_clients_per_city,
        "total_clients": total_clients,
        "total_samples": total_samples,
        "alpha": alpha,
        "samples_per_client": samples_per_client,
        "min_samples": min(samples_per_client.values()),
        "max_samples": max(samples_per_client.values()),
        "mean_samples": np.mean(list(samples_per_client.values())),
        "std_samples": np.std(list(samples_per_client.values())),
        "city_statistics": city_statistics,
    }
    
    # Prepare output
    result = {
        "client_data": all_client_data,
        "client_paths": all_client_paths,
        "statistics": statistics,
        "config": {
            "n_cities": n_cities,
            "n_clients_per_city": requested_n_clients_per_city,
            "total_clients": total_clients,
            "alpha": alpha,
            "clusters_per_city": clusters_per_city,
            "seed": seed,
        },
    }
    
    # Save if requested
    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "client_splits_city_based.pth")
        print(f"\n{'='*60}")
        print(f"Saving to {output_path}")
        torch.save(result, output_path)
    
    print(f"\n{'='*60}")
    print("CITY-BASED PARTITIONING COMPLETE")
    print(f"{'='*60}")
    print(f"Total cities: {n_cities}")
    clients_per_city_display = (
        str(requested_n_clients_per_city)
        if requested_n_clients_per_city is not None
        else "adaptive"
    )
    print(f"Total clients: {total_clients} ({clients_per_city_display} per city)")
    print(f"Total samples: {total_samples}")
    print(f"Samples per client: {statistics['min_samples']} - {statistics['max_samples']} "
          f"(mean: {statistics['mean_samples']:.1f})")
    print(f"{'='*60}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="City-based partitioning with per-city clustering"
    )
    parser.add_argument("--embeddings_path", type=str, required=True,
                        help="Path to embeddings.pth file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--n_clients_per_city", type=int, default=None,
                        help="Number of clients per city")
    parser.add_argument("--clusters_per_city", type=int, default=None,
                        help="Number of clusters per city (auto if not specified)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha parameter")
    parser.add_argument("--min_samples_per_client", type=int, default=5,
                        help="Minimum samples per client")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    partition_city_based(
        embeddings_path=args.embeddings_path,
        n_clients_per_city=args.n_clients_per_city,
        clusters_per_city=args.clusters_per_city,
        alpha=args.alpha,
        min_samples_per_client=args.min_samples_per_client,
        seed=args.seed,
        output_dir=args.output_dir,
        save=True,
    )
