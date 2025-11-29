#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Phase 2: Cluster embeddings using K-Means.

Usage:
    python -m dinov2.fl.scripts.run_clustering \
        --output_dir ./fl_outputs \
        --n_clusters 16

Input:
    embeddings.pth from Phase 1

Output:
    clusters.pth containing:
        - cluster_labels: Array of cluster assignments
        - centroids: Cluster centers
        - cluster_sizes: Distribution over clusters
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dinov2.fl.config import FLConfig
from dinov2.fl.clustering.kmeans import cluster_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster DINOv2 embeddings using K-Means"
    )
    
    # Add common args
    FLConfig.add_common_args(parser)
    
    # Clustering-specific args
    parser.add_argument(
        "--n_clusters", type=int, default=16,
        help="Number of K-Means clusters (scene categories)"
    )
    parser.add_argument(
        "--embeddings_path", type=str, default=None,
        help="Path to embeddings.pth (default: <output_dir>/embeddings.pth)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config from args
    config = FLConfig(
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )
    
    # Override embeddings path if provided
    embeddings_path = args.embeddings_path or config.embeddings_path
    
    print("=" * 60)
    print("Phase 2: K-Means Clustering")
    print("=" * 60)
    print(f"Embeddings: {embeddings_path}")
    print(f"Number of clusters: {config.n_clusters}")
    print(f"Output: {config.clusters_path}")
    print("=" * 60)
    
    # Run clustering
    result = cluster_embeddings(config, embeddings_path=embeddings_path, save=True)
    
    print("\n" + "=" * 60)
    print("Clustering complete!")
    print(f"Clustered {result['n_samples']} samples into {result['n_clusters']} clusters")
    print(f"Cluster sizes: {result['cluster_sizes']}")
    print(f"Saved to: {config.clusters_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
