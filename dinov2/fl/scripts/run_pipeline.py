#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Run the complete FL data heterogeneity pipeline.

This script runs all three phases sequentially:
    1. Extract DINOv2 embeddings
    2. Cluster embeddings with K-Means
    3. Partition data with Dirichlet distribution

Usage:
    python -m dinov2.fl.scripts.run_pipeline \
        --dataset_list_file train_fine.txt \
        --base_path /kaggle/input/cityscapes \
        --output_dir ./fl_outputs \
        --n_clusters 16 \
        --n_clients 10 \
        --alpha 0.5
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dinov2.fl.config import FLConfig
from dinov2.fl.embedding.extractor import extract_embeddings
from dinov2.fl.clustering.kmeans import cluster_embeddings
from dinov2.fl.partitioning.dirichlet import partition_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete FL data heterogeneity pipeline"
    )
    
    # Add common args
    FLConfig.add_common_args(parser)
    
    # Model args
    parser.add_argument(
        "--model_name", type=str, default="dinov2_vitl14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        help="DINOv2 model variant to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of DataLoader workers"
    )
    
    # Clustering args
    parser.add_argument(
        "--n_clusters", type=int, default=16,
        help="Number of K-Means clusters"
    )
    
    # Partitioning args
    parser.add_argument(
        "--n_clients", type=int, default=10,
        help="Number of FL clients"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Dirichlet concentration parameter"
    )
    parser.add_argument(
        "--min_samples_per_client", type=int, default=10,
        help="Minimum samples per client"
    )
    
    # Phase control
    parser.add_argument(
        "--skip_extraction", action="store_true",
        help="Skip Phase 1 (use existing embeddings.pth)"
    )
    parser.add_argument(
        "--skip_clustering", action="store_true",
        help="Skip Phase 2 (use existing clusters.pth)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    
    # Create config from args
    config = FLConfig(
        dataset_list_file=args.dataset_list_file,
        base_path=args.base_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_clusters=args.n_clusters,
        n_clients=args.n_clients,
        alpha=args.alpha,
        min_samples_per_client=args.min_samples_per_client,
        seed=args.seed,
    )
    
    print("=" * 70)
    print("FL Data Heterogeneity Pipeline (Embedding-Based)")
    print("=" * 70)
    print(f"Dataset: {config.dataset_list_file}")
    print(f"Model: {config.model_name}")
    print(f"Clusters: {config.n_clusters}")
    print(f"Clients: {config.n_clients}")
    print(f"Alpha: {config.alpha}")
    print(f"Output: {config.output_dir}")
    print("=" * 70)
    
    # Phase 1: Extract embeddings
    if not args.skip_extraction:
        print("\n" + "=" * 70)
        print("PHASE 1: Extracting DINOv2 Embeddings")
        print("=" * 70)
        phase1_start = time.time()
        extract_embeddings(config, save=True)
        print(f"Phase 1 completed in {time.time() - phase1_start:.1f}s")
    else:
        print("\nSkipping Phase 1 (--skip_extraction)")
    
    # Phase 2: Cluster embeddings
    if not args.skip_clustering:
        print("\n" + "=" * 70)
        print("PHASE 2: Clustering Embeddings")
        print("=" * 70)
        phase2_start = time.time()
        cluster_embeddings(config, save=True)
        print(f"Phase 2 completed in {time.time() - phase2_start:.1f}s")
    else:
        print("\nSkipping Phase 2 (--skip_clustering)")
    
    # Phase 3: Partition data
    print("\n" + "=" * 70)
    print("PHASE 3: Dirichlet Partitioning")
    print("=" * 70)
    phase3_start = time.time()
    result = partition_data(config, save=True)
    print(f"Phase 3 completed in {time.time() - phase3_start:.1f}s")
    
    # Final summary
    total_time = time.time() - start_time
    stats = result["statistics"]
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")
    print(f"\nOutputs:")
    print(f"  Embeddings: {config.embeddings_path}")
    print(f"  Clusters: {config.clusters_path}")
    print(f"  Client splits: {config.splits_path}")
    print(f"\nPartition summary:")
    print(f"  {stats['n_clients']} clients, {stats['total_samples']} total samples")
    print(f"  Samples per client: {stats['min_samples']}-{stats['max_samples']} "
          f"(mean: {stats['mean_samples']:.1f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
