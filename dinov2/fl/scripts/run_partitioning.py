#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Phase 3: Partition data across FL clients using Dirichlet distribution.

Usage:
    python -m dinov2.fl.scripts.run_partitioning \
        --output_dir ./fl_outputs \
        --n_clients 10 \
        --alpha 0.5

Input:
    clusters.pth from Phase 2

Output:
    client_splits.pth containing:
        - client_data: Dict[client_id -> list of sample indices]
        - client_paths: Dict[client_id -> list of image paths]
        - statistics: Partition statistics
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dinov2.fl.config import FLConfig
from dinov2.fl.partitioning.dirichlet import partition_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition data across FL clients using Dirichlet distribution"
    )
    
    # Add common args
    FLConfig.add_common_args(parser)
    
    # Partitioning-specific args
    parser.add_argument(
        "--n_clients", type=int, default=10,
        help="Number of FL clients"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Dirichlet concentration parameter (lower = more non-IID)"
    )
    parser.add_argument(
        "--min_samples_per_client", type=int, default=10,
        help="Minimum samples each client must receive"
    )
    parser.add_argument(
        "--clusters_path", type=str, default=None,
        help="Path to clusters.pth (default: <output_dir>/clusters.pth)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config from args
    config = FLConfig(
        output_dir=args.output_dir,
        n_clients=args.n_clients,
        alpha=args.alpha,
        min_samples_per_client=args.min_samples_per_client,
        seed=args.seed,
    )
    
    # Override clusters path if provided
    clusters_path = args.clusters_path or config.clusters_path
    
    print("=" * 60)
    print("Phase 3: Dirichlet Data Partitioning")
    print("=" * 60)
    print(f"Clusters: {clusters_path}")
    print(f"Number of clients: {config.n_clients}")
    print(f"Alpha (heterogeneity): {config.alpha}")
    print(f"  - Low alpha (0.1): Highly non-IID")
    print(f"  - High alpha (100): Nearly IID")
    print(f"Output: {config.splits_path}")
    print("=" * 60)
    
    # Run partitioning
    result = partition_data(config, clusters_path=clusters_path, save=True)
    
    stats = result["statistics"]
    
    print("\n" + "=" * 60)
    print("Partitioning complete!")
    print(f"Created splits for {stats['n_clients']} clients")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples per client:")
    for client_id in sorted(stats["samples_per_client"].keys()):
        n_samples = stats["samples_per_client"][client_id]
        n_clusters = stats["clusters_per_client"][client_id]
        print(f"  Client {client_id}: {n_samples} samples from {n_clusters} clusters")
    print(f"Saved to: {config.splits_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
