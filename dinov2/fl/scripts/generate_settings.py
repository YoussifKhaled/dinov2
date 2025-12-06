#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Generate all three FL partition settings.

This script orchestrates the complete pipeline to generate:
1. IID Setting: Large alpha (α=100) for near-IID distribution
2. Non-IID Setting: Small alpha (α=0.1) for extreme heterogeneity
3. City-Based Non-IID: Per-city clustering with 5 clients each

All outputs are saved in both .pth and JSON formats.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dinov2.fl.config import FLConfig
from dinov2.fl.clustering.kmeans import cluster_embeddings
from dinov2.fl.partitioning.dirichlet import partition_data
from dinov2.fl.partitioning.city_based import partition_city_based
from dinov2.fl.partitioning.export_partitions import export_partition_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all three FL partition settings"
    )
    
    # Input paths
    parser.add_argument(
        "--embeddings_path", type=str, required=True,
        help="Path to embeddings.pth file from extraction phase"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for all settings"
    )
    
    # Global settings
    parser.add_argument(
        "--n_clusters", type=int, default=16,
        help="Number of clusters for global clustering (Settings 1 & 2)"
    )
    parser.add_argument(
        "--n_clients", type=int, default=10,
        help="Number of clients for Settings 1 & 2"
    )
    parser.add_argument(
        "--n_clients_per_city", type=int, default=5,
        help="Number of clients per city for Setting 3"
    )
    parser.add_argument(
        "--clusters_per_city", type=int, default=None,
        help="Clusters per city for Setting 3 (auto if not specified)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--base_path", type=str, default="/kaggle/input/cityscapes-fine-dataset",
        help="Base path to strip from output JSON paths"
    )
    
    # Alpha values
    parser.add_argument(
        "--alpha_iid", type=float, default=100.0,
        help="Alpha value for IID setting (large = more IID)"
    )
    parser.add_argument(
        "--alpha_noniid", type=float, default=0.1,
        help="Alpha value for non-IID setting (small = more heterogeneous)"
    )
    parser.add_argument(
        "--alpha_city", type=float, default=0.5,
        help="Alpha value for city-based setting"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("FL PARTITION SETTINGS GENERATOR")
    print("=" * 80)
    print(f"Input: {args.embeddings_path}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    
    # ========================================================================
    # PHASE 1: Global Clustering (for Settings 1 & 2)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Global Clustering")
    print("=" * 80)
    
    clusters_path = f"{args.output_dir}/clusters_global.pth"
    
    # Create config for clustering
    cluster_config = FLConfig(
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )
    
    cluster_result = cluster_embeddings(
        config=cluster_config,
        embeddings_path=args.embeddings_path,
        save=True,
    )
    
    # Rename to avoid overwriting
    import os
    if os.path.exists(f"{args.output_dir}/clusters.pth"):
        os.rename(f"{args.output_dir}/clusters.pth", clusters_path)
    
    print(f"✓ Global clustering complete: {args.n_clusters} clusters")
    
    # ========================================================================
    # PHASE 2: Setting 1 - IID (α = 100)
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"PHASE 2: Setting 1 - IID (α = {args.alpha_iid})")
    print("=" * 80)
    
    config_iid = FLConfig(
        output_dir=args.output_dir,
        n_clients=args.n_clients,
        alpha=args.alpha_iid,
        seed=args.seed,
    )
    
    result_iid = partition_data(
        config=config_iid,
        clusters_path=clusters_path,
        save=False,  # We'll save with custom name
    )
    
    # Save .pth file
    import torch
    splits_iid_path = f"{args.output_dir}/client_splits_iid.pth"
    torch.save(result_iid, splits_iid_path)
    print(f"✓ Saved: {splits_iid_path}")
    
    # Export to JSON
    json_iid_path = f"{args.output_dir}/setting1_iid.json"
    export_partition_to_json(
        splits_path=splits_iid_path,
        output_path=json_iid_path,
        client_naming="client",
        base_path_to_strip=args.base_path,
    )
    
    # ========================================================================
    # PHASE 3: Setting 2 - Non-IID (α = 0.1)
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"PHASE 3: Setting 2 - Non-IID (α = {args.alpha_noniid})")
    print("=" * 80)
    
    config_noniid = FLConfig(
        output_dir=args.output_dir,
        n_clients=args.n_clients,
        alpha=args.alpha_noniid,
        seed=args.seed,
    )
    
    result_noniid = partition_data(
        config=config_noniid,
        clusters_path=clusters_path,
        save=False,
    )
    
    # Save .pth file
    splits_noniid_path = f"{args.output_dir}/client_splits_noniid.pth"
    torch.save(result_noniid, splits_noniid_path)
    print(f"✓ Saved: {splits_noniid_path}")
    
    # Export to JSON
    json_noniid_path = f"{args.output_dir}/setting2_noniid.json"
    export_partition_to_json(
        splits_path=splits_noniid_path,
        output_path=json_noniid_path,
        client_naming="client",
        base_path_to_strip=args.base_path,
    )
    
    # ========================================================================
    # PHASE 4: Setting 3 - City-Based Non-IID
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"PHASE 4: Setting 3 - City-Based Non-IID (α = {args.alpha_city})")
    print("=" * 80)
    
    result_city = partition_city_based(
        embeddings_path=args.embeddings_path,
        n_clients_per_city=args.n_clients_per_city,
        clusters_per_city=args.clusters_per_city,
        alpha=args.alpha_city,
        seed=args.seed,
        output_dir=args.output_dir,
        save=False,
    )
    
    # Save .pth file
    splits_city_path = f"{args.output_dir}/client_splits_city_based.pth"
    torch.save(result_city, splits_city_path)
    print(f"✓ Saved: {splits_city_path}")
    
    # Export to JSON with city naming
    json_city_path = f"{args.output_dir}/setting3_city_based.json"
    export_partition_to_json(
        splits_path=splits_city_path,
        output_path=json_city_path,
        client_naming="numeric",  # Use numeric IDs for city-based
        base_path_to_strip=args.base_path,
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ ALL SETTINGS GENERATED SUCCESSFULLY")
    print("=" * 80)
    
    print("\n📁 PTH FILES (for programmatic use):")
    print(f"  • {splits_iid_path}")
    print(f"  • {splits_noniid_path}")
    print(f"  • {splits_city_path}")
    
    print("\n📄 JSON FILES (for FL training):")
    print(f"  • {json_iid_path}")
    print(f"  • {json_noniid_path}")
    print(f"  • {json_city_path}")
    
    print("\n📊 STATISTICS:")
    
    print(f"\n  Setting 1 - IID (α={args.alpha_iid}):")
    print(f"    Clients: {result_iid['config']['n_clients']}")
    print(f"    Samples: {result_iid['statistics']['total_samples']}")
    print(f"    Range: {result_iid['statistics']['min_samples']}-"
          f"{result_iid['statistics']['max_samples']} per client")
    
    print(f"\n  Setting 2 - Non-IID (α={args.alpha_noniid}):")
    print(f"    Clients: {result_noniid['config']['n_clients']}")
    print(f"    Samples: {result_noniid['statistics']['total_samples']}")
    print(f"    Range: {result_noniid['statistics']['min_samples']}-"
          f"{result_noniid['statistics']['max_samples']} per client")
    
    print(f"\n  Setting 3 - City-Based (α={args.alpha_city}):")
    print(f"    Cities: {result_city['config']['n_cities']}")
    print(f"    Total Clients: {result_city['config']['total_clients']} "
          f"({args.n_clients_per_city} per city)")
    print(f"    Samples: {result_city['statistics']['total_samples']}")
    print(f"    Range: {result_city['statistics']['min_samples']}-"
          f"{result_city['statistics']['max_samples']} per client")
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE - Ready for FL Training!")
    print("=" * 80)


if __name__ == "__main__":
    main()
