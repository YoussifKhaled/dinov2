#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Phase 1: Extract DINOv2 embeddings from Cityscapes images.

Usage:
    python -m dinov2.fl.scripts.run_extraction \
        --dataset_list_file train_fine.txt \
        --base_path /kaggle/input/cityscapes \
        --output_dir ./fl_outputs \
        --batch_size 32

Output:
    embeddings.pth containing:
        - embeddings: Tensor [N, embed_dim]
        - indices: List of sample indices
        - image_paths: List of image file paths
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dinov2.fl.config import FLConfig
from dinov2.fl.embedding.extractor import extract_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 embeddings from Cityscapes images"
    )
    
    # Add common args
    FLConfig.add_common_args(parser)
    
    # Extraction-specific args
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config from args
    config = FLConfig(
        dataset_list_file=args.dataset_list_file,
        base_path=args.base_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    
    print("=" * 60)
    print("Phase 1: DINOv2 Embedding Extraction")
    print("=" * 60)
    print(f"Dataset list: {config.dataset_list_file}")
    print(f"Base path: {config.base_path}")
    print(f"Model: {config.model_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Output: {config.embeddings_path}")
    print("=" * 60)
    
    # Run extraction
    result = extract_embeddings(config, save=True)
    
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print(f"Extracted {result['n_samples']} embeddings")
    print(f"Embedding dimension: {result['embed_dim']}")
    print(f"Saved to: {config.embeddings_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
