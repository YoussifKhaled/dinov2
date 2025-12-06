# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Export FL client partitions to JSON format.

Converts .pth partition files to the city_partitions.json format:
{
    "client_id": {
        "client_name": str,
        "num_samples": int,
        "data": [
            [image_path, label_path],
            ...
        ]
    }
}
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch


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


def strip_base_path(path: str, base_path: str = "/kaggle/input/cityscapes-fine-dataset") -> str:
    """Remove base path prefix to get relative path.
    
    Args:
        path: Full path like /kaggle/input/cityscapes-fine-dataset/leftImg8bit/...
        base_path: Base path to remove
        
    Returns:
        Relative path like leftImg8bit/train/aachen/...
    """
    if path.startswith(base_path):
        relative = path[len(base_path):]
        return relative.lstrip("/")
    return path


def get_label_path_from_image(image_path: str) -> str:
    """Convert image path to corresponding label path.
    
    Example:
        leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        -> gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png
    """
    # Replace directory and filename components
    label_path = image_path.replace("leftImg8bit", "gtFine")
    label_path = label_path.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
    return label_path


def export_partition_to_json(
    splits_path: str,
    output_path: str,
    client_naming: str = "client",
    base_path_to_strip: Optional[str] = "/kaggle/input/cityscapes-fine-dataset",
) -> Dict:
    """Export partition .pth file to JSON format.
    
    Args:
        splits_path: Path to client_splits.pth file
        output_path: Path for output JSON file
        client_naming: Naming scheme for clients:
            - "client": client_0, client_1, ...
            - "city": aachen, bochum, ... (extracts from paths)
            - "numeric": 0, 1, 2, ...
        base_path_to_strip: Base path to remove from image paths
        
    Returns:
        Dictionary in city_partitions.json format
    """
    # Load partition data
    splits_data = torch.load(splits_path)
    client_paths = splits_data["client_paths"]  # Dict[int, List[str]]
    n_clients = len(client_paths)
    
    print(f"Exporting {n_clients} clients from {splits_path}")
    
    # Build output structure
    output_data = {}
    
    for client_id in sorted(client_paths.keys()):
        image_paths = client_paths[client_id]
        
        # Determine client name
        if client_naming == "city":
            # Use most common city in this client's data
            cities = [extract_city_from_path(p) for p in image_paths]
            client_name = max(set(cities), key=cities.count)
        elif client_naming == "numeric":
            client_name = str(client_id)
        else:  # "client"
            client_name = f"client_{client_id}"
        
        # Build data array with [image_path, label_path] pairs
        data_pairs = []
        for img_path in image_paths:
            # Strip base path to get relative path
            if base_path_to_strip:
                img_relative = strip_base_path(img_path, base_path_to_strip)
            else:
                img_relative = img_path
            
            # Get corresponding label path
            label_relative = get_label_path_from_image(img_relative)
            
            data_pairs.append([img_relative, label_relative])
        
        # Add client entry
        output_data[str(client_id)] = {
            "client_name": client_name,
            "num_samples": len(data_pairs),
            "data": data_pairs
        }
        
        print(f"  Client {client_id} ({client_name}): {len(data_pairs)} samples")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\n✓ Saved to {output_path}")
    print(f"  Total clients: {n_clients}")
    print(f"  Total samples: {sum(d['num_samples'] for d in output_data.values())}")
    
    return output_data


def export_multiple_partitions(
    splits_paths: List[str],
    output_paths: List[str],
    names: List[str],
    client_naming: str = "client",
    base_path_to_strip: Optional[str] = "/kaggle/input/cityscapes-fine-dataset",
) -> List[Dict]:
    """Export multiple partition files to JSON format.
    
    Args:
        splits_paths: List of paths to .pth files
        output_paths: List of output JSON paths
        names: List of setting names for logging
        client_naming: Naming scheme for clients
        base_path_to_strip: Base path to remove from paths
        
    Returns:
        List of output dictionaries
    """
    results = []
    
    for splits_path, output_path, name in zip(splits_paths, output_paths, names):
        print("=" * 70)
        print(f"EXPORTING: {name}")
        print("=" * 70)
        
        result = export_partition_to_json(
            splits_path=splits_path,
            output_path=output_path,
            client_naming=client_naming,
            base_path_to_strip=base_path_to_strip,
        )
        results.append(result)
        print()
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Export partition to JSON format")
    parser.add_argument("--splits_path", type=str, required=True,
                        help="Path to client_splits.pth file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path for output JSON file")
    parser.add_argument("--client_naming", type=str, default="client",
                        choices=["client", "city", "numeric"],
                        help="Naming scheme for clients")
    parser.add_argument("--base_path", type=str, 
                        default="/kaggle/input/cityscapes-fine-dataset",
                        help="Base path to strip from image paths")
    
    args = parser.parse_args()
    
    export_partition_to_json(
        splits_path=args.splits_path,
        output_path=args.output_path,
        client_naming=args.client_naming,
        base_path_to_strip=args.base_path,
    )
