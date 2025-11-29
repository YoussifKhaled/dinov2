# FL Data Heterogeneity Pipeline using DINOv2

This module implements **Embedding-Based Data Heterogeneity** for Federated Learning (FL) on semantic segmentation tasks, following the methodology from *"Redefining non-IID Data in Federated Learning for Computer Vision Tasks"* (Borazjani et al.).

## Overview

Standard non-IID partitioning based on class labels is **ineffective for semantic segmentation** where each image contains multiple classes. This pipeline solves that by:

1. **Extracting scene embeddings** using DINOv2's [CLS] token
2. **Clustering scenes** using K-Means to discover latent visual categories
3. **Partitioning data** using Dirichlet distribution over clusters

---

## Quick Start on Kaggle

### Step 1: Setup Environment

Run these cells at the start of your Kaggle notebook:

```python
# Cell 1: Install dependencies (run once)
!pip install -q omegaconf fvcore xformers

# Cell 2: Clone/setup the repo (if not already present)
# Skip if you uploaded the dinov2 folder directly
!git clone https://github.com/YoussifKhaled/dinov2.git
%cd dinov2
```

### Step 2: Prepare Dataset Path Mapping

```python
# Cell 3: Configure paths
# Update BASE_PATH to match your Kaggle dataset location
BASE_PATH = "/kaggle/input/cityscapes"  # Adjust to your dataset path
DATASET_LIST = "train_fine.txt"
OUTPUT_DIR = "/kaggle/working/fl_outputs"
```

### Step 3: Run the Pipeline

**Option A: Run Full Pipeline (Recommended)**

```python
# Cell 4: Run complete pipeline
!python -m dinov2.fl.scripts.run_pipeline \
    --dataset_list_file {DATASET_LIST} \
    --base_path {BASE_PATH} \
    --output_dir {OUTPUT_DIR} \
    --model_name dinov2_vitl14 \
    --batch_size 16 \
    --n_clusters 16 \
    --n_clients 10 \
    --alpha 0.5
```

**Option B: Run Phases Separately**

```python
# Cell 4a: Phase 1 - Extract embeddings (~15-20 min for Cityscapes)
!python -m dinov2.fl.scripts.run_extraction \
    --dataset_list_file {DATASET_LIST} \
    --base_path {BASE_PATH} \
    --output_dir {OUTPUT_DIR} \
    --model_name dinov2_vitl14 \
    --batch_size 16

# Cell 4b: Phase 2 - Cluster embeddings (~1 min)
!python -m dinov2.fl.scripts.run_clustering \
    --output_dir {OUTPUT_DIR} \
    --n_clusters 16

# Cell 4c: Phase 3 - Partition data (~seconds)
!python -m dinov2.fl.scripts.run_partitioning \
    --output_dir {OUTPUT_DIR} \
    --n_clients 10 \
    --alpha 0.5
```

### Step 4: Load and Use the Splits

```python
# Cell 5: Load client splits for FL training
import torch

splits = torch.load(f"{OUTPUT_DIR}/client_splits.pth")

# Access client data
for client_id in range(10):
    indices = splits["client_data"][client_id]
    paths = splits["client_paths"][client_id]
    print(f"Client {client_id}: {len(indices)} samples")

# Get statistics
stats = splits["statistics"]
print(f"\nPartition stats:")
print(f"  Alpha: {splits['config']['alpha']}")
print(f"  Samples range: {stats['min_samples']} - {stats['max_samples']}")
```

---

## Command Reference

### Full Pipeline

```bash
python -m dinov2.fl.scripts.run_pipeline \
    --dataset_list_file <path_to_txt>     # train_fine.txt
    --base_path <kaggle_data_path>        # /kaggle/input/cityscapes
    --output_dir <output_path>            # ./fl_outputs
    --model_name <model>                  # dinov2_vitl14 (default)
    --batch_size <int>                    # 32 (default), use 16 for T4 GPU
    --n_clusters <int>                    # 16 (default)
    --n_clients <int>                     # 10 (default)
    --alpha <float>                       # 0.5 (default)
    --seed <int>                          # 42 (default)
```

### Individual Phases

```bash
# Phase 1: Extract embeddings
python -m dinov2.fl.scripts.run_extraction \
    --dataset_list_file train_fine.txt \
    --base_path /kaggle/input/cityscapes \
    --output_dir ./fl_outputs \
    --model_name dinov2_vitl14 \
    --batch_size 16

# Phase 2: Cluster (requires embeddings.pth)
python -m dinov2.fl.scripts.run_clustering \
    --output_dir ./fl_outputs \
    --n_clusters 16

# Phase 3: Partition (requires clusters.pth)
python -m dinov2.fl.scripts.run_partitioning \
    --output_dir ./fl_outputs \
    --n_clients 10 \
    --alpha 0.5
```

---

## Output Files

| File | Description |
|------|-------------|
| `embeddings.pth` | DINOv2 [CLS] token embeddings for all images |
| `clusters.pth` | K-Means cluster assignments and centroids |
| `client_splits.pth` | Final FL client data partitions |

### `client_splits.pth` Structure

```python
{
    "client_data": {
        0: [idx1, idx2, ...],  # Sample indices for client 0
        1: [idx3, idx4, ...],  # Sample indices for client 1
        ...
    },
    "client_paths": {
        0: ["/path/to/img1.png", ...],  # Image paths for client 0
        ...
    },
    "statistics": {
        "n_clients": 10,
        "alpha": 0.5,
        "samples_per_client": {0: 297, 1: 312, ...},
        "clusters_per_client": {0: 8, 1: 12, ...},
        ...
    },
    "config": {
        "n_clients": 10,
        "alpha": 0.5,
        "n_clusters": 16,
        "seed": 42
    }
}
```

---

## Key Parameters

### Alpha (Dirichlet Concentration)

Controls the degree of data heterogeneity:

| Alpha | Heterogeneity | Description |
|-------|---------------|-------------|
| 0.1   | Extreme       | Clients receive data from 1-2 clusters only |
| 0.5   | High          | Significant cluster imbalance across clients |
| 1.0   | Moderate      | Noticeable but balanced non-IID |
| 10.0  | Low           | Nearly uniform distribution |
| 100.0 | IID           | Approximately uniform (baseline) |

### Number of Clusters (K)

Recommended values for Cityscapes:
- **K=8**: Coarse scene categories
- **K=16**: Balanced granularity (recommended)
- **K=32**: Fine-grained scene types

---

## Python API Usage

```python
from dinov2.fl import FLConfig, extract_embeddings, cluster_embeddings, partition_data

# Create configuration
config = FLConfig(
    dataset_list_file="train_fine.txt",
    base_path="/kaggle/input/cityscapes",
    output_dir="./fl_outputs",
    model_name="dinov2_vitl14",
    batch_size=16,
    n_clusters=16,
    n_clients=10,
    alpha=0.5,
)

# Run pipeline
extract_embeddings(config)
cluster_embeddings(config)
result = partition_data(config)

# Use results
client_data = result["client_data"]
for client_id, indices in client_data.items():
    print(f"Client {client_id}: {len(indices)} samples")
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
--batch_size 8  # or even 4 for very limited GPU memory
```

### Path Errors

Ensure `base_path` correctly maps to your Kaggle dataset structure:
```
/kaggle/input/cityscapes/
├── leftImg8bit/
│   └── train/
│       ├── aachen/
│       └── ...
└── gtFine/
    └── train/
        ├── aachen/
        └── ...
```

### Missing Dependencies

```bash
pip install omegaconf fvcore xformers scikit-learn tqdm pillow
```

---

## Citation

If using this pipeline, please cite:

```bibtex
@article{borazjani2023redefining,
  title={Redefining non-IID Data in Federated Learning for Computer Vision Tasks},
  author={Borazjani, et al.},
  year={2023}
}

@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```
