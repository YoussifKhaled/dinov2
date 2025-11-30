# FL Data Heterogeneity Pipeline

Embedding-based non-IID data partitioning for Federated Learning using DINOv2.

---

## Section 1: Kaggle Notebook Setup

Follow these steps **exactly in order** to prepare your Kaggle notebook.

### Prerequisites

1. Create a new Kaggle notebook
2. Enable **GPU** (Settings → Accelerator → GPU T4 x2 or P100)
3. Enable **Internet** (Settings → Internet → On)
4. Add your **Cityscapes dataset** (+ Add Data → search "cityscapes")

---

### Cell 1: Clone Repository

```python
!git clone https://github.com/YoussifKhaled/dinov2.git
%cd dinov2
```

---

### Cell 2: Fix PyTorch & Install Dependencies

```python
# Uninstall Kaggle's broken PyTorch
!pip uninstall -y torch torchvision torchaudio

# Install from our requirements.txt (has correct versions)
!pip install -r dinov2/fl/requirements.txt
```

---

### Cell 3: Restart Kernel

**⚠️ CRITICAL: You MUST restart the kernel now.**

- Click **Runtime → Restart session** (or the restart button)
- After restart, **run Cell 4 first** (skip Cells 1-3)

---

### Cell 4: Verify Setup (Run After Restart)

```python
%cd /kaggle/working/dinov2

import torch
import torchvision
import os

print("=" * 50)
print("ENVIRONMENT CHECK")
print("=" * 50)
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check dataset
BASE_PATH = "/kaggle/input/cityscapes-fine-dataset"  # ← CHANGE IF DIFFERENT
print(f"\nDataset path: {BASE_PATH}")
print(f"Dataset exists: {os.path.exists(BASE_PATH)}")
if os.path.exists(BASE_PATH):
    print(f"Contents: {os.listdir(BASE_PATH)}")
else:
    print("ERROR: Dataset not found! Check your data path.")
    print("Go to: + Add Data → find your Cityscapes dataset")
    print("Then update BASE_PATH above")
print("=" * 50)
```

**Expected output:**
```
PyTorch: 2.1.0+cu118
Torchvision: 0.16.0+cu118
CUDA available: True
GPU: Tesla T4
Dataset exists: True
Contents: ['leftImg8bit', 'gtFine']
```

If dataset not found, update `BASE_PATH` to match your actual dataset path from Kaggle's Data tab.

---

## Section 2: Run Pipeline

### Option A: Full Pipeline (Recommended)

Runs all 3 phases sequentially.

```python
!python -m dinov2.fl.scripts.run_pipeline \
    --dataset_list_file train_fine.txt \
    --base_path /kaggle/input/cityscapes-fine-dataset \
    --output_dir /kaggle/working/fl_outputs \
    --model_name dinov2_vitl14 \
    --batch_size 16 \
    --n_clusters 16 \
    --n_clients 10 \
    --alpha 0.5
```

---

### Option B: Run Phases Separately

Useful if you want to resume from a checkpoint or experiment with different parameters.

**Phase 1: Extract Embeddings** (~15-20 min on T4)

```python
!python -m dinov2.fl.scripts.run_extraction \
    --dataset_list_file train_fine.txt \
    --base_path /kaggle/input/cityscapes-fine-dataset \
    --output_dir /kaggle/working/fl_outputs \
    --model_name dinov2_vitl14 \
    --batch_size 16
```

**Phase 2: Cluster Embeddings** (~1 min)

```python
!python -m dinov2.fl.scripts.run_clustering \
    --output_dir /kaggle/working/fl_outputs \
    --n_clusters 16
```

**Phase 3: Partition Data** (~seconds)

```python
!python -m dinov2.fl.scripts.run_partitioning \
    --output_dir /kaggle/working/fl_outputs \
    --n_clients 10 \
    --alpha 0.5
```

---

### Load & Use Results

```python
import torch

splits = torch.load("/kaggle/working/fl_outputs/client_splits.pth")

print(f"Total clients: {len(splits['client_data'])}")
print(f"Alpha (heterogeneity): {splits['config']['alpha']}")
print()
for cid in range(len(splits['client_data'])):
    n = len(splits['client_data'][cid])
    print(f"Client {cid}: {n} samples")
```

---

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_path` | - | Path to Cityscapes dataset on Kaggle |
| `--output_dir` | - | Where to save outputs |
| `--alpha` | 0.5 | **Heterogeneity control**: 0.1=extreme non-IID, 1.0=moderate, 100=IID |
| `--n_clusters` | 16 | Number of scene clusters (K-Means) |
| `--n_clients` | 10 | Number of FL clients to partition data into |
| `--batch_size` | 32 | Batch size for embedding extraction (use 16 for T4 GPU) |
| `--model_name` | dinov2_vitl14 | Model variant: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14` |

---

## Output Files

All outputs saved to `--output_dir`:

| File | Description |
|------|-------------|
| `embeddings.pth` | DINOv2 CLS embeddings (N × 1024) |
| `clusters.pth` | K-Means cluster assignments |
| `client_splits.pth` | Final client partitions with paths and statistics |
