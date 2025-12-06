# FL Partition Settings Generator

## Overview

This module generates **three partition settings** for Federated Learning experiments on Cityscapes, each saved in the `city_partitions.json` format.

## Three Settings

### Setting 1: IID (Near-IID Distribution)
- **Strategy**: Dirichlet distribution with α=100 (large α → near-IID)
- **Clients**: 10
- **Characteristics**: Balanced, diverse data across all clients
- **Use Case**: Baseline for FL performance comparison

### Setting 2: Non-IID (Extreme Heterogeneity)
- **Strategy**: Dirichlet distribution with α=0.1 (small α → highly non-IID)
- **Clients**: 10
- **Characteristics**: Specialized, imbalanced data per client
- **Use Case**: Realistic FL scenario with strong heterogeneity

### Setting 3: City-Based Non-IID (Geographic + Semantic)
- **Strategy**: Per-city clustering + Dirichlet distribution (α=0.5)
- **Clients**: 90 (5 per city × 18 cities)
- **Characteristics**: Geographic isolation + scene-type specialization
- **Use Case**: Multi-level heterogeneity (location + semantics)

## Output Format

All settings are exported to JSON files matching the `city_partitions.json` format:

```json
{
    "0": {
        "client_name": "client_0",
        "num_samples": 297,
        "data": [
            ["leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png",
             "gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png"],
            ...
        ]
    },
    "1": { ... },
    ...
}
```

## Usage

### Option 1: Kaggle Notebook (Recommended)

1. Upload `fl_partition_settings_generator.ipynb` to Kaggle
2. Add Cityscapes dataset
3. Enable GPU + Internet
4. Run all cells
5. Download `fl_settings_complete.zip`

**Runtime**: ~20 minutes total
- Embedding extraction: ~15 min
- Settings generation: ~3 min
- Export & visualization: ~2 min

### Option 2: Command Line

```bash
# Step 1: Extract embeddings
python -m dinov2.fl.scripts.run_extraction \
    --dataset_list_file train_fine.txt \
    --base_path /path/to/cityscapes \
    --output_dir ./fl_outputs \
    --model_name dinov2_vitl14 \
    --batch_size 16

# Step 2: Generate all settings
python -m dinov2.fl.scripts.generate_settings \
    --embeddings_path ./fl_outputs/embeddings.pth \
    --output_dir ./fl_outputs \
    --n_clusters 16 \
    --n_clients 10 \
    --n_clients_per_city 5 \
    --alpha_iid 100.0 \
    --alpha_noniid 0.1 \
    --alpha_city 0.5 \
    --base_path /path/to/cityscapes \
    --seed 42
```

## Files Generated

### JSON Files (for FL training)
- `setting1_iid.json` - IID baseline
- `setting2_noniid.json` - Non-IID heterogeneity
- `setting3_city_based.json` - City-based partitioning

### PTH Files (for analysis)
- `client_splits_iid.pth`
- `client_splits_noniid.pth`
- `client_splits_city_based.pth`
- `embeddings.pth`
- `clusters_global.pth`

## New Modules

### `dinov2/fl/partitioning/export_partitions.py`
Converts `.pth` partition files to JSON format.

**Functions**:
- `export_partition_to_json()` - Export single partition
- `export_multiple_partitions()` - Batch export
- `extract_city_from_path()` - Parse city names
- `get_label_path_from_image()` - Map images to labels

### `dinov2/fl/partitioning/city_based.py`
Implements city-based partitioning strategy.

**Functions**:
- `partition_city_based()` - Main partitioning logic
- `cluster_city_embeddings()` - Per-city clustering
- `extract_city_from_path()` - City name extraction

### `dinov2/fl/scripts/generate_settings.py`
Orchestrates the complete pipeline.

**Workflow**:
1. Global clustering (for Settings 1 & 2)
2. Generate Setting 1 (IID)
3. Generate Setting 2 (Non-IID)
4. Generate Setting 3 (City-Based)
5. Export all to JSON
6. Print statistics

## Parameters

### Global Settings
- `--n_clusters`: Number of clusters for global clustering (default: 16)
- `--n_clients`: Number of clients for Settings 1 & 2 (default: 10)
- `--seed`: Random seed (default: 42)

### Alpha Values (Dirichlet Concentration)
- `--alpha_iid`: Alpha for Setting 1 (default: 100.0)
- `--alpha_noniid`: Alpha for Setting 2 (default: 0.1)
- `--alpha_city`: Alpha for Setting 3 (default: 0.5)

### City-Based Settings
- `--n_clients_per_city`: Clients per city (default: 5)
- `--clusters_per_city`: Clusters per city (auto if not specified)

## Statistics

After generation, you'll see:

```
Setting 1 - IID (α=100):
  Clients: 10
  Samples: 2975
  Range: 285-310 per client
  Interpretation: Balanced, near-IID

Setting 2 - Non-IID (α=0.1):
  Clients: 10
  Samples: 2975
  Range: 45-680 per client
  Interpretation: Highly imbalanced, specialized

Setting 3 - City-Based (α=0.5):
  Cities: 18
  Total Clients: 90 (5 per city)
  Samples: 2975
  Range: 10-75 per client
  Interpretation: Geographic isolation + semantic specialization
```

## Integration with Existing Code

The new modules integrate seamlessly with the existing FL pipeline:

```python
# Existing imports work
from dinov2.fl.partitioning import partition_data, load_client_splits

# New imports available
from dinov2.fl.partitioning import (
    partition_city_based,
    export_partition_to_json,
    export_multiple_partitions
)
```

## Validation

Each JSON file is validated to ensure:
- ✓ Correct format matching `city_partitions.json`
- ✓ All clients have valid data entries
- ✓ Image and label paths are relative (not absolute)
- ✓ Paths use correct structure (`leftImg8bit/...`, `gtFine/...`)
- ✓ `client_name`, `num_samples`, `data` fields present

## Troubleshooting

### Dataset Path Issues
If extraction fails with file not found:
1. Check `BASE_PATH` in notebook matches your dataset location
2. Verify dataset structure has `leftImg8bit/` and `gtFine/` folders
3. Ensure `train_fine.txt` paths match your setup

### Memory Issues
If clustering runs out of memory:
- Reduce `batch_size` in extraction (default: 16 → try 8)
- Use smaller `n_clusters` (default: 16 → try 8)

### Import Errors
If modules not found after restart:
```python
import sys
sys.path.insert(0, '/kaggle/working/dinov2')
```

## Citation

If you use these partition settings in your research:

```bibtex
@misc{dinov2_fl_partitions,
  title={Semantic-Aware Data Partitioning for Federated Learning},
  author={DINOv2 FL Team},
  year={2024},
  howpublished={\url{https://github.com/YoussifKhaled/dinov2}}
}
```
