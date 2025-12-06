# Quick Start Guide - FL Partition Settings

## 🚀 Kaggle Notebook (Easiest Method)

### Step 1: Setup
1. Go to Kaggle → New Notebook
2. Upload `fl_partition_settings_generator.ipynb`
3. Settings → Accelerator → **GPU T4**
4. Settings → Internet → **ON**
5. + Add Data → Search **"cityscapes"** → Add dataset

### Step 2: Run
```
Run All Cells (Ctrl+Enter through notebook)
```

**Runtime**: ~20 minutes total

### Step 3: Download
```
Output → fl_settings_complete.zip → Download
```

### Step 4: Extract & Use
```
Unzip → Use the 3 JSON files:
├── setting1_iid.json          ← IID baseline (10 clients)
├── setting2_noniid.json        ← Non-IID challenge (10 clients)
└── setting3_city_based.json    ← City-based (90 clients)
```

---

## 💻 Command Line (Advanced)

### Prerequisites
```bash
git clone https://github.com/YoussifKhaled/dinov2.git
cd dinov2
pip install -r dinov2/fl/requirements.txt
```

### One-Line Execution
```bash
# Step 1: Extract embeddings (15 min)
python -m dinov2.fl.scripts.run_extraction \
    --dataset_list_file train_fine.txt \
    --base_path /path/to/cityscapes \
    --output_dir ./fl_outputs \
    --model_name dinov2_vitl14 \
    --batch_size 16

# Step 2: Generate all settings (3 min)
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

### Output Files
```
fl_outputs/
├── setting1_iid.json          ✅ Use this
├── setting2_noniid.json        ✅ Use this
├── setting3_city_based.json    ✅ Use this
├── embeddings.pth
├── clusters_global.pth
└── client_splits_*.pth
```

---

## 📊 What Each Setting Gives You

### Setting 1: IID (Baseline)
```json
{
  "0": {"client_name": "client_0", "num_samples": 297, "data": [...]},
  "1": {"client_name": "client_1", "num_samples": 298, "data": [...]},
  ...
  "9": {"client_name": "client_9", "num_samples": 296, "data": [...]}
}
```
- **10 clients** with ~300 samples each
- **Balanced** distribution (285-310 per client)
- **Diverse** data (14-16 clusters per client)
- 🎯 **Use for**: FL baseline experiments

---

### Setting 2: Non-IID (Challenge)
```json
{
  "0": {"client_name": "client_0", "num_samples": 45, "data": [...]},
  "1": {"client_name": "client_1", "num_samples": 680, "data": [...]},
  ...
  "9": {"client_name": "client_9", "num_samples": 123, "data": [...]}
}
```
- **10 clients** with varying samples
- **Imbalanced** distribution (45-680 per client)
- **Specialized** data (2-5 clusters per client)
- 🎯 **Use for**: Heterogeneity challenges

---

### Setting 3: City-Based (Multi-Level)
```json
{
  "0": {"client_name": "0", "num_samples": 35, "data": [aachen...]},
  "1": {"client_name": "1", "num_samples": 32, "data": [aachen...]},
  ...
  "5": {"client_name": "5", "num_samples": 20, "data": [bochum...]},
  ...
  "89": {"client_name": "89", "num_samples": 15, "data": [zurich...]}
}
```
- **90 clients** (5 per city × 18 cities)
- **Variable** sizes (10-75 per client)
- **Geographic** isolation (cities separate)
- 🎯 **Use for**: Location-based FL studies

---

## ✅ Validation

### Verify Format
```bash
python -m dinov2.fl.scripts.validate_settings \
    setting1_iid.json \
    setting2_noniid.json \
    setting3_city_based.json
```

Expected:
```
✅ VALIDATION PASSED
  Clients: 10
  Total samples: 2975
  Format: Matches city_partitions.json
```

---

## 🔧 Customization

### Change Alpha Values
```bash
--alpha_iid 200.0      # More IID (default: 100)
--alpha_noniid 0.05    # More heterogeneous (default: 0.1)
--alpha_city 1.0       # Less city specialization (default: 0.5)
```

### Change Client Counts
```bash
--n_clients 20              # Setting 1 & 2 (default: 10)
--n_clients_per_city 10     # Setting 3 (default: 5)
```

### Change Clustering
```bash
--n_clusters 32             # Global clusters (default: 16)
--clusters_per_city 6       # Per-city clusters (default: auto)
```

---

## 🆘 Troubleshooting

### Problem: "Dataset not found"
**Solution**: Update `BASE_PATH` in notebook cell 3:
```python
BASE_PATH = "/kaggle/input/YOUR-DATASET-NAME"
```

### Problem: "Out of memory"
**Solution**: Reduce batch size:
```bash
--batch_size 8  # Default is 16
```

### Problem: "Import errors after restart"
**Solution**: Add to first cell after restart:
```python
import sys
sys.path.insert(0, '/kaggle/working/dinov2')
```

### Problem: "Slow embedding extraction"
**Solution**: This is normal. ~15 min on T4 GPU for 2975 images.

---

## 📚 Files Reference

### Source Code
- `dinov2/fl/partitioning/export_partitions.py` - JSON export
- `dinov2/fl/partitioning/city_based.py` - City partitioning
- `dinov2/fl/scripts/generate_settings.py` - Main pipeline
- `dinov2/fl/scripts/validate_settings.py` - Validation

### Documentation
- `FL_SETTINGS_README.md` - Full documentation
- `FL_SETTINGS_SUMMARY.md` - Implementation details
- `FL_SETTINGS_QUICKSTART.md` - This file

### Notebook
- `fl_partition_settings_generator.ipynb` - Kaggle notebook

---

## 🎯 Bottom Line

**One notebook. Three settings. Zero problems.**

Upload → Run → Download → Use in FL training.

That's it. You're done. 🎉
